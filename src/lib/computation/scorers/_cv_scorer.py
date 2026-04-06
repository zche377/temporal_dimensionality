import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import xarray as xr
import torch
from joblib import Parallel, delayed

from lib.computation.classifiers import (
    Classifier,
    load_classifier,
    list_all_classifiers
)
from lib.computation.regressions import (
    load_regression,
    list_all_regressions
)
from lib.computation.scorers import Scorer, GeneralizationScorer
from lib.computation.metrics import compute_metric
from lib.utilities import _append_path, SEED
from bonner.caching import cache
from bonner.computation.regression import (
    create_stratified_splits,
    create_splits
)
from bonner.computation.regression import Regression



def _metric_scores(
    metric: str,
    score_across_folds: bool,
    n_permutations: int,
    dims: tuple[str],
    coords: dict[str, np.ndarray],
    metric_attr: dict[str, any],
    **compute_metric_kwargs,
) -> xr.DataArray:
    metric_scores = compute_metric(
        metric=metric, 
        n_permutations=n_permutations, 
        score_across_folds=score_across_folds,  
        **compute_metric_kwargs
    )
    
    if metric_scores.dim() == 1 - int(score_across_folds) + int(n_permutations is not None):
        metric_scores = metric_scores.unsqueeze(-1)
        
    return xr.DataArray(
        data=metric_scores.cpu().numpy(),
        dims=dims,
        coords=coords,
        attrs={
            "metric": metric,
            **metric_attr
        },
    )

def _model_type(model_name: str) -> str:
    if model_name in list_all_classifiers():
        return "classifier"
    elif model_name in list_all_regressions():
        return "regression"
    else:
        raise ValueError("model_name must be within the list of classifiers or regressions")
    
def _default_metrics(model_type: str) -> str:
    match model_type:
        case "classifier":
            return ["accuracy"]
        case "regression":
            return ["pearsonr"]
        case _:
            raise ValueError("model_type must be 'classifier' or 'regression'")

def _group_and_average_by_labels(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = y[:, 0]
    unique_labels, inverse_indices = torch.unique_consecutive(labels, return_inverse=True)
    n_unique = len(unique_labels)
    one_hot = torch.nn.functional.one_hot(inverse_indices, num_classes=n_unique).float()
    
    counts = one_hot.sum(dim=0, keepdim=True).t() 
    averaged_x = (one_hot.t() @ x) / counts
    averaged_y = (one_hot.t() @ y) / counts
    
    return averaged_x, averaged_y

class ModelScorer(Scorer):
    def __init__(
        self,
        model_name: str = "lda",
        metrics: (str | list[str]) = None,
        n_folds: int = 10,
        n_permutations: int = 1000,
        score_across_folds: bool = True,
        shuffle: bool = True,
        stratified: bool = None,
        average_repetition: bool = 0, 
        seed: int = SEED,
        device: torch.device | str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        train_score: bool = False,
        cache_predictors: bool = False,
        cache_subpath: str = None,
        **model_kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_type = _model_type(model_name)
        if metrics is None:
            metrics = _default_metrics(self.model_type)
        elif isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self.n_permutations = n_permutations
        self.n_folds = n_folds
        self.score_across_folds = score_across_folds
        self.shuffle = shuffle
        if stratified is None:
            stratified = self.model_type == "classifier"
        self.stratified = stratified
        if average_repetition:
            assert stratified
        self.average_repetition = average_repetition
        self.seed = seed
        self.model_kwargs = model_kwargs
        if self.model_name.split("_")[0] == "sklearn":
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.train_score = train_score
        self.cache_predictors = cache_predictors
        identifier = _append_path(f"{self.model_type}.{model_name}", "model_kwargs", model_kwargs)
        identifier += (
            f"/n_folds={self.n_folds}"
            f".metrics={self.metrics}"
            f".score_across_folds={self.score_across_folds}"
            f".shuffle={self.shuffle}"
            f".stratified={self.stratified}"
            f".average_repetition={self.average_repetition}"
            f".seed={self.seed}"
        )
        self.metric_attr = {
            "model": self.model_name,
            "n_folds": self.n_folds,
            "score_across_folds": int(self.score_across_folds),
            "n_permutations": str(self.n_permutations),
            "shuffle": int(self.shuffle),
            "stratified": int(self.stratified),
            "average_repetition": int(self.average_repetition),
            "seed": self.seed,
            **self.model_kwargs
        }
        super().__init__(identifier=identifier)
        if self.cache_predictors:
            assert cache_subpath is not None
            self.cache_path = f"scorers/{identifier}/{cache_subpath}"
        else:
            self.cache_path = None

    def _fit_model(
        self,
        *,
        x_train: torch.Tensor, 
        y_train: torch.Tensor,
    ) -> Classifier | Regression:
        match self.model_type:
            case "classifier":
                model = load_classifier(self.model_name, **self.model_kwargs, device=self.device)
            case "regression":
                model = load_regression(self.model_name, **self.model_kwargs, device=self.device)
            case _:
                raise ValueError("model_type must be 'classifier' or 'regression'")
        
        model.fit(x_train, y_train)
        return model
        
    def _compute_predictions(
        self,
        *,
        predictor: xr.DataArray,
        target: xr.DataArray,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x = torch.tensor(predictor.values).float().to(self.device)
        y = torch.tensor(target.values).float().to(self.device)
        if self.stratified:
            splits = create_stratified_splits(y=y.cpu().numpy(), n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
        else:
            splits = create_splits(n=y.size(0), n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
            
        y_true, y_predicted = [], []
        
        for i_split, indices_test in enumerate(splits):
            indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
            
            x_train, x_test = x[..., indices_train, :], x[..., indices_test, :]
            y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]
            
            if self.average_repetition:
                x_train, y_train = _group_and_average_by_labels(x_train, y_train)
                x_test, y_test = _group_and_average_by_labels(x_test, y_test)
            
            cacher = cache(
                f"{self.cache_path}/split={i_split}.pkl",
                mode = "normal" if self.cache_predictors else "ignore",
            )
            model = cacher(self._fit_model)(
                x_train=x_train, y_train=y_train
            )
            if self.model_name.split("_")[0] != "sklearn":
                model.to(self.device)
              
            if self.train_score:
                y_true.append(y_train)
                y_predicted.append(model.predict(x_train))
            else:
                y_true.append(y_test)
                y_predicted.append(model.predict(x_test))
        
        if self.cache_predictors:
            cacher = cache(f"{self.cache_path}/split=full.pkl")
            cacher(self._fit_model)(x_train=x, y_train=y)
        
        return y_true, y_predicted
    
    def _score(
        self,
        *,
        predictor: xr.DataArray,
        target: xr.DataArray,
        target_dim: str,
    ) -> xr.Dataset:
        y_true, y_predicted = self._compute_predictions(
            predictor=predictor,
            target=target,
        )
        scores = {}
        dims = set([target_dim]) if self.score_across_folds else ("fold", target_dim)
        coords = {target_dim: target[target_dim].values}
        for metric in self.metrics:
            scores[metric] = _metric_scores(
                metric=metric,
                score_across_folds=self.score_across_folds,
                n_permutations=None,
                dims=dims,
                coords=coords,
                metric_attr=self.metric_attr,
                y_true=y_true,
                y_predicted=y_predicted,
                device=self.device,
            )
            if self.n_permutations is not None:
                scores[f"null.{metric}.n_permutations={self.n_permutations}"] = _metric_scores(
                    metric=metric,
                    score_across_folds=self.score_across_folds,
                    n_permutations=self.n_permutations,
                    dims=("permutation", *dims),
                    coords=coords,
                    metric_attr=self.metric_attr,
                    y_true=y_true,
                    y_predicted=y_predicted,
                    device=self.device,
                )
        return xr.Dataset(scores)
    
class ModelGeneralizationScorer(GeneralizationScorer):
    def __init__(
        self,
        model_name: str = "lda",
        metrics: (str | list[str]) = None,
        n_folds: int = 10,
        n_permutations: int = None,
        score_across_folds: bool = True,
        shuffle: bool = True,
        stratified: bool = None,
        seed: int = SEED,
        device: torch.device | str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        train_score: bool = False,
        cache_predictors: bool = False,
        cache_subpath: str = None,
        **model_kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_type = _model_type(model_name)
        if metrics is None:
            metrics = _default_metrics(self.model_type)
        elif isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self.n_permutations = n_permutations
        self.n_folds = n_folds
        self.score_across_folds = score_across_folds
        self.shuffle = shuffle
        if stratified is None:
            stratified = self.model_type == "classifier"
        self.stratified = stratified
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.device = device
        self.train_score = train_score
        self.cache_predictors = cache_predictors
        identifier = _append_path(f"{self.model_type}.{model_name}", "model_kwargs", model_kwargs)
        identifier += (
            f"/n_folds={self.n_folds}"
            f".metrics={self.metrics}"
            f".score_across_folds={self.score_across_folds}"
            f".shuffle={self.shuffle}"
            f".stratified={self.stratified}"
            f".seed={self.seed}"
        )
        self.metric_attr = {
            "model": self.model_name,
            "n_folds": self.n_folds,
            "score_across_folds": int(self.score_across_folds),
            "n_permutations": str(self.n_permutations),
            "shuffle": int(self.shuffle),
            "stratified": int(self.stratified),
            "seed": self.seed,
            **self.model_kwargs
        }
        super().__init__(identifier=identifier)
        if self.cache_predictors:
            assert cache_subpath is not None
            self.cache_path = f"scorers/{identifier}/{cache_subpath}"
        else:
            self.cache_path = None


    def _fit_model(
        self,
        *,
        x_train: torch.Tensor, 
        y_train: torch.Tensor,
    ) -> Classifier | Regression:
        match self.model_type:
            case "classifier":
                model = load_classifier(self.model_name, **self.model_kwargs, device=self.device)
            case "regression":
                model = load_regression(self.model_name, **self.model_kwargs, device=self.device)
            case _:
                raise ValueError("model_type must be 'classifier' or 'regression'")
        
        model.fit(x_train, y_train)
        return model

    def _compute_predictions(
        self,
        *,
        fitting_predictor_var: str,
        predictors: xr.DataArray,
        target: xr.DataArray,
        predictor_dim: str,
    ) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        x = torch.tensor(predictors.sel({predictor_dim: fitting_predictor_var}).values).float().to(self.device)
        y = torch.tensor(target.values).float().to(self.device)
        if self.stratified:
            splits = create_stratified_splits(y=y.cpu().numpy(), n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
        else:
            splits = create_splits(n=y.size(0), n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
            
        y_true = []
        for indices_test in splits:
            indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
            if self.train_score:
                y_true.append(y[..., indices_train, :])
            else:
                y_true.append(y[..., indices_test, :])
        
        def _ys_predicted(predictor_var):
            split_scores = []
            for i_split, indices_test in enumerate(splits):
                indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
                
                x_train = x[..., indices_train, :]
                y_train= y[..., indices_train, :]
                
                cacher = cache(
                    f"{self.cache_path}/predictor={predictor_dim}={fitting_predictor_var}/split={i_split}.pkl",
                    mode="normal" if self.cache_predictors else "ignore",
                )
                model = cacher(self._fit_model)(
                    x_train=x_train, y_train=y_train
                )
                if self.model_name.split("_")[0] != "sklearn":
                    model.to(self.device)
                
                x_pred = torch.tensor(predictors.sel({predictor_dim: predictor_var}).values).float().to(self.device)
                
                if self.train_score:
                    temp = model.predict(x_pred[..., indices_train, :])
                else:
                    temp = model.predict(x_pred[..., indices_test, :])
                split_scores.append(temp)
            return split_scores
        
        ys_predicted = Parallel(n_jobs=1, backend="loky")(
            delayed(_ys_predicted)(predictor_var) for predictor_var in predictors[predictor_dim].values
        )

        if self.cache_predictors:
            cacher = cache(f"{self.cache_path}/predictor={predictor_dim}={fitting_predictor_var}/split=full.pkl")
            cacher(self._fit_model)(x_train=x, y_train=y)
            
        return y_true, ys_predicted
    
    def _score(
        self,
        *,
        predictors: xr.DataArray,
        target: xr.DataArray,
        target_dim: str,
        predictor_dim: str,
    ) -> xr.Dataset:
        predictor_scores = []
        for predictor_var in predictors[predictor_dim].values:
            # logging.info(predictor_var)
            y_true, ys_predicted = self._compute_predictions(
                fitting_predictor_var=predictor_var,
                predictors=predictors,
                target=target,
                predictor_dim=predictor_dim,
            )
            scores = {}
            generalization_dim = f"{predictor_dim}_generalization"
            dims = set([target_dim]) if self.score_across_folds else ("fold", target_dim)
            coords = {target_dim: target[target_dim].values}
            for metric in self.metrics:
                predictor_metric_scores = Parallel(n_jobs=-1, backend="loky")(
                    delayed(_metric_scores)(
                        metric=metric,
                        score_across_folds=self.score_across_folds,
                        n_permutations=None,
                        dims=dims,
                        coords=coords,
                        metric_attr=self.metric_attr,
                        y_true=y_true,
                        y_predicted=y_predicted,
                        device=self.device,
                    ) for y_predicted in ys_predicted
                )
                predictor_metric_scores = [predictor_metric_scores[i].assign_coords({generalization_dim: key}) for i, key in enumerate(predictors[predictor_dim].values)]
                scores[metric] = xr.concat(predictor_metric_scores, dim=generalization_dim)
                if self.n_permutations is not None:
                    predictor_metric_scores = Parallel(n_jobs=-1, backend="loky")(
                        delayed(_metric_scores)(
                            metric=metric,
                                score_across_folds=self.score_across_folds,
                                n_permutations=self.n_permutations,
                                dims=("permutation", *dims),
                                coords=coords,
                                metric_attr=self.metric_attr,
                                y_true=y_true,
                                y_predicted=y_predicted,
                                device=self.device,
                        ) for y_predicted in ys_predicted
                    )
                    predictor_metric_scores = [predictor_metric_scores[i].assign_coords({generalization_dim: key}) for i, key in enumerate(predictors[predictor_dim].values)]
                    scores[f"null.{metric}.n_permutations={self.n_permutations}"] = xr.concat(predictor_metric_scores, dim=generalization_dim)
            predictor_scores.append(xr.Dataset(scores).assign_coords({predictor_dim: predictor_var}))
        
        return xr.concat(predictor_scores, dim=predictor_dim)  
    
class RotInvGeneralizationScorer(GeneralizationScorer):
    def __init__(
        self,
        model_name: str = "lda",
        metrics: (str | list[str]) = None,
        n_folds: int = 10,
        n_permutations: int = None,
        score_across_folds: bool = True,
        shuffle: bool = True,
        stratified: bool = None,
        seed: int = SEED,
        device: torch.device | str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        train_score: bool = False,
        cache_predictors: bool = False,
        cache_subpath: str = None,
        **model_kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_type = _model_type(model_name)
        if metrics is None:
            metrics = _default_metrics(self.model_type)
        elif isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self.n_permutations = n_permutations
        self.n_folds = n_folds
        self.score_across_folds = score_across_folds
        self.shuffle = shuffle
        if stratified is None:
            stratified = self.model_type == "classifier"
        self.stratified = stratified
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.device = device
        self.train_score = train_score
        self.cache_predictors = cache_predictors
        identifier = _append_path(f"{self.model_type}.{model_name}", "model_kwargs", model_kwargs)
        identifier += (
            f"/n_folds={self.n_folds}"
            f".metrics={self.metrics}"
            f".score_across_folds={self.score_across_folds}"
            f".shuffle={self.shuffle}"
            f".stratified={self.stratified}"
            f".seed={self.seed}"
        )
        self.metric_attr = {
            "model": self.model_name,
            "n_folds": self.n_folds,
            "score_across_folds": int(self.score_across_folds),
            "n_permutations": str(self.n_permutations),
            "shuffle": int(self.shuffle),
            "stratified": int(self.stratified),
            "seed": self.seed,
            **self.model_kwargs
        }
        super().__init__(identifier=identifier)
        if self.cache_predictors:
            assert cache_subpath is not None
            self.cache_path = f"scorers/{identifier}/{cache_subpath}"
        else:
            self.cache_path = None


    def _fit_model(
        self,
        *,
        x_train: torch.Tensor, 
        y_train: torch.Tensor,
        model_type: str = None,
    ) -> Classifier | Regression:
        if model_type is None:
            model_type = self.model_type
        match model_type:
            case "classifier":
                model = load_classifier(self.model_name, **self.model_kwargs, device=self.device)
            case "regression":
                model = load_regression(self.model_name, **self.model_kwargs, device=self.device)
            case _:
                raise ValueError("model_type must be 'classifier' or 'regression'")
        model.fit(x_train, y_train)
        return model

    def _compute_predictions(
        self,
        *,
        fitting_predictor_var: str,
        predictors: xr.DataArray,
        target: xr.DataArray,
        predictor_dim: str,
    ) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        x = torch.tensor(predictors.sel({predictor_dim: fitting_predictor_var}).values).float().to(self.device)
        y = torch.tensor(target.values).float().to(self.device)
        if self.stratified:
            splits = create_stratified_splits(y=y.cpu().numpy(), n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
        else:
            splits = create_splits(n=y.size(0), n_folds=self.n_folds, shuffle=self.shuffle, seed=self.seed)
        
        y_true = []
        for indices_test in splits:
            indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
            y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]
            if self.train_score:
                y_true.append(y_train)
            else:
                y_true.append(y_test)
        
        def _ys_predicted(predictor_var):
            split_scores = []
            for i_split, indices_test in enumerate(splits):
                indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
                
                x_train = x[..., indices_train, :]
                y_train= y[..., indices_train, :]
                
                cacher = cache(
                    f"{self.cache_path}/predictor={predictor_dim}={fitting_predictor_var}/split={i_split}.pkl",
                    mode="normal" if self.cache_predictors else "ignore",
                )
                model = cacher(self._fit_model)(
                    x_train=x_train, y_train=y_train
                )
                if self.model_name.split("_")[0] != "sklearn":
                    model.to(self.device)
                
                if fitting_predictor_var != predictor_var:
                    x_train_pred = torch.tensor(predictors.sel({predictor_dim: predictor_var}).values).float().to(self.device)[..., indices_train, :]
                    model_pred = self._fit_model(
                        x_train=x_train_pred, y_train=x_train, model_type="regression"
                    )
                    
                    if self.train_score:
                        temp = model.predict(model_pred.predict(x_train_pred))
                    else:
                        temp = model.predict(model_pred.predict(torch.tensor(predictors.sel({predictor_dim: predictor_var}).values).float().to(self.device)[..., indices_test, :]))
                else:
                    if self.train_score:
                        temp = model.predict(x_train)
                    else:
                        temp = model.predict(x[..., indices_test, :])
                split_scores.append(temp)
            return split_scores
        
        ys_predicted = Parallel(n_jobs=1, backend="loky")(
            delayed(_ys_predicted)(predictor_var) for predictor_var in predictors[predictor_dim].values
        )
            
        return y_true, ys_predicted
    
    def _score(
        self,
        *,
        predictors: xr.DataArray,
        target: xr.DataArray,
        target_dim: str,
        predictor_dim: str,
    ) -> xr.Dataset:
        predictor_scores = []
        for predictor_var in predictors[predictor_dim].values:
            logging.info(predictor_var)
            y_true, ys_predicted = self._compute_predictions(
                fitting_predictor_var=predictor_var,
                predictors=predictors,
                target=target,
                predictor_dim=predictor_dim,
            )
            scores = {}
            generalization_dim = f"{predictor_dim}_generalization"
            dims = set([target_dim]) if self.score_across_folds else ("fold", target_dim)
            coords = {target_dim: target[target_dim].values}
            for metric in self.metrics:
                predictor_metric_scores = Parallel(n_jobs=-1, backend="loky")(
                    delayed(_metric_scores)(
                        metric=metric,
                        score_across_folds=self.score_across_folds,
                        n_permutations=None,
                        dims=dims,
                        coords=coords,
                        metric_attr=self.metric_attr,
                        y_true=y_true,
                        y_predicted=y_predicted,
                        device=self.device,
                    ) for y_predicted in ys_predicted
                )
                predictor_metric_scores = [predictor_metric_scores[i].assign_coords({generalization_dim: key}) for i, key in enumerate(predictors[predictor_dim].values)]
                scores[metric] = xr.concat(predictor_metric_scores, dim=generalization_dim)
                if self.n_permutations is not None:
                    predictor_metric_scores = Parallel(n_jobs=-1, backend="loky")(
                        delayed(_metric_scores)(
                            metric=metric,
                                score_across_folds=self.score_across_folds,
                                n_permutations=self.n_permutations,
                                dims=("permutation", *dims),
                                coords=coords,
                                metric_attr=self.metric_attr,
                                y_true=y_true,
                                y_predicted=y_predicted,
                                device=self.device,
                        ) for y_predicted in ys_predicted
                    )
                    predictor_metric_scores = [predictor_metric_scores[i].assign_coords({generalization_dim: key}) for i, key in enumerate(predictors[predictor_dim].values)]
                    scores[f"null.{metric}.n_permutations={self.n_permutations}"] = xr.concat(predictor_metric_scores, dim=generalization_dim)
            predictor_scores.append(xr.Dataset(scores).assign_coords({predictor_dim: predictor_var}))
        
        return xr.concat(predictor_scores, dim=predictor_dim)  
