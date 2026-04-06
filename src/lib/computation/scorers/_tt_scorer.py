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
from lib.computation.scorers import TrainTestScorer,TrainTestGeneralizationScorer
from lib.computation.metrics import compute_metric
from lib.utilities import _append_path, SEED
from bonner.caching import cache
from bonner.computation.regression import Regression
from bonner.computation.decomposition import PLSSVD
from bonner.computation.metrics import covariance, pearson_r



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
        case "regression" | "plssvd":
            return ["pearsonr"]
        case _:
            raise ValueError("model_type must be 'classifier' or 'regression'")

def _group_and_average_by_labels(x: torch.Tensor, y: torch.Tensor, average_by_y: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    labels = y[:, 0] if average_by_y else x[:, 0]
    unique_labels, inverse_indices = torch.unique_consecutive(labels, return_inverse=True)
    n_unique = len(unique_labels)
    one_hot = torch.nn.functional.one_hot(inverse_indices, num_classes=n_unique).float()
    
    counts = one_hot.sum(dim=0, keepdim=True).t() 
    averaged_x = (one_hot.t() @ x) / counts if x is not None else None
    averaged_y = (one_hot.t() @ y) / counts
    
    return averaged_x, averaged_y

class TrainTestModelScorer(TrainTestScorer):
    def __init__(
        self,
        model_name: str = "lda",
        metrics: (str | list[str]) = None,
        n_permutations: int = 1000,
        average_repetition: bool = 0, 
        average_by_y: bool= True,
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
        self.average_repetition = average_repetition
        self.average_by_y = average_by_y
        self.model_kwargs = model_kwargs
        if self.model_name.split("_")[0] == "sklearn":
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.train_score = train_score
        self.cache_predictors = cache_predictors
        identifier = _append_path(f"{self.model_type}.{model_name}", "model_kwargs", model_kwargs)
        identifier += (
            f"/metrics={self.metrics}"
            f".average_repetition={self.average_repetition}"
        )
        if not self.average_by_y:
            identifier += (
                f".average_by_y={self.average_by_y}"
            )
        self.metric_attr = {
            "model": self.model_name,
            "n_permutations": str(self.n_permutations),
            "average_repetition": int(self.average_repetition),
            "average_by_y": int(self.average_by_y),
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
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x_train = torch.tensor(predictor_train.values).float().to(self.device)
        y_train = torch.tensor(target_train.values).float().to(self.device)
        x_test = torch.tensor(predictor_test.values).float().to(self.device)
        y_test = torch.tensor(target_test.values).float().to(self.device)
        
        y_true, y_predicted = [], []
        
        if self.average_repetition:
            x_train, y_train = _group_and_average_by_labels(x_train, y_train, self.average_by_y)
            x_test, y_test = _group_and_average_by_labels(x_test, y_test, self.average_by_y)
            
        cacher = cache(
            f"{self.cache_path}.pkl",
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
        
        return y_true, y_predicted   
        
    def _score(
        self,
        *,
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        target_dim: str,
    ) -> xr.Dataset:
        y_true, y_predicted = self._compute_predictions(
            predictor_train=predictor_train,
            target_train=target_train,
            predictor_test=predictor_test,
            target_test=target_test
        )
        scores = {}
        dims = set([target_dim])
        coords = {target_dim: target_test[target_dim].values}
        for metric in self.metrics:
            scores[metric] = _metric_scores(
                metric=metric,
                score_across_folds=True,
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
                    score_across_folds=True,
                    n_permutations=self.n_permutations,
                    dims=("permutation", *dims),
                    coords=coords,
                    metric_attr=self.metric_attr,
                    y_true=y_true,
                    y_predicted=y_predicted,
                    device=self.device,
                )
        return xr.Dataset(scores)    

class TrainTestModelGeneralizationScorer(TrainTestGeneralizationScorer):
    def __init__(
        self,
        model_name: str = "lda",
        metrics: (str | list[str]) = None,
        n_permutations: int = 1000,
        average_repetition: bool = 0, 
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
        self.average_repetition = average_repetition
        self.model_kwargs = model_kwargs
        if self.model_name.split("_")[0] == "sklearn":
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.train_score = train_score
        self.cache_predictors = cache_predictors
        identifier = _append_path(f"{self.model_type}.{model_name}", "model_kwargs", model_kwargs)
        identifier += (
            f"/metrics={self.metrics}"
            f".average_repetition={self.average_repetition}"
        )
        self.metric_attr = {
            "model": self.model_name,
            "n_permutations": str(self.n_permutations),
            "average_repetition": int(self.average_repetition),
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
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        predictor_dim: str,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x_train = torch.tensor(predictor_train.sel({predictor_dim: fitting_predictor_var}).values).float().to(self.device)
        y_train = torch.tensor(target_train.values).float().to(self.device)
        y_test = torch.tensor(target_test.values).float().to(self.device)
        
        if self.average_repetition:
            x_train, y_train = _group_and_average_by_labels(x_train, y_train)
            _, y_test_true = _group_and_average_by_labels(None, y_test)
        else:
            y_test_true = y_test
        
        y_true = [y_train] if self.train_score else [y_test_true]
        
        def _ys_predicted(predictor_var):
            cacher = cache(
                f"{self.cache_path}/predictor={predictor_dim}={fitting_predictor_var}.pkl",
                mode = "normal" if self.cache_predictors else "ignore",
            )
            model = cacher(self._fit_model)(
                x_train=x_train, y_train=y_train
            )
            if self.model_name.split("_")[0] != "sklearn":
                model.to(self.device)
                
            x_pred = torch.tensor(predictor_test.sel({predictor_dim: predictor_var}).values).float().to(self.device)
            
            if self.average_repetition:
                x_pred, _ = _group_and_average_by_labels(x_pred, y_test)
                
            return [model.predict(x_train if self.train_score else x_pred)]
        
        ys_predicted = Parallel(n_jobs=1, backend="loky")(
            delayed(_ys_predicted)(predictor_var) for predictor_var in predictor_test[predictor_dim].values
        )
        
        return y_true, ys_predicted   
    
    def _score_aux(
        self,
        *,
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        target_dim: str,
        predictor_dim: str,
        predictor_var: float,
    ) -> xr.Dataset:
        y_true, ys_predicted = self._compute_predictions(
            fitting_predictor_var=predictor_var,
            predictor_train=predictor_train,
            target_train=target_train,
            predictor_test=predictor_test,
            target_test=target_test,
            predictor_dim=predictor_dim,
        )
        scores = {}
        generalization_dim = f"{predictor_dim}_generalization"
        dims = set([target_dim])
        coords = {target_dim: target_test[target_dim].values}
        for metric in self.metrics:
            predictor_metric_scores = Parallel(n_jobs=1, backend="loky")(
                delayed(_metric_scores)(
                    metric=metric,
                    score_across_folds=True,
                    n_permutations=None,
                    dims=dims,
                    coords=coords,
                    metric_attr=self.metric_attr,
                    y_true=y_true,
                    y_predicted=y_predicted,
                    device=self.device,
                ) for y_predicted in ys_predicted
            )
            predictor_metric_scores = [predictor_metric_scores[i].assign_coords({generalization_dim: key}) for i, key in enumerate(predictor_test[predictor_dim].values)]
            scores[metric] = xr.concat(predictor_metric_scores, dim=generalization_dim)
            
            if self.n_permutations is not None:
                predictor_metric_scores = Parallel(n_jobs=1, backend="loky")(
                    delayed(_metric_scores)(
                        metric=metric,
                            score_across_folds=True,
                            n_permutations=self.n_permutations,
                            dims=("permutation", *dims),
                            coords=coords,
                            metric_attr=self.metric_attr,
                            y_true=y_true,
                            y_predicted=y_predicted,
                            device=self.device,
                    ) for y_predicted in ys_predicted
                )
                predictor_metric_scores = [predictor_metric_scores[i].assign_coords({generalization_dim: key}) for i, key in enumerate(predictor_test[predictor_dim].values)]
                scores[f"null.{metric}.n_permutations={self.n_permutations}"] = xr.concat(predictor_metric_scores, dim=generalization_dim)
        return xr.Dataset(scores).assign_coords({predictor_dim: predictor_var})
        
    def _score(
        self,
        *,
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        target_dim: str,
        predictor_dim: str,
    ) -> xr.Dataset:
        # predictor_scores = Parallel(n_jobs=2, backend="loky")(
        #     delayed(self._score_aux)(
        #         predictor_train=predictor_train,
        #         target_train=target_train,
        #         predictor_test=predictor_test,
        #         target_test=target_test,
        #         target_dim=target_dim,
        #         predictor_dim=predictor_dim,
        #         predictor_var=predictor_var,
        #     ) for predictor_var in predictor_train[predictor_dim].values
        # )
        
        predictor_scores = []
        for predictor_var in predictor_train[predictor_dim].values:
            logging.info(f"Scoring for {predictor_dim}={predictor_var}")
            predictor_scores.append(self._score_aux(
                predictor_train=predictor_train,
                target_train=target_train,
                predictor_test=predictor_test,
                target_test=target_test,
                target_dim=target_dim,
                predictor_dim=predictor_dim,
                predictor_var=predictor_var,
            ))
        return xr.concat(predictor_scores, dim=predictor_dim)  

class TrainTestPLSSVDScorer(TrainTestScorer):
    def __init__(
        self,
        metrics: (str | list[str]) = None,
        n_permutations: int = 1000,
        average_repetition: bool = 0, 
        score_space: str = "latent",
        device: torch.device | str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        train_score: bool = False,
        cache_predictors: bool = False,
        cache_subpath: str = None,
        **kwargs
    ) -> None:
        if metrics is None:
            metrics = _default_metrics("plssvd")
        elif isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self.n_permutations = n_permutations
        self.average_repetition = average_repetition
        self.score_space = score_space
        self.device = device
        self.train_score = train_score
        self.cache_predictors = cache_predictors
        identifier = (
            "plssvd"
            f"/metrics={self.metrics}"
            f".average_repetition={self.average_repetition}"
            f".score_space={self.score_space}"
        )
        self.metric_attr = {
            "model": "plssvd",
            "n_permutations": str(self.n_permutations),
            "average_repetition": int(self.average_repetition),
            "score_space": score_space,
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
    ) -> PLSSVD:
        model = PLSSVD(randomized=False)
        model.fit(x_train, y_train)
        return model
    
    def _compute_predictions(
        self,
        *,
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x_train = torch.tensor(predictor_train.values).float().to(self.device)
        y_train = torch.tensor(target_train.values).float().to(self.device)
        x_test = torch.tensor(predictor_test.values).float().to(self.device)
        y_test = torch.tensor(target_test.values).float().to(self.device)
        
        y_true, y_predicted = [], []
        
        if self.average_repetition:
            x_train, y_train = _group_and_average_by_labels(x_train, y_train)
            x_test, y_test = _group_and_average_by_labels(x_test, y_test)
            
        cacher = cache(
            f"{self.cache_path}.pkl",
            mode = "normal" if self.cache_predictors else "ignore",
        )
        model = cacher(self._fit_model)(
            x_train=x_train, y_train=y_train
        )
        model.to(self.device)
        
        if self.train_score:
            x, y = x_train, y_train
        else:
            x, y = x_test, y_test
        
        match self.score_space:
            case "latent":
                y_true.append(model.transform(y, direction="right"))
                y_predicted.append(model.transform(x, direction="left"))
            case "y":
                y_true.append(y)
                y_predicted.append(model.inverse_transform(model.transform(x, direction="left"), direction="right"))
            case "x":
                y_true.append(model.inverse_transform(model.transform(y, direction="right"), direction="left"))
                y_predicted.append(x)
        
        return y_true, y_predicted   
    
    def _score(
        self,
        *,
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        target_dim: str,
    ) -> xr.Dataset:
        y_true, y_predicted = self._compute_predictions(
            predictor_train=predictor_train,
            target_train=target_train,
            predictor_test=predictor_test,
            target_test=target_test
        )
        scores = {}
        match self.score_space:
            case "latent":
                dims = set(["latent"])
                coords = {"latent": np.arange(y_true[0].size(-1))}
            case "y":
                dims = set([target_dim])
                coords = {target_dim: target_test[target_dim].values}
            case "x":
                dims = set(["x"])
                coords = {"neuroid": predictor_test["neuroid"].values}
        for metric in self.metrics:
            scores[metric] = _metric_scores(
                metric=metric,
                score_across_folds=True,
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
                    score_across_folds=True,
                    n_permutations=self.n_permutations,
                    dims=("permutation", *dims),
                    coords=coords,
                    metric_attr=self.metric_attr,
                    y_true=y_true,
                    y_predicted=y_predicted,
                    device=self.device,
                )
        return xr.Dataset(scores)    
     