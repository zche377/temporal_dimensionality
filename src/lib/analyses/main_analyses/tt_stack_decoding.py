import shutil
import logging

logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import torch
from joblib import Parallel, delayed
from copy import deepcopy

from lib.analyses._utilities import _cache_path
from lib.computation.scorers import TrainTestModelScorer, TrainTestPLSSVDScorer
from lib.datasets import load_dataset, load_target_var
from lib.utilities import (
    _append_path,
    SEED,
)
from lib.datasets.hebart2022_things_behavior import sort_embeddings
from lib.models import DJModel
from lib.analyses.loaders import load_significant_times

from bonner.computation.decomposition import PCA
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)



def _model_srp_decoding(
    dataset: str,
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    pca: bool,
    model_uid: str,
    n_components,
) -> xr.Dataset:
    subject_data_train, subject_data_test = data_train.sel(subject=subject), data_test.sel(subject=subject)
    
    model = DJModel(model_uid, hook="srp", n_components=n_components)
    target = model(dataset, dataloader_kwargs={"batch_size": 32})
    
    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer
        
    scores = []
    i = 0
    for node, feature_map in target.items():
        # TEST: only run the first and last node
        i+=1
        if i not in [1, len(target)]:
            continue
        feature_map_train = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_train.img_files)))
        feature_map_test = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_test.img_files)))
        
        if pca:
            pmodel = PCA()
            pmodel.fit(torch.tensor(feature_map_train.values))
            feature_map_train = feature_map_train.copy(data=pmodel.transform(torch.tensor(feature_map_train.values)))
            feature_map_test = feature_map_test.copy(data=pmodel.transform(torch.tensor(feature_map_test.values)))
            feature_map_train["neuroid"] = np.arange(len(feature_map_train.neuroid))+1
            feature_map_test["neuroid"] = np.arange(len(feature_map_test.neuroid))+1
            
        time_scores = scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=node={node}/predictor=time=stack",
        )(
            predictor_train=subject_data_train,
            target_train=feature_map_train,
            predictor_test=subject_data_test,
            target_test=feature_map_test,
            target_dim="neuroid",
        )
    
        scores.append(time_scores.assign_coords({"node": node}))

    return xr.concat(scores, dim="node").assign_coords({"subject": subject})

def _behavior_decoding(
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    subset: bool,
    pca: bool,
) -> xr.Dataset:
    subject_data_train, subject_data_test = data_train.sel(subject=subject), data_test.sel(subject=subject)

    if subset:
        embd = sort_embeddings()
    else:
        embd = load_embeddings()
    target_train = embd.sel(object=subject_data_train.presentation.values)
    target_test = embd.sel(object=subject_data_test.presentation.values)
    
    if pca:
        pmodel = PCA()
        pmodel.fit(torch.tensor(target_train.values))
        target_train = target_train.copy(data=pmodel.transform(torch.tensor(target_train.values)))
        target_test = target_test.copy(data=pmodel.transform(torch.tensor(target_test.values)))
        target_train["behavior"] = np.arange(len(target_train.behavior))+1
        target_test["behavior"] = np.arange(len(target_test.behavior))+1
    
    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer
    
    time_scores = scorer_fn(
        **scorer_kwargs,
        cache_predictors=True,
        cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=time=stack",
    )(
        predictor_train=subject_data_train,
        target_train=target_train,
        predictor_test=subject_data_test,
        target_test=target_test,
        target_dim="behavior",
    )
    
    return time_scores.assign_coords({"subject": subject})

def _reshape(X, stack_dims, target_var):
    return (X
        .mean("presentation")
        .rename({target_var: "presentation"})
        .stack(stack=[stack_dims[0], stack_dims[1]])
        .reset_index("stack")
        .drop_vars(stack_dims[1])
        .rename({"stack": stack_dims[1]})
        .transpose("subject", "presentation", stack_dims[1])
    )

def tt_stack_decoding(
    analysis: str,
    dataset: str,
    subject: int,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    stack_dims: set[str, str], # "time", "neuroid"
    subset: bool = False,
    pca: bool = False,
    model_kwargs: dict = None,
    time: str = "significant", # significant or control
    alpha: float = 0.05,
) -> xr.Dataset:
    cache_str = f"main_analyses/{analysis}_tt_stack_decoding.stack_dims={stack_dims[0]}_{stack_dims[1]}.subset={subset}.pca={pca}"
        
    cache_path = _cache_path(
        cache_str,
        dataset,
        load_dataset_kwargs,
        scorer_kwargs,
        include_root=False,
    )
    predictor_cache_path = f"{cache_str}/dataset={dataset}"
    predictor_cache_path = _append_path(
        predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs
    )

    match analysis:
        case "behavior":
            analysis_fn = _behavior_decoding
            cache_kwargs = {"subset": subset}
        case "model_srp":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srp_decoding
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
        case _:
            raise ValueError(
                "analysis not implemented"
            )
            
    target_var = load_target_var(f"{dataset}_train")
    
    data_train = load_dataset(
        f"{dataset}_train",
        subjects=subject,
        **load_dataset_kwargs,
    )
    data_test = load_dataset(
        f"{dataset}_test",
        subjects=subject,
        **load_dataset_kwargs,
    )
    
    # TODO: significant time
    if "time" in [*stack_dims]:
        if analysis not in ["behavior", "bhvpc", "bhvsubset"]:
            feature_dim = target_var
        else:
            feature_dim = "behavior"
        
        match time:
            case "significant":
                temp_scorer_kwargs = {"model_name": "linear", "l2_penalty": 1e-2}
                times = load_significant_times(f"{analysis}_tt_decoding.subset=False.pca=False", dataset, load_dataset_kwargs, temp_scorer_kwargs, feature_dim, alpha, subjects="intersection").mean("subject").dropna("time").values
                times = np.round(times, 2)
                
                data_train = data_train.sel(time=[t in times for t in data_train.time.values])
                data_test = data_test.sel(time=[t in times for t in data_test.time.values])
            case "control":
                data_train = data_train.sel(time=[t<0 for t in data_train.time.values])
                data_test = data_test.sel(time=[t<0 for t in data_test.time.values])
    
    data_train = _reshape(data_train, stack_dims, target_var)
    data_test = _reshape(data_test, stack_dims, target_var)
    
    if 'freq' not in data_train.coords:
        subject_path = f"{cache_path}/subject={subject}.nc"
        cacher = cache(
            subject_path,
            # mode="ignore",
        )
        score = cacher(analysis_fn)(
            data_train=data_train,
            data_test=data_test,
            subject=subject,
            scorer_kwargs=scorer_kwargs,
            predictor_cache_path=f"{predictor_cache_path}/subject={subject}",
            pca=pca,
            **cache_kwargs,
        )
        
        return xr.concat([score], dim="subject")
    else:
        scores = []
        for f in tqdm(data_train.freq.values, desc="freq"):
            subject_path = f"{cache_path}/freq={f}/subject={subject}.nc"
            cacher = cache(
                subject_path,
                # mode="ignore",
            )
            scores.append(cacher(analysis_fn)(
                data_train=data_train.sel(freq=f),
                data_test=data_test.sel(freq=f),
                subject=subject,
                scorer_kwargs=scorer_kwargs,
                predictor_cache_path=f"{predictor_cache_path}/freq={f}/subject={subject}",
                subset=subset,
                pca=pca,
                **cache_kwargs,
            ))

        return xr.concat([xr.concat(scores, dim="freq")], dim="subject")
