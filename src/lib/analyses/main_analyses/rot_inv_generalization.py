import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
import xarray as xr
import torch

from lib.analyses._utilities import _cache_path
from lib.analyses.main_analyses.decoding import _reshape_subject_data
from lib.computation.scorers import RotInvGeneralizationScorer
from lib.datasets import (
    load_dataset,
    load_target_var
)
from lib.datasets.hebart2022_things_behavior import sort_embeddings
from lib.utilities import _append_path

from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import cache



def _behavior_generalization(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values
    
    embd = load_embeddings()
    target = embd.sel(object=subject_var_values)
    
    return RotInvGeneralizationScorer(
        **scorer_kwargs,
        cache_predictors=True,
        cache_subpath=f"{predictor_cache_path}/target=behavior=all",
    )(
        predictors=subject_data,
        target=target,
        target_dim="behavior",
        predictor_dim="time",
    ).assign_coords({"subject": subject})
    
    
def _behavior_subset_generalization(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values
    
    embd = sort_embeddings()
    target = embd.sel(object=subject_var_values)
    
    return RotInvGeneralizationScorer(
        **scorer_kwargs,
        cache_predictors=True,
        cache_subpath=f"{predictor_cache_path}/target=behavior=sortby=std.ntop=8",
    )(
        predictors=subject_data,
        target=target,
        target_dim="behavior",
        predictor_dim="time",
    ).assign_coords({"subject": subject})


def rot_inv_generalization(
    analysis: str, # "behavior"
    dataset: str,
    subject: int,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
) -> xr.Dataset:
    cache_path = _cache_path(f"main_analyses/{analysis}_rot_inv_generalization", dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    predictor_cache_path = f"main_analyses/{analysis}_decoding/dataset={dataset}"
    predictor_cache_path = _append_path(predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    
    match analysis:
        case "behavior":
            analysis_fn = _behavior_generalization
            cache_kwargs = {}
        case "bhvsubset":
            analysis_fn = _behavior_subset_generalization
            cache_kwargs = {}
        case _:
            raise ValueError("analysis must be 'binary' or 'multiclass'")
    
    data = load_dataset(dataset, subjects=subject, **load_dataset_kwargs,)
    if data is None:
        return xr.DataArray(None)
    
    subject_path = f"{cache_path}/subject={subject}.nc"
    cacher = cache(
        subject_path,
    )
    score = cacher(analysis_fn)(
        data=data,
        subject=subject,
        target_var=load_target_var(dataset),
        scorer_kwargs=scorer_kwargs,
        predictor_cache_path=f"{predictor_cache_path}/subject={subject}",
        **cache_kwargs,
    )
        
    return xr.concat([score], dim="subject")
