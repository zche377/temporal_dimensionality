import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import torch

from lib.analyses._utilities import _cache_path
from lib.analyses.main_analyses.decoding import _reshape_subject_data
from lib.computation.scorers import ModelGeneralizationScorer
from lib.datasets import (
    load_dataset,
    load_target_var
)
from lib.utilities import _append_path
from lib.datasets.hebart2022_things_behavior import sort_embeddings

from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import cache



def _object_binary_generalization(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values
    var_values = data[target_var].values
    
    target = np.zeros((len(subject_var_values), len(var_values)), dtype=int)
    if subject_var_values.dtype != np.dtype("int64"):
        one_hot_index = pd.Series(subject_var_values).factorize()[0]
    else:
        one_hot_index = subject_var_values
    target[np.arange(len(subject_var_values)), one_hot_index] = 1
    target = xr.DataArray(
        target,
        dims=("presentation", target_var),
        coords={target_var: var_values},
    )
    
    return ModelGeneralizationScorer(
        **scorer_kwargs,
        cache_predictors=True,
        cache_subpath=f"{predictor_cache_path}/target={target_var}=binary",
    )(
        predictors=subject_data,
        target=target,
        target_dim=target_var,
        predictor_dim="time",
    ).assign_coords({"subject": subject})
    
def _object_multiclass_generalization(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values
    
    target = xr.DataArray(
        np.expand_dims(
            pd.Series(subject_var_values).factorize()[0], axis=-1
        ),
        dims=("presentation", target_var),
        coords={target_var: ['multiclass']},
    )
    
    return ModelGeneralizationScorer(
        **scorer_kwargs,
        # TODO: permutation necessary for this analysis?
        cache_predictors=True,
        cache_subpath=f"{predictor_cache_path}/target={target_var}=multiclass",
    )(
        predictors=subject_data,
        target=target,
        target_dim=target_var,
        predictor_dim="time",
    ).assign_coords({"subject": subject})
    

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
    
    return ModelGeneralizationScorer(
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
    
    return ModelGeneralizationScorer(
        **scorer_kwargs,
        cache_predictors=True,
        cache_subpath=f"{predictor_cache_path}/target=behavior=all",
    )(
        predictors=subject_data,
        target=target,
        target_dim="behavior",
        predictor_dim="time",
    ).assign_coords({"subject": subject})


def decoding_generalization(
    analysis: str, # "binary" | "multiclass" | "behavior"
    dataset: str,
    subject: int,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
) -> xr.Dataset:
    cache_path = _cache_path(f"main_analyses/{analysis}_generalization", dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    predictor_cache_path = f"main_analyses/{analysis}_decoding/dataset={dataset}"
    predictor_cache_path = _append_path(predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    
    match analysis:
        case "binary":
            analysis_fn = _object_binary_generalization
            cache_kwargs = {}
        case "multiclass":
            analysis_fn = _object_multiclass_generalization
            cache_kwargs = {}
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
    
    if 'freq' not in data.coords:
        subject_path = f"{cache_path}/subject={subject}.nc"
        cacher = cache(
            subject_path,
            # mode="ignore",
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
    else:
        scores = []
        for f in tqdm(data.freq.values, desc="freq"):
            subject_path = f"{cache_path}/freq={f}/subject={subject}.nc"
            cacher = cache(
                subject_path,
                # mode="ignore",
            )
            scores.append(cacher(analysis_fn)(
                data=data.sel(freq=f),
                subject=subject,
                target_var=load_target_var(dataset),
                scorer_kwargs=scorer_kwargs,
                predictor_cache_path=f"{predictor_cache_path}/freq={f}/subject={subject}",
                **cache_kwargs,
            ))

        return xr.concat([xr.concat(scores, dim="freq")], dim="subject")

