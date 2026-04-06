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
from lib.datasets import load_dataset, load_target_var, load_n_subjects
from lib.utilities import (
    _append_path,
    SEED,
)
from lib.datasets.hebart2022_things_behavior import sort_embeddings
from lib.models import DJModel

from bonner.computation.decomposition import PCA
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)


def _reshape_subject_data(data, target_var, average=False,) -> xr.DataArray:
    if average:
        return (
            data.isel(subject=0)
            .mean("presentation")
            .transpose(target_var, "neuroid", "time")
            .rename({target_var: "presentation"})
        )
    else:
        return (
            data.isel(subject=0)
            .stack(stack_presentation=(target_var, "presentation"))
            .transpose("stack_presentation", "neuroid", "time")
            .reset_index("stack_presentation")
            .drop_vars("presentation")
            .rename({"stack_presentation": "presentation"})
        )

def _subject_data(data_train, data_test, target_var, average=True,):
    subject_data_train = _reshape_subject_data(data_train, target_var, average)
    subject_data_test = _reshape_subject_data(data_test, target_var, average)
    return subject_data_train.sortby("img_files"), subject_data_test.sortby("img_files")


def _plssvd(
    data0_train: xr.DataArray,
    data0_test: xr.DataArray,
    data1_train: xr.DataArray,
    data1_test: xr.DataArray,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
):
    data0_train, data0_test = _subject_data(
        data0_train, data0_test, target_var, average=True,
    )
    data1_train, data1_test = _subject_data(
        data1_train, data1_test, target_var, average=True,
    )
    
    assert scorer_kwargs["model_name"] == "plssvd"
    scorer_fn = TrainTestPLSSVDScorer
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=time={t}",
        ))(
            predictor_train=data0_train.sel(time=t),
            target_train=data1_train.sel(time=t),
            predictor_test=data0_test.sel(time=t),
            target_test=data1_test.sel(time=t),
            target_dim="behavior",
        ) for t in data0_train.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(data0_train.time.values)]
    
    return xr.concat(time_scores, dim="time")


def plssvd_cross_subject(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    **kwargs,
) -> xr.Dataset:
    cache_str = f"main_analyses/plssvd_cross_subject"
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

    def _scores():
        n_subject = load_n_subjects(dataset)
        scores = []
        for subject0, subject1 in tqdm(itertools.combinations(range(1, n_subject+1), 2), desc="subject_pair"):
            data0_train = load_dataset(
                f"{dataset}_train",
                subjects=subject0,
                **load_dataset_kwargs,
            )
            data0_test = load_dataset(
                f"{dataset}_test",
                subjects=subject0,
                **load_dataset_kwargs,
            )
            data1_train = load_dataset(
                f"{dataset}_train",
                subjects=subject1,
                **load_dataset_kwargs,
            )
            data1_test = load_dataset(
                f"{dataset}_test",
                subjects=subject1,
                **load_dataset_kwargs,
            )
            target_var = load_target_var(f"{dataset}_train")
            scores.append(_plssvd(
                data0_train,
                data0_test,
                data1_train,
                data1_test,
                target_var,
                scorer_kwargs,
                predictor_cache_path=f"{predictor_cache_path}/subjects={subject0}-{subject1}",
            ).assign_coords({"subject": f"{subject0}-{subject1}"}))

        return xr.concat(scores, dim="subject")
    
    cache_path += ".nc"
    cacher = cache(cache_path)
    return cacher(_scores)()