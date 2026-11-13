import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import torch

from lib.analyses._utilities import _cache_path
from lib.computation.statistics import cluster_correction
from lib.datasets import load_n_subjects

from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)
from bonner.datasets.grootswagers2022_things_eeg import IDENTIFIER



def _subjects(subjects, dataset) -> list[int]:
    if subjects == "all" or subjects == "intersection" or subjects == "group":
        subjects = list(range(1, load_n_subjects(dataset) + 1))
    if isinstance(subjects, int):
        subjects = [subjects]
    return subjects

def _load_significant_times(
    score_path: str,
    dataset: str,
    feature_dim: str,
    alpha: float = 0.05,
    subjects: (int | list[int] | str) = "all",
) -> xr.DataArray:
    times = []
    if subjects != "group":
        for subject in _subjects(subjects, dataset):
            # try:
            sx = xr.open_dataset(BONNER_CACHING_HOME / score_path / f"subject={subject}.nc")
            
            assert len(sx.data_vars) == 2
            for v in list(sx.data_vars):
                if v.startswith("null"):
                    null_var = v
                else:
                    score_var = v
            sx = (
                sx
                .mean(feature_dim)
                .transpose("permutation", "time")
            )
            p_vals = cluster_correction(sx[score_var].values, sx[null_var].values, alpha=alpha)
            times.append(sx.time.isel(time=p_vals<alpha))
            # except:
            #     pass
            
        times = xr.concat(times, dim="subject")
        if subjects == "intersection":
            times = times.where(~times.isnull().any(dim="subject").broadcast_like(times), np.nan)
    else:
        x = []
        subject_list = []
        for subject in _subjects(subjects, dataset):
            try:
                x.append(xr.open_dataset(BONNER_CACHING_HOME / score_path / f"subject={subject}.nc"))
                subject_list.append(subject)
            except:
                pass
        n_subject = len(x)
        x = xr.concat(x, dim="subject").mean(["subject", feature_dim]).transpose("permutation", "time")
        
        assert len(x.data_vars) == 2
        for v in list(x.data_vars):
            if v.startswith("null"):
                null_var = v
            else:
                score_var = v
        p_vals = cluster_correction(x[score_var].values, x[null_var].values, alpha=alpha)
        times = xr.concat([x.time.isel(time=p_vals<alpha) for _ in range(n_subject)], dim="subject").assign_coords(subject=subject_list)
        
    return times

def load_significant_times(
    analysis: str, # "behavior_tt_decoding.subset=False.pca=False"
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    feature_dim: str,
    alpha: float = 0.05,
    subjects: (int | list[int] | str) = "all",
    subset: bool = False,
    pca: bool = False,
) -> xr.DataArray:
    cache_path = _cache_path(f"loaders/significant_times/analysis={analysis}", dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    score_path = cache_path.replace(f"loaders/significant_times/analysis={analysis}", f"main_analyses/{analysis}")
    
    cache_path += f"/alpha={alpha}/subject={subjects}.nc"
    
    cacher = cache(cache_path)
    return cacher(_load_significant_times)(score_path, dataset, feature_dim, alpha, subjects,)
                





