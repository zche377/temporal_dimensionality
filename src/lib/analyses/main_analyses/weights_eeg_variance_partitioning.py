import shutil
import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import torch

from lib.analyses.loaders import load_weights
from lib.datasets import (
    load_dataset,
    load_target_var
)
from lib.analyses._utilities import _cache_path
from lib.utilities import (
    _append_path,
    SEED,
)

from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)
from bonner.datasets.grootswagers2022_things_eeg import IDENTIFIER
from bonner.computation.decomposition import PCA
from bonner.computation.metrics import covariance, pearson_r




def _pearson_r_with_eeg_channel_pc(
    weights: xr.DataArray,
    data: xr.DataArray,
    target_var: str,
) -> xr.DataArray:
    rs = []
    for subject in weights.subject.values:
        subject_rs = []
        w = weights.sel(subject=subject).dropna("time")
        d = data.sel(subject=subject)
        for t in w.time.values:
            w_t = w.sel(time=t)
            d_t = d.sel(time=t)
            pca = PCA()
            pca.fit(torch.tensor(d_t.values).float())
            evectors = pca.eigenvectors
            # r is shape (#target_vars, #eeg_channel_pcs)?
            r = pearson_r(torch.from_numpy(w_t.values), evectors, return_diagonal=False)
            subject_rs.append(xr.DataArray(
                r.cpu().numpy(),
                dims=(target_var, "eeg_channel_pc"),
            ).assign_coords(time=t))
        if len(subject_rs) > 0:
            rs.append(xr.concat(subject_rs, dim="time").assign_coords(subject=subject))
    return xr.concat(rs, dim="subject")


def weights_eeg_variance_partitioning(
    analysis: str, # "multiclass"
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
) -> xr.DataArray:
    target_var = load_target_var(dataset)
    weights = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, subjects, significant_only=True, alpha=alpha)
    data = (
        load_dataset(dataset, subjects=subjects, **load_dataset_kwargs,)
        .stack(stack_presentation=(target_var, "presentation"))
        .transpose("subject", "stack_presentation", "neuroid", "time")
    )
    
    # TODO: what variance partitioning method to use?
    
    # naive way: pearson r?
    cache_path = _cache_path(f"main_analyses/weights_channel_pc_eeg_variance_partitioning/pearson_r_with_eeg_channel_pc", dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    cache_path += f"/subject={subjects}.nc"
    cacher = cache(cache_path)
    return cacher(_pearson_r_with_eeg_channel_pc)(weights, data, target_var)
    