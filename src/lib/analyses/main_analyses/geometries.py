import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
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
from lib.analyses._plots import _plot_generalization
from lib.computation.statistics import compute_p
from bonner.computation.decomposition import PCA
from bonner.computation.metrics import pearson_r
from bonner.caching import cache




def _effective_dimensionality(X: torch.Tensor) -> float:
    pca = PCA()
    pca.fit(X)
    eigenvalues = pca.eigenvalues
    return (torch.pow(eigenvalues.sum(), 2) / torch.pow(eigenvalues, 2).sum()).item()

def _yield_metrics():
    yield "effective_dimensionality", _effective_dimensionality


def _geometry(X: xr.DataArray, metric_id: str, metric_fn: callable, geo_dims: set[str, str], list_dim: str) -> pd.DataFrame:
    subjects, list_dims, scores = [], [], []
    for subject in X.subject.values:
        x = (X
            .sel(subject=subject)
            .dropna("time")
            .transpose(*geo_dims, list_dim)
        )
        for l in x[list_dim].values:
            x_l = torch.tensor(x.sel({list_dim: l}).values)
            score = metric_fn(x_l)
            subjects.append(subject)
            list_dims.append(l)
            scores.append(score)
    return pd.DataFrame.from_dict({
        "subject": subjects,
        list_dim: list_dims,
        metric_id: scores,
    }).set_index(["subject", list_dim])


def geometries(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    space: str,
    geo_dims: set[str, str],
    list_dim: str,
    analysis: str = "multiclass",
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
    **kwargs,
):
    geo_id = f"space={space}.geo_dims={geo_dims[0]}_{geo_dims[1]}.list_dim={list_dim}"
    
    target_var = load_target_var(dataset)
    if list_dim == "target_var":
        list_dim = target_var
    elif geo_dims[0] == "target_var":
        geo_dims = (target_var, geo_dims[1])
    elif geo_dims[1] == "target_var":
        geo_dims = (geo_dims[0], target_var)
        
    match space:
        case "weights":
            X = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, subjects="intersection", significant_only=True, alpha=alpha)
        case "eeg":
            # X = (
            #     load_dataset(dataset, subjects=subjects, **load_dataset_kwargs,)
            #     .mean("presentation")
            # )
            # X = X.assign_coords({target_var: np.arange(len(X[target_var]))})
            X = (
                load_dataset(dataset, subjects=subjects, **load_dataset_kwargs,)
                .stack(stack_presentation=(target_var, "presentation"))
                .reset_index("stack_presentation")
                .drop_vars("presentation")
                .rename({"stack_presentation": "presentation"})
            )
    
    cache_path = _cache_path(f"main_analyses/geometries/analysis={analysis}/{geo_id}", dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    
    dfs = []
    for metric_id, metric_fn in _yield_metrics():
        cacher = cache(f"{cache_path}/metric={metric_id}.pkl")
        dfs.append(cacher(_geometry)(X, metric_id, metric_fn, geo_dims, list_dim))
    
    return pd.concat(dfs, axis=1)    
    
    