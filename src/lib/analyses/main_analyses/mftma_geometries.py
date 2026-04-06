import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch
from tqdm import tqdm

from lib.analyses.loaders import load_weights
from lib.datasets import (
    load_dataset,
    load_target_var,
    load_presentation_reshaped_data
)
from lib.analyses.loaders import load_significant_times
from lib.analyses._utilities import _cache_path
from lib.computation.metrics import manifold_analysis_corr
from bonner.caching import cache

KAPPA = 0
N_T = 100
N_REPS = 1

    
    
def _mftma_geometries(X: xr.DataArray, geo_dims: set[str, str], list_dim: str, loop_dim: str):
    
    def _ard(x: xr.DataArray):
        x = [x.sel({list_dim: li}).transpose(*geo_dims).values for li in x[list_dim].values]
        n_t = x[0].shape[-1]
        if n_t > N_T:
            n_t = N_T
        a, r, d, _, _ = manifold_analysis_corr(x, KAPPA, n_t, n_reps=N_REPS)
        return a, r, d
    
    capacities, radii, dimensions = [], [], []
    for s in tqdm(X.subject.values, desc="subject"):
        if loop_dim is not None:
            s_capacities, s_radii, s_dimensions = [], [], []
            for lo in tqdm(X[loop_dim].values, desc=loop_dim):
                a, r, d = _ard(X.sel({"subject": s, loop_dim: lo}))
                s_capacities.append(a)
                s_radii.append(r)
                s_dimensions.append(d)
            capacities.append(np.stack(s_capacities))
            radii.append(np.stack(s_radii))
            dimensions.append(np.stack(s_dimensions))
        else:
            a, r, d = _ard(X.sel({"subject": s}))
            capacities.append(a)
            radii.append(r)
            dimensions.append(d)
    capacities = np.stack(capacities)
    radii = np.stack(radii)
    dimensions = np.stack(dimensions)
    
    if loop_dim is not None:
        return xr.Dataset(
            {
                "capacities": (("subject", loop_dim, list_dim), capacities),
                "radii": (("subject", loop_dim, list_dim), radii),
                "dimensions": (("subject", loop_dim, list_dim), dimensions),
            },
            coords={"subject": X.subject.values, loop_dim: X[loop_dim].values, list_dim: X[list_dim].values}
        )
    else:
        return xr.Dataset(
            {
                "capacities": (("subject", list_dim), capacities),
                "radii": (("subject", list_dim), radii),
                "dimensions": (("subject", list_dim), dimensions),
            },
            coords={"subject": X.subject.values, list_dim: X[list_dim].values}
        )
                
def mftma_geometries(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    space: str,
    geo_dims: set[str, str],
    list_dim: str,
    loop_dim: str = None,
    analysis: str = "multiclass",
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
    split_dim: str = None,
    n_splits: int = 2,
    **kwargs,
):
    geo_id = f"space={space}.geo_dims={geo_dims[0]}_{geo_dims[1]}.list_dim={list_dim}.loop_dim={loop_dim}"
    
    target_var = load_target_var(dataset)
    if list_dim == "target_var":
        list_dim = target_var
    elif loop_dim is not None and loop_dim == "target_var":
        loop_dim = target_var
    elif geo_dims[0] == "target_var":
        geo_dims = (target_var, geo_dims[1])
    elif geo_dims[1] == "target_var":
        geo_dims = (geo_dims[0], target_var)
        
    match space:
        case "weights":
            X = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, subjects="group", significant_only=True, alpha=alpha, split_dim=split_dim, n_splits=n_splits, n_seeds=1)
        case "eeg":
            X = load_presentation_reshaped_data(
                dataset,
                split=split_dim is not None,
                n_splits=n_splits,
                subjects=subjects, 
                **load_dataset_kwargs
            )
        
            
            # X = load_presentation_reshaped_data(
            #     dataset,
            #     split=split_dim is not None,
            #     n_splits=n_splits,
            #     subjects=subjects, 
            #     stack=True,
            #     stack_dims=["time", "presentation"],
            #     **load_dataset_kwargs
            # )
            if analysis != "behavior":
                feature_dim = target_var
            else:
                feature_dim = "behavior"
                
            if geo_dims[0] == "time":
                times = load_significant_times(analysis, dataset, load_dataset_kwargs, scorer_kwargs, feature_dim, alpha, subjects="group").mean("subject").dropna("time").values
                X = X.sel(time=[t in times for t in X.time.values])
    
    cache_path = _cache_path(f"main_analyses/mftma_geometries/analysis={analysis}/{geo_id}", dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    
    cacher = cache(f"{cache_path}.nc")
    
    return cacher(_mftma_geometries)(X, geo_dims, list_dim, loop_dim)
    
    