import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch
import itertools

from lib.analyses.loaders import load_weights
from lib.datasets import (
    load_dataset,
    load_target_var,
    load_presentation_reshaped_data
)
from lib.analyses.loaders import load_significant_times
from lib.analyses._utilities import _cache_path
from lib.analyses._plots import _plot_generalization
from lib.computation.statistics import compute_p
from bonner.computation.metrics import pearson_r
from bonner.caching import cache

def _flattend_tril(x: torch.Tensor) -> torch.Tensor:
    lower_tri_indices = torch.tril_indices(x.size(0), x.size(1), offset=-1)
    return x[lower_tri_indices[0], lower_tri_indices[1]]

def _melt_df(kc, subject, comparison_dim, x):
    df = (
        pd.DataFrame(kc, columns=x[comparison_dim].values, index=x[comparison_dim].values)
        .melt(var_name=f"{comparison_dim}_test", value_name="r", ignore_index=False)
        .reset_index(names=f"{comparison_dim}_train")
    )
    df["subject"] = subject
    return df

def _compute_kernels(x, comparison_dim, split=None):
    if split is not None:
        kernels = [
            _flattend_tril(1 - pearson_r(torch.tensor(x.sel({comparison_dim: c, "split": split}).values).T, return_diagonal=False))
            for c in x[comparison_dim]
        ]
    else:
        kernels = [
            _flattend_tril(1 - pearson_r(torch.tensor(x.sel({comparison_dim: c}).values).T, return_diagonal=False))
            for c in x[comparison_dim]
        ]
    return torch.stack(kernels, dim=-1)

def _compute_null_kernels(x, comparison_dim, rng, split=None):
    kernels = []
    for c in x[comparison_dim].values:
        rdm = 1 - pearson_r(torch.tensor(x.sel({comparison_dim: c, "split": split}).values).T, return_diagonal=False) if split is not None else 1 - pearson_r(torch.tensor(x.sel({comparison_dim: c}).values).T, return_diagonal=False)
        kn = _flattend_tril(rdm)
        kernels.append(kn[rng.permutation(len(kn))])
    return torch.stack(kernels, dim=-1)

def _kernel_comparison(X, kernel_dims, comparison_dim, cache_path, reorder_by_cluster, split_dim=None):
    if split_dim:
        X = X.mean("seed")
        
    X = X.transpose("subject", "split", *kernel_dims, comparison_dim).dropna("time") if split_dim else X.transpose("subject", *kernel_dims, comparison_dim).dropna("time")
    split_pairs = itertools.combinations(X.split.values, 2) if split_dim else [None]
    subject_pairs = itertools.combinations(X.subject.values, 2)
    
    kcs, null_kcs = [], []
    
    for subject0, subject1 in subject_pairs:
        for split_pair in split_pairs:
            kernels0 = _compute_kernels(X.sel(subject=subject0), comparison_dim, split=split_pair[0]) if split_pair else _compute_kernels(X.sel(subject=subject0), comparison_dim)
            
            kernels1 = _compute_kernels(X.sel(subject=subject1), comparison_dim, split=split_pair[1]) if split_pair else _compute_kernels(X.sel(subject=subject1), comparison_dim)
            kcs.append(pearson_r(kernels0, kernels1, return_diagonal=False).cpu().numpy())
            
            null_kc = []
            for seed in range(100):
                rng = np.random.default_rng(seed)
                kernels0 = _compute_null_kernels(X.sel(subject=subject1), comparison_dim, rng, split=split_pair[0]) if split_pair else _compute_null_kernels(X.sel(subject=subject0), comparison_dim, rng)
                kernels1 = _compute_null_kernels(X.sel(subject=subject1), comparison_dim, rng, split=split_pair[1]) if split_pair else _compute_null_kernels(X.sel(subject=subject1), comparison_dim, rng)
                null_kc.append(pearson_r(kernels0, kernels1, return_diagonal=False).cpu().numpy())
            null_kcs.append(np.stack(null_kc, axis=0))
    
    # Average and plot
    kc = np.average(kcs, axis=0)
    null_kc = np.mean(null_kcs, axis=0)
    p = compute_p(kc, null_kc, 'two_tailed')
    kc[p >= 0.05] = np.nan
    df = _melt_df(kc, "average", comparison_dim, X)
    _plot_generalization(df, 'subject', 'r', comparison_dim, cache_path, "significant_average", reorder_by_cluster=reorder_by_cluster, symmetric=True)
    
    return df

def kernel_comparison_cross_subject(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    space: str,
    kernel_dims: set[str, str],
    comparison_dim: str,
    analysis: str = "multiclass",
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
    split_dim: str = None,
    n_splits: int = 2,
    **kwargs,
):
    comparison_id = f"space={space}.kernel_dims={kernel_dims[0]}_{kernel_dims[1]}.comparison_dim={comparison_dim}"
    if split_dim is not None:
        comparison_id += f".split_dim={split_dim}.n_splits={n_splits}"
    else:
        comparison_id += ".split_dim=None"
    
    temp_cache_path = f"main_analyses/kernel_comparison.cross_subject/analysis={analysis}/{comparison_id}"
    
    cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=True)
    
    target_var = load_target_var(dataset)
    if comparison_dim == "target_var":
        comparison_dim = target_var
    elif kernel_dims[0] == "target_var":
        kernel_dims = (target_var, kernel_dims[1])
    elif kernel_dims[1] == "target_var":
        kernel_dims = (kernel_dims[0], target_var)
        
    match space:
        case "weights":
            X = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, subjects="intersection", significant_only=True, alpha=alpha, split_dim=split_dim, n_splits=n_splits, n_seeds=1)
        case "eeg":
            X = load_presentation_reshaped_data(
                dataset,
                split=split_dim is not None,
                average=True,
                n_splits=n_splits,
                subjects=subjects, 
                **load_dataset_kwargs
            )
            if analysis != "behavior":
                feature_dim = target_var
            else:
                feature_dim = "behavior"
            times = load_significant_times(analysis, dataset, load_dataset_kwargs, scorer_kwargs, feature_dim, alpha, subjects="intersection").mean("subject").dropna("time").values
            X = X.sel(time=[t in times for t in X.time.values])
    
    # TODO: temp change
    # reorder_by_cluster = comparison_dim != "time"
    reorder_by_cluster = False
    df_cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    cacher = cache(f"{df_cache_path}/results.pkl")
    
    return cacher(_kernel_comparison)(X, kernel_dims, comparison_dim, cache_path, reorder_by_cluster, split_dim=split_dim)


