import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch
import itertools

from lib.analyses.loaders import load_weights
from lib.datasets import (
    load_dataset,
    load_target_var
)
from lib.utilities import SEED
from lib.analyses._utilities import _cache_path
from lib.analyses._plots import _plot_generalization


from bonner.computation.metrics import pearson_r
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from lib.datasets.hebart2022_things_behavior import sort_embeddings



def _kernels(
    X: xr.DataArray,
    cache_path: Path,
    kernel_dims: set[str, str],
    list_dim: str,
):
    def _melt_df(kc, subject):
        df = (
            pd.DataFrame(kc, columns=x[kernel_dims[0]].values, index=x[kernel_dims[0]].values)
            .melt(var_name=f"{kernel_dims[0]}_test", value_name="r", ignore_index=False)
            .reset_index(names=f"{kernel_dims[0]}_train")
        )
        df["subject"] = subject
        return df
    
    for subject in X.subject.values:
        x = (X
            .sel(subject=subject)
            .dropna("time")
            .transpose(*kernel_dims, list_dim)
        )
        for l in x[list_dim].values:
            kc = pearson_r(torch.tensor(x.sel({list_dim: l}).values).T, return_diagonal=False).cpu().numpy()
            df = _melt_df(kc, subject)
            if isinstance(l, str):
                l = l.replace('/', '_or_')
            df[list_dim] = l
            _plot_generalization(df, list_dim, "r", kernel_dims[0], cache_path / f"{list_dim}={l}", subject, symmetric=True)
            

def _kernels_average(
    X: xr.DataArray,
    cache_path: Path,
    kernel_dims: set[str, str],
    list_dim: str,
    split_dim: str,
):
    def _melt_df(kc, subject):
        df = (
            pd.DataFrame(kc, columns=x[kernel_dims[0]].values, index=x[kernel_dims[0]].values)
            .melt(var_name=f"{kernel_dims[0]}_test", value_name="r", ignore_index=False)
            .reset_index(names=f"{kernel_dims[0]}_train")
        )
        df["subject"] = subject
        return df
    
    for subject in X.subject.values:
        kc = []
        if split_dim is None:
            x = (X
                .mean("presentation")
                .sel(subject=subject)
                .dropna("time")
                .transpose(*kernel_dims, list_dim)
                .stack(list_neuroid=(kernel_dims[1], list_dim))
            )
            
            kc.append(pearson_r(torch.tensor(x.values).T, return_diagonal=False).cpu().numpy())
            # for l in x[list_dim].values:
            #     kc.append(pearson_r(torch.tensor(x.sel({list_dim: l}).values).T, return_diagonal=False).cpu().numpy())
        else:
            assert split_dim == "presentation"
            split_pairs = list(itertools.permutations(np.arange(2), 2))
            # n_presentation_per_split
            npps = len(X.presentation.values) // 2
            
            rng = np.random.default_rng(11)
            random_idx = rng.permutation(len(X.presentation.values))
            
            
            x = (X
                .sel(subject=subject)
                .dropna("time")
                .transpose("presentation", *kernel_dims, list_dim)
                .stack(list_neuroid=(kernel_dims[1], list_dim))
            )
            for split_pair in split_pairs:
                x1 = torch.tensor(x.sel({
                    "presentation": random_idx[np.arange(split_pair[0]*npps, (split_pair[0]+1)*npps)]
                }).mean("presentation").values).T
                x2 = torch.tensor(x.sel({
                    "presentation": random_idx[np.arange(split_pair[1]*npps, (split_pair[1]+1)*npps)]
                }).mean("presentation").values).T
                kc.append(pearson_r(x1, x2, return_diagonal=False).cpu().numpy())
        kc = np.mean(kc, axis=0)
        df = _melt_df(kc, subject)
        df[list_dim] = "average"
        _plot_generalization(df, list_dim, "r", kernel_dims[0], cache_path / f"{list_dim}=average", subject, symmetric=True)
       
            
def _kernels_behavior(
    X: xr.DataArray,
    cache_path: Path,
    kernel_dims: set[str, str],
):
    def _melt_df(kc,):
        df = (
            pd.DataFrame(kc, columns=x[kernel_dims[0]].values, index=x[kernel_dims[0]].values)
            .melt(var_name=f"{kernel_dims[0]}_test", value_name="r", ignore_index=False)
            .reset_index(names=f"{kernel_dims[0]}_train")
        )
        return df
    
    x = X.transpose(*kernel_dims)
    kc = pearson_r(torch.tensor(x.values).T, return_diagonal=False).cpu().numpy()
    df = _melt_df(kc,)
    _plot_generalization(df, None, "r", kernel_dims[0], cache_path, None, symmetric=True, reorder_by_cluster=True)
        
        
def kernels(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    space: str,
    kernel_dims: set[str, str],
    list_dim: str,
    analysis: str = "multiclass",
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
    split_dim: str = None,
    **kwargs,
):
    list_id = f"space={space}.kernel_dims={kernel_dims[0]}_{kernel_dims[1]}.list_dim={list_dim}"
    if split_dim is not None:
        list_id += f".split_dim={split_dim}.n_splits=session"
    else:
        list_id += ".split_dim=None"
    
    cache_path = _cache_path(f"main_analyses/kernels/analysis={analysis}/{list_id}", dataset, load_dataset_kwargs, scorer_kwargs, include_root=True)
    
    target_var = load_target_var(dataset)
    if list_dim == "target_var":
        list_dim = target_var
    elif kernel_dims[0] == "target_var":
        kernel_dims = (target_var, kernel_dims[1])
    elif kernel_dims[1] == "target_var":
        kernel_dims = (kernel_dims[0], target_var)
        
    match space:
        case "weights":
            X = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, subjects="intersection", significant_only=True, alpha=alpha)
            _kernels(X, cache_path, kernel_dims, list_dim,)
        case "behavior":
            assert dataset == "things_behavior"
            X = load_embeddings()
            _kernels_behavior(X, cache_path, kernel_dims,)
        case "bhvsubset":
            assert dataset == "things_behavior"
            X = sort_embeddings()
            _kernels_behavior(X, cache_path, kernel_dims,)
        case "eeg":
            X = load_dataset(dataset, subjects=subjects, **load_dataset_kwargs,)
            X = X.assign_coords({target_var: np.arange(len(X[target_var]))})
            if 'freq' not in X.coords:
                _kernels_average(X, cache_path, kernel_dims, list_dim, split_dim,)
            else:
                for f in tqdm(X.freq.values, desc="freq"):
                    _kernels_average(X.sel(freq=f), cache_path / f"freq={f}", kernel_dims, list_dim, split_dim,)
                    
            
    



