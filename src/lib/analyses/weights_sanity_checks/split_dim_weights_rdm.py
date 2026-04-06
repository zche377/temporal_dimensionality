import logging
logging.basicConfig(level=logging.INFO)

import itertools
import numpy as np
import pandas as pd
import xarray as xr
import torch


from lib.analyses._utilities import _cache_path
from lib.analyses.loaders import load_weights
from lib.datasets import load_target_var

from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)
from bonner.computation.metrics import covariance, pearson_r



def _flattend_tril(x: torch.Tensor) -> torch.Tensor:
    lower_tri_indices = torch.tril_indices(x.size(0), x.size(1), offset=-1)
    return x[lower_tri_indices[0], lower_tri_indices[1]]


def _split_dim_weights_rdm(analysis, dataset, load_dataset_kwargs, scorer_kwargs, split_dim, n_splits, n_seeds, target_var,):
    weights = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, significant_only=False, split_dim=split_dim, n_splits=n_splits, n_seeds=n_seeds,)
    
    subject_scores = []
    for subject in weights.subject.values:
        seed_scores = []
        for seed in weights.seed.values:
            time_scores = []
            for t in weights.time.values:
                split_weights = weights.sel(subject=subject, seed=seed, time=t)
                pair_scores = []
                for split_pair in itertools.combinations(split_weights.split.values, 2):
                    sw1 = torch.tensor(split_weights.sel(split=split_pair[0]).values)
                    sw1 = _flattend_tril(1-pearson_r(sw1, return_diagonal=False))
                    sw2 = torch.tensor(split_weights.sel(split=split_pair[1]).values)
                    sw2 = _flattend_tril(1-pearson_r(sw2, return_diagonal=False))
                    
                    r = pearson_r(
                        sw1, sw2, return_diagonal=False
                    )
                    cov = covariance(
                        sw1, sw2,return_diagonal=False
                    )
                    pair_scores.append(
                        xr.Dataset(
                            {
                                "r": r,
                                "cov": cov,
                            }
                        ).assign_coords({"split_pair": str(split_pair)})
                    )
                time_scores.append(xr.concat(pair_scores, dim="split_pair").assign_coords({"time": t}))
            seed_scores.append(xr.concat(time_scores, dim="time").assign_coords({"seed": seed}))
        subject_scores.append(xr.concat(seed_scores, dim="seed").assign_coords({"subject": subject}))
        
    return xr.concat(subject_scores, dim="subject")

def split_dim_weights_rdm(
    analysis: str, # "multiclass"
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    split_dim: str,
    n_splits: int = 2,
    n_seeds: int = 10
) -> xr.DataArray:
    split_id = f"split_dim={split_dim}.n_splits={n_splits}.n_seeds={n_seeds}"
    
    temp_cache_path = f"weights_sanity_checks/split_dim_weights_rdm/{analysis}_decoding/{split_id}"
    
    cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    cacher = cache(f"{cache_path}/subject=all.nc")
    
    target_var = load_target_var(dataset)
    if split_dim == "target_var":
        split_dim = target_var

    return cacher(_split_dim_weights_rdm)(
        analysis, dataset, load_dataset_kwargs, scorer_kwargs, split_dim, n_splits, n_seeds, target_var,
    )
    
    