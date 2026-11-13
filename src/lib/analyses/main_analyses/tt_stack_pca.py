import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN axis')

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch
from tqdm import tqdm
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from copy import deepcopy
from joblib import Parallel, delayed
import seaborn as sns

from lib.computation.statistics import cluster_correction
from lib.analyses.loaders import load_weights
from lib.datasets import (
    load_dataset,
    load_target_var,
    load_presentation_reshaped_data
)
from lib.analyses.loaders import load_significant_times
from lib.analyses._utilities import _cache_path
from lib.analyses._plots import _plot_generalization, _plot_eigenspectrum, _plot_gen_with_p
from lib.utilities import SEED
from lib.computation.statistics import compute_p

from bonner.computation.decomposition import PCA
from bonner.computation.metrics import covariance, pearson_r
from bonner.caching import cache



def _sb_rel(x, n=2):
        return n*x/(1+(n-1)*x)

def _tt_stack_pca(
    X_train: xr.DataArray,
    X_test: xr.DataArray,
    cache_path: Path,
    pc_dims: set[str, str],
    split_dim=None
) -> pd.DataFrame:
    if split_dim:
        X_test = X_test.mean("seed")
    
    rs, null_rs, cvnsrs, null_cvnsrs, neg_cvnsrs, null_neg_cvnsrs = [], [], [], [], [], []
    covs, null_covs = [], []
    
    for subject in tqdm(X_train.subject.values, desc="subject"):
        x_train = X_train.sel(subject=subject)
        x_test = X_test.sel(subject=subject)
        x_test = x_test.transpose("split", *pc_dims) if split_dim else x_test.transpose(*pc_dims)
        
        pca = PCA()
        x_train_fit = torch.tensor(x_train.values).to("cuda")
        pca.fit(x_train_fit)
        
        g_rs, g_null_rs, g_cvnsrs, g_null_cvnsrs, g_neg_cvnsrs, g_null_neg_cvnsrs = [], [], [], [], [], []
        g_covs, g_null_covs = [], []
        for split_pair in itertools.permutations(x_test.split.values, 2) if split_dim else [None]:
            x_test_split0 = torch.tensor(x_test.sel({"split": split_pair[0]}).values if split_dim else x_test.values).to("cuda")
            
            x_test_split1 = torch.tensor(x_test.sel({"split": split_pair[1]}).values if split_dim else x_test.values).to("cuda")
            z_test_split0= pca.transform(x_test_split0)
            
            x_test_split1 = x_test.sel({"split": split_pair[1]}) if split_dim else x_test
            z_test_split1 = pca.transform(torch.tensor(x_test_split1.values))
            
            # TODO: do we want to spearman brown here for good?
            r = _sb_rel(pearson_r(z_test_split0, z_test_split1))
            # neuroid
            
            cov = covariance(z_test_split0, z_test_split1)
            
            null_r = []
            null_cov = []
            for seed in range(100):
                rng = torch.Generator().manual_seed(seed)
                randi = torch.argsort(torch.rand(z_test_split0.size(), generator=rng), dim=0).to("cuda")
                z_test_split1_shuffled = z_test_split1.gather(1, randi)
                null_r.append(_sb_rel(pearson_r(z_test_split0, z_test_split1_shuffled)))
                null_cov.append(covariance(z_test_split0, z_test_split1_shuffled))
            null_r = torch.stack(null_r, dim=0)
            null_cov = torch.stack(null_cov, dim=0)
            # permute * neuroid
            
            p = compute_p(r.cpu().numpy(), null_r.cpu().numpy(), 'greater')
            cvnsr = torch.tensor(np.sum(p < .05, axis=-1)).float()
            p = compute_p(r.cpu().numpy(), null_r.cpu().numpy(), 'less')
            neg_cvnsr = torch.tensor(np.sum(p < .05, axis=-1)).float()
            # 1
            
            null_r2 = []
            for seed in range(100, 200):
                rng = torch.Generator().manual_seed(seed)
                randi = torch.argsort(torch.rand(z_test_split0.size(), generator=rng), dim=0).to("cuda")
                z_test_split1_shuffled = z_test_split1.gather(1, randi)
                null_r2.append(_sb_rel(pearson_r(z_test_split0, z_test_split1_shuffled)))
            null_r2 = torch.stack(null_r2, dim=0)
            
            null_cvnsr = []
            null_neg_cvnsr = []
            for seed in range(100):
                temp_r = null_r[seed]
                p = compute_p(temp_r.cpu().numpy(), null_r2.cpu().numpy(), 'greater')
                null_cvnsr.append(np.sum(p < .05, axis=-1))
                p = compute_p(temp_r.cpu().numpy(), null_r2.cpu().numpy(), 'less')
                null_neg_cvnsr.append(np.sum(p < .05, axis=-1))
            null_cvnsr = torch.tensor(np.stack(null_cvnsr, axis=0)).float()
            null_neg_cvnsr = torch.tensor(np.stack(null_neg_cvnsr, axis=0)).float()
            # permute
            
            g_rs.append(r)
            g_null_rs.append(null_r)
            g_cvnsrs.append(cvnsr)
            g_neg_cvnsrs.append(neg_cvnsr)
            g_null_cvnsrs.append(null_cvnsr)
            g_null_neg_cvnsrs.append(null_neg_cvnsr)
            g_covs.append(cov)
            g_null_covs.append(null_cov)
            
        rs.append(torch.stack(g_rs, dim=0))
        null_rs.append(torch.stack(g_null_rs, dim=0))
        cvnsrs.append(torch.stack(g_cvnsrs, dim=0))
        neg_cvnsrs.append(torch.stack(g_neg_cvnsrs, dim=0))
        null_cvnsrs.append(torch.stack(g_null_cvnsrs, dim=0))
        null_neg_cvnsrs.append(torch.stack(g_null_neg_cvnsrs, dim=0))
        covs.append(torch.stack(g_covs, dim=0))
        null_covs.append(torch.stack(g_null_covs, dim=0))
        # split * ...
        
    rs = torch.stack(rs, dim=0)
    null_rs = torch.stack(null_rs, dim=0)
    cvnsrs = torch.stack(cvnsrs, dim=0)
    neg_cvnsrs = torch.stack(neg_cvnsrs, dim=0)
    null_cvnsrs = torch.stack(null_cvnsrs, dim=0)
    null_neg_cvnsrs = torch.stack(null_neg_cvnsrs, dim=0)
    covs = torch.stack(covs, dim=0)
    null_covs = torch.stack(null_covs, dim=0)
    # subject * split * ...
    
    results = {}
    results["r"] = xr.DataArray(
        rs.cpu().numpy(),
        dims=["subject", "split", "neuroid"]
    )
    results["null_r"] = xr.DataArray(
        null_rs.cpu().numpy(),
        dims=["subject", "split", "permute", "neuroid"]
    )
    results["cvnsr"] = xr.DataArray(
        cvnsrs.cpu().numpy(),
        dims=["subject", "split", ]
    )
    results["null_cvnsr"] = xr.DataArray(
        null_cvnsrs.cpu().numpy(),
        dims=["subject", "split", "permute",]
    )
    results["neg_cvnsr"] = xr.DataArray(
        neg_cvnsrs.cpu().numpy(),
        dims=["subject", "split", ]
    )
    results["null_neg_cvnsr"] = xr.DataArray(
        null_neg_cvnsrs.cpu().numpy(),
        dims=["subject", "split", "permute", ]
    )
    results["cov"] = xr.DataArray(
        covs.cpu().numpy(),
        dims=["subject", "split", "neuroid"]
    )
    results["null_cov"] = xr.DataArray(
        null_covs.cpu().numpy(),
        dims=["subject", "split", "permute", "neuroid"]
    )
    
    return xr.Dataset(results)

def _stack(X, pc_dims, stack_dim):
    return (X
        .stack(stack=[stack_dim, pc_dims[1]])
        .reset_index("stack")
        .drop_vars(pc_dims[1])
        .rename({"stack": pc_dims[1]})
    )

def tt_stack_pca(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    space: str,
    pc_dims: set[str, str],
    stack_dim: str,
    analysis: str = "multiclass",
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
    split_dim: str = None,
    n_splits: int = 2,
    time: str = "significant", # significant or control
    **kwargs,
):
    stack_id = f"space={space}.pc_dims={pc_dims[0]}_{pc_dims[1]}.stack_dim={stack_dim}.time={time}"
    if split_dim is not None:
        stack_id += f".split_dim={split_dim}.n_splits={n_splits}"
    else:
        stack_id += ".split_dim=None"
    
    temp_cache_path = f"main_analyses/tt_stack_pca/analysis={analysis}/{stack_id}"
    cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, None if space=="eeg" and "time" not in [*pc_dims, stack_dim] else scorer_kwargs, include_root=True)
    
    target_var = load_target_var(dataset)
    if stack_dim == "target_var":
        stack_dim = target_var
    elif pc_dims[0] == "target_var":
        pc_dims = (target_var, pc_dims[1])
    elif pc_dims[1] == "target_var":
        pc_dims = (pc_dims[0], target_var)
    
    match space:
        case "eeg":
            X_train = load_presentation_reshaped_data(
                f"{dataset}_train",
                split=False,
                average=True,
                subjects=subjects, 
                **load_dataset_kwargs
            )
            X_test = load_presentation_reshaped_data(
                f"{dataset}_test",
                split=split_dim is not None,
                average=True,
                n_splits=n_splits,
                subjects=subjects, 
                **load_dataset_kwargs
            )
            
            if analysis not in ["behavior", "bhvpc", "bhvsubset"]:
                feature_dim = target_var
            else:
                feature_dim = "behavior"
                
            if "time" in [*pc_dims, stack_dim]:
                match time:
                    case "significant":
                        temp_scorer_kwargs = deepcopy(scorer_kwargs)
                        times = load_significant_times(f"{analysis}_tt_decoding.subset=False.pca=False", dataset, load_dataset_kwargs, temp_scorer_kwargs, feature_dim, alpha, subjects="intersection").mean("subject").dropna("time").values
                        times = np.round(times, 2)
                        
                        X_train = X_train.sel(time=[t in times for t in X_train.time.values])
                        X_test = X_test.sel(time=[t in times for t in X_test.time.values])
                    case "control":
                        X_train = X_train.sel(time=[t<0 for t in X_train.time.values])
                        X_test = X_test.sel(time=[t<0 for t in X_test.time.values])
            
            X_train = _stack(X_train, pc_dims, stack_dim)
            X_test = _stack(X_test, pc_dims, stack_dim)
            
    df_cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    cacher = cache(
        f"{df_cache_path}/results.nc", 
        # mode="ignore"
    )
    
    return cacher(_tt_stack_pca)(X_train, X_test, cache_path, pc_dims, split_dim=split_dim)