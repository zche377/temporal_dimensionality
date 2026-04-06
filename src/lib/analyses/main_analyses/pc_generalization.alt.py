import logging
logging.basicConfig(level=logging.INFO)

import itertools
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
from lib.analyses._plots import _plot_generalization
from lib.utilities import SEED
from bonner.computation.decomposition import PCA
from bonner.computation.metrics import covariance, pearson_r
from bonner.caching import cache
from lib.computation.statistics import compute_p


def aggregate_dfs(dfs):
    def aggregate_column(col):
        return col.iloc[0] if col.nunique() == 1 else np.mean(col)
    df = pd.concat(dfs, axis=0)
    return df.groupby(df.index).agg(aggregate_column)

def _pc_generalization(
    X: xr.DataArray,
    cache_path: Path,
    pc_dims: set[str, str],
    generalization_dim: str,
    reorder_by_cluster: bool,
    split_dim=None
) -> pd.DataFrame:
    dfs, rs, null_rs = [], [], []
    if split_dim:
        X = X.mean("seed")
        
    def _dict(pc, r, ev, g_train, g_test, subject):
        return pd.DataFrame.from_dict({
            "pc": pc,
            "r": r,
            "ev": ev,
            f"{generalization_dim}_train": g_train,
            f"{generalization_dim}_test": g_test,
            "subject": subject,
        })
    
    def _plots(dfs, subject, subject_prefix=None):
        subject_prefix = f"{subject_prefix}_" if subject_prefix else ""
        if reorder_by_cluster:
            _plot_generalization(dfs, 'pc', 'ev', generalization_dim, cache_path, f"fixedby1_{subject_prefix}{subject}", reorder_by_cluster=reorder_by_cluster, center=None)
            _plot_generalization(dfs, 'pc', 'r', generalization_dim, cache_path, f"fixedby1_{subject_prefix}{subject}", reorder_by_cluster=reorder_by_cluster,)
        _plot_generalization(dfs, 'pc', 'ev', generalization_dim, cache_path, f"{subject_prefix}{subject}", first_row_order_only=False, reorder_by_cluster=reorder_by_cluster, center=None)
        _plot_generalization(dfs, 'pc', 'r', generalization_dim, cache_path, f"{subject_prefix}{subject}", first_row_order_only=False, reorder_by_cluster=reorder_by_cluster,)
    
    rng_pt = torch.Generator().manual_seed(11)
    random_indices, test_random_indices = None, None
    
    for subject in tqdm(X.subject.values, desc="subject"):
        x = (X
            .sel(subject=subject)
            .dropna("time")
        )
        x = x.transpose("split", *pc_dims, generalization_dim) if split_dim else x.transpose(*pc_dims, generalization_dim)
        
        subject_dfs, subject_significant_dfs = [], []
        for split_pair in itertools.permutations(x.split.values, 2) if split_dim else [None]:
            df, significant_df, split_null_rs = [], [], []
            for g_train in x[generalization_dim]:
                pca = PCA()
                x_train_fit = torch.tensor(x.sel({generalization_dim: g_train, "split": split_pair[0]}).values if split_dim else x.sel({generalization_dim: g_train}).values).to("cuda")
                
                # x_train_fit = (x_train_fit - x_train_fit.mean(dim=1, keepdim=True) / x_train_fit.std(dim=1, keepdim=True))
                
                if random_indices is None:
                    random_indices = torch.argsort(torch.rand(x_train_fit.size(), generator=rng_pt), dim=1).to("cuda")
                # random_indices = torch.argsort(torch.rand(x_train_fit.size(), generator=rng_pt), dim=1).to("cuda")
                x_train_fit = x_train_fit.gather(1, random_indices)
                
                # if random_indices is None:
                #     random_indices = torch.argsort(torch.rand(x_train_fit.t().size(), generator=rng_pt), dim=1).to("cuda")
                # x_train_fit = x_train_fit.t().gather(1, random_indices).t()
                
                pca.fit(x_train_fit)
                z_train_split0 = pca.transform(x_train_fit)
                
                x_test_split1 = x.sel({"split": split_pair[1]}) if split_dim else x
                
                z_test_split1 = torch.tensor(
                    x_test_split1.stack(stack=[generalization_dim, pc_dims[0]]).transpose("stack", pc_dims[1]).values
                ).to("cuda")
                
                # z_test_split1 = (z_test_split1 -z_test_split1.mean(dim=1, keepdim=True) / z_test_split1.std(dim=1, keepdim=True))
                
                lg = len(x[generalization_dim])
                lpc1 = len(x[pc_dims[0]])
                lpc2 = len(x[pc_dims[1]])
                
                if test_random_indices is None:
                    test_random_indices = random_indices.repeat(lg, 1)
                # test_random_indices = torch.argsort(torch.rand(z_test_split1.size(), generator=rng_pt), dim=1).to("cuda")
                z_test_split1 = z_test_split1.gather(1, test_random_indices)
                
                # if test_random_indices is None:
                #     group_offsets = torch.arange(lg).to("cuda").repeat_interleave(lpc1) * lpc1
                #     group_offsets = group_offsets.unsqueeze(0).expand(lpc2, -1)
                #     test_random_indices = random_indices.repeat(1, len(x[generalization_dim])) + group_offsets  
                # z_test_split1 = z_test_split1.t().gather(1, test_random_indices).t()
                
                z_test_split1 = pca.transform(z_test_split1).reshape([x_test_split1.shape[2], z_train_split0.shape[0], z_train_split0.shape[1]])
                r = pearson_r(z_train_split0, z_test_split1).cpu().numpy()
                cvev = covariance(z_train_split0, z_test_split1).log10().cpu().numpy()
                
                g_test = x_test_split1[generalization_dim].values.repeat(r.shape[1])
                pc = np.tile((np.arange(r.shape[1]) + 1), r.shape[0])
                
                r = r.flatten()
                cvev = cvev.flatten()
                
                df.append(_dict(pc, r, cvev, g_train.values, g_test, subject))
                
                null_r = []
                for seed in range(100):
                    rng_pt1 = torch.Generator().manual_seed(seed)
                    randi = torch.argsort(torch.rand(z_train_split0.size(), generator=rng_pt1), dim=0).to("cuda")
                    randi = torch.stack([randi for _ in range(z_test_split1.size(0))], dim=0)
                    z_test_split1_shuffled = z_test_split1.gather(1, randi)
                    null_r.append(pearson_r(z_train_split0, z_test_split1_shuffled).cpu().numpy().flatten())
                null_r = np.stack(null_r, axis=0)
                split_null_rs.append(null_r)
                
                p = compute_p(r, null_r, 'two_tailed')
                insignificant = p > 0.05
                r[insignificant] = np.nan
                cvev[insignificant] = np.nan
                significant_df.append(_dict(pc, r, cvev, g_train.values, g_test, subject))
            
            subject_dfs.append(pd.concat(df, axis=0).reset_index(drop=True))
            subject_significant_dfs.append(pd.concat(significant_df, axis=0).reset_index(drop=True))
            null_rs.append(np.concatenate(split_null_rs, axis=1))
        
        subject_dfs = aggregate_dfs(subject_dfs)
        subject_significant_dfs = aggregate_dfs(subject_significant_dfs)
        
        _plots(subject_dfs, subject)
        _plots(subject_significant_dfs, subject, subject_prefix="significant")
        
        dfs.append(subject_dfs)
        rs.append(subject_dfs["r"].values)
    
    # Averaging across subjects
    r_avg = np.mean(rs, axis=0)
    dfs = pd.concat(dfs, axis=0)
    subject_dfs["r"] = r_avg
    
    _plots(subject_dfs, "average")
    
    # Null significance test for the average
    null_rs = np.mean(null_rs, axis=0)
    p = compute_p(r_avg, null_rs, 'two_tailed')
    insignificant = p > 0.05
    r_avg[insignificant] = np.nan
    subject_dfs["r"] = r_avg
    _plots(subject_dfs, "average", subject_prefix="significant")
    
    return dfs

def pc_generalization(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    space: str,
    pc_dims: set[str, str],
    generalization_dim: str,
    analysis: str = "multiclass",
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
    split_dim: str = None,
    n_splits: int = 2,
    **kwargs,
):
    generalization_id = f"space={space}.pc_dims={pc_dims[0]}_{pc_dims[1]}.generalization_dim={generalization_dim}"
    if split_dim is not None:
        generalization_id += f".split_dim={split_dim}.n_splits={n_splits}"
    else:
        generalization_id += ".split_dim=None"
    
    temp_cache_path = f"main_analyses/pc_generalization.shuffle_pcdim_same_across_gendim/analysis={analysis}/{generalization_id}"
    cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=True)
    
    target_var = load_target_var(dataset)
    if generalization_dim == "target_var":
        generalization_dim = target_var
    elif pc_dims[0] == "target_var":
        pc_dims = (target_var, pc_dims[1])
    elif pc_dims[1] == "target_var":
        pc_dims = (pc_dims[0], target_var)
    
    match space:
        case "weights":
            X = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, subjects="group", significant_only=True, alpha=alpha, split_dim=split_dim, n_splits=n_splits, n_seeds=1)
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
                
            if pc_dims[0] == "time":
                times = load_significant_times(analysis, dataset, load_dataset_kwargs, scorer_kwargs, feature_dim, alpha, subjects="group").mean("subject").dropna("time").values
                X = X.sel(time=[t in times for t in X.time.values])
             
    reorder_by_cluster = generalization_dim != "time"
    # reorder_by_cluster = False
    
    df_cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    cacher = cache(f"{df_cache_path}/results.pkl",)
    
    return cacher(_pc_generalization)(X, cache_path, pc_dims, generalization_dim, reorder_by_cluster, split_dim=split_dim)