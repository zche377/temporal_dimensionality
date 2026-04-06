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

def _tt_pca_cross_subject(
    X_train: xr.DataArray,
    X_test: xr.DataArray,
    cache_path: Path,
    pc_dims: set[str, str],
    generalization_dim: str,
    split_dim=None
) -> pd.DataFrame:
    if split_dim:
        X_test = X_test.mean("seed")
    
    rs, null_rs, cvnsrs, null_cvnsrs, neg_cvnsrs, null_neg_cvnsrs = [], [], [], [], [], []
    
    spectrum_dict = dict()
    for g_train in X_train[generalization_dim].values:
        spectrum_dict[g_train] = []    
    
    for subject0, subject1 in tqdm(itertools.permutations(X_train.subject.values, 2), desc="subjects"):
        x_train = X_train.sel(subject=subject0).dropna("time")
        x_test = X_test.sel(subject=subject1).dropna("time")
        x_test = x_test.transpose("split", *pc_dims, generalization_dim) if split_dim else x_test.transpose(*pc_dims, generalization_dim)
        
        subject_rs, subject_null_rs, subject_cvnsrs, subject_null_cvnsrs, subject_neg_cvnsrs, subject_null_neg_cvnsrs = [], [], [], [], [], []
        for i, g_train in enumerate(x_train[generalization_dim].values):
            pca = PCA()
            x_train_fit = torch.tensor(x_train.sel({generalization_dim: g_train}).values).to("cuda")
            pca.fit(x_train_fit)
            
            g_rs, g_null_rs, g_cvnsrs, g_null_cvnsrs, g_neg_cvnsrs, g_null_neg_cvnsrs = [], [], [], [], [], []
            for split_pair in itertools.permutations(x_test.split.values, 2) if split_dim else [None]:
                x_test_split0 = torch.tensor(x_test.sel({"split": split_pair[0], generalization_dim: g_train}).values if split_dim else x_test.sel({generalization_dim: g_train}).values).to("cuda")
                
                # x_test_split1 = torch.tensor(x_test.sel({"split": split_pair[1], generalization_dim: g_train}).values if split_dim else x_test.sel({generalization_dim: g_train}).values).to("cuda")
                z_test_split0= pca.transform(x_test_split0)
                
                x_test_split1 = x_test.sel({"split": split_pair[1]}) if split_dim else x_test
                z_test_split1 = pca.transform(torch.tensor(
                    x_test_split1.stack(stack=[generalization_dim, pc_dims[0]]).transpose("stack", pc_dims[1]).values
                )).reshape([x_test_split1.shape[-1], z_test_split0.shape[0], z_test_split0.shape[1]])
                
                # TODO: do we want to spearman brown here for good?
                r = _sb_rel(pearson_r(z_test_split0, z_test_split1))
                # gendim * neuroid
                
                cvev = covariance(z_test_split0, z_test_split1)
                spectrum_dict[g_train].append(pd.DataFrame({
                    "eigenvalue": cvev[i].cpu().numpy(),
                    "pc": np.arange(len(cvev[i]))+1
                }))
                
                null_r = []
                for seed in range(100):
                    rng = torch.Generator().manual_seed(seed)
                    randi = torch.argsort(torch.rand(z_test_split0.size(), generator=rng), dim=0).to("cuda")
                    randi = torch.stack([randi for _ in range(z_test_split1.size(0))], dim=0)
                    z_test_split1_shuffled = z_test_split1.gather(1, randi)
                    null_r.append(_sb_rel(pearson_r(z_test_split0, z_test_split1_shuffled)))
                null_r = torch.stack(null_r, dim=0)
                # permute * gendim * neuroid
                
                p = compute_p(r.cpu().numpy(), null_r.cpu().numpy(), 'greater')
                cvnsr = torch.tensor(np.sum(p < .05, axis=-1)).float()
                p = compute_p(r.cpu().numpy(), null_r.cpu().numpy(), 'less')
                neg_cvnsr = torch.tensor(np.sum(p < .05, axis=-1)).float()
                # gendim
                
                null_r2 = []
                for seed in range(100, 200):
                    rng = torch.Generator().manual_seed(seed)
                    randi = torch.argsort(torch.rand(z_test_split0.size(), generator=rng), dim=0).to("cuda")
                    randi = torch.stack([randi for _ in range(z_test_split1.size(0))], dim=0)
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
                # permute * gendim
                
                g_rs.append(r.cpu())
                g_null_rs.append(null_r.cpu())
                g_cvnsrs.append(cvnsr.cpu())
                g_neg_cvnsrs.append(neg_cvnsr.cpu())
                g_null_cvnsrs.append(null_cvnsr.cpu())
                g_null_neg_cvnsrs.append(null_neg_cvnsr.cpu())
                
            subject_rs.append(torch.stack(g_rs, dim=0))
            subject_null_rs.append(torch.stack(g_null_rs, dim=0))
            subject_cvnsrs.append(torch.stack(g_cvnsrs, dim=0))
            subject_neg_cvnsrs.append(torch.stack(g_neg_cvnsrs, dim=0))
            subject_null_cvnsrs.append(torch.stack(g_null_cvnsrs, dim=0))
            subject_null_neg_cvnsrs.append(torch.stack(g_null_neg_cvnsrs, dim=0))
            # split * ...
                
        rs.append(torch.stack(subject_rs, dim=0).diagonal(dim1=0, dim2=2))
        null_rs.append(torch.stack(subject_null_rs, dim=0).diagonal(dim1=0, dim2=3))
        cvnsrs.append(torch.stack(subject_cvnsrs, dim=0))
        neg_cvnsrs.append(torch.stack(subject_neg_cvnsrs, dim=0))
        null_cvnsrs.append(torch.stack(subject_null_cvnsrs, dim=0))
        null_neg_cvnsrs.append(torch.stack(subject_null_neg_cvnsrs, dim=0))
        # gendim * split * ...
        
    rs = torch.stack(rs, dim=0)
    null_rs = torch.stack(null_rs, dim=0)
    cvnsrs = torch.stack(cvnsrs, dim=0)
    neg_cvnsrs = torch.stack(neg_cvnsrs, dim=0)
    null_cvnsrs = torch.stack(null_cvnsrs, dim=0)
    null_neg_cvnsrs = torch.stack(null_neg_cvnsrs, dim=0)
    # subject_pair * gendim * split * ...
    
    # for g_train in X_train[generalization_dim].values:
    #     if generalization_dim == "behavior":
    #         g_train_alt = g_train.replace('/', '_or_')
    #     else:
    #         g_train_alt = g_train
    #     _plot_eigenspectrum(
    #         pd.concat(spectrum_dict[g_train], axis=0),
    #         cache_path / "eigenspectrum" / f"{g_train_alt}.png"
    #     )
    
    def _to_da(x):
        return xr.DataArray(
            x.cpu(),
            dims=[f"{generalization_dim}_train", f"{generalization_dim}_test"],
            coords={
                f"{generalization_dim}_train": x_train[generalization_dim].values,
                f"{generalization_dim}_test": x_train[generalization_dim].values
            }
        )
    
    def _cc_per_row(x, null_x):
        p = []
        for i in range(x.size(0)):
            p.append(cluster_correction(x[i].cpu().numpy(), null_x[:, i].cpu().numpy()))
        return np.stack(p, axis=0)
            
    mean_cvnsrs = cvnsrs.mean(dim=[0, 2])
    mean_null_cvnsrs = null_cvnsrs.mean(dim=[0, 2]).permute(1, 0, 2)
    p = _cc_per_row(mean_cvnsrs, mean_null_cvnsrs)
    
    mean_neg_cvnsrs = neg_cvnsrs.mean(dim=[0, 2])
    mean_null_neg_cvnsrs = null_cvnsrs.mean(dim=[0, 2]).permute(1, 0, 2)
    neg_p = _cc_per_row(mean_neg_cvnsrs, mean_null_neg_cvnsrs)
    df = _to_da(mean_cvnsrs).to_dataframe("cvnsr").reset_index()
    df["neg_cvnsr"] = _to_da(mean_neg_cvnsrs).to_dataframe("neg_cvnsr").reset_index()["neg_cvnsr"].values
    
    temp = []
    for i in range(len(df)):
        v = np.nanmax([df.loc[i, "cvnsr"], df.loc[i, "neg_cvnsr"]])
        if ~np.isnan(v):
            v *= (.5-np.nanargmax([df.loc[i, "cvnsr"], df.loc[i, "neg_cvnsr"]]))*2
        temp.append(v)
    df["cvnsr"] = temp
    
    p = p.T.flatten()
    neg_p = neg_p.T.flatten()
    cbp = [p[i] if c>0 else neg_p[i] for i, c in enumerate(df.cvnsr.values)]
    df["p"] = cbp
    df["sig"] = [p < .05 for p in cbp]
    
    def _plots(df, subject_prefix=None, order_index=None):
        subject_prefix = f"{subject_prefix}_" if subject_prefix else ""
        
        score_df = df.pivot(index=f"{generalization_dim}_test", columns=f"{generalization_dim}_train", values="cvnsr")
        sig_df = df.pivot(index=f"{generalization_dim}_test", columns=f"{generalization_dim}_train", values="sig")
        
        _plot_gen_with_p(score_df, sig_df, "cvnsr", cache_path=cache_path, cbar_label="dimensionality", cmap=sns.diverging_palette(230, 20, as_cmap=True, sep=1, s=85, l=55))
        return order_index
    
    _plots(df, subject_prefix="significant",)
    
    results = {}
    results["gen_cvnsr"] = xr.DataArray(
        cvnsrs.cpu().numpy(),
        dims=["subject_pair", f"{generalization_dim}_train", "split", f"{generalization_dim}_test",]
    )
    results["gen_null_cvnsr"] = xr.DataArray(
        null_cvnsrs.cpu().numpy(),
        dims=["subject_pair", f"{generalization_dim}_train", "split", "permute", f"{generalization_dim}_test"]
    )
    results["gen_neg_cvnsr"] = xr.DataArray(
        neg_cvnsrs.cpu().numpy(),
        dims=["subject_pair", f"{generalization_dim}_train", "split", f"{generalization_dim}_test",]
    )
    results["gen_null_neg_cvnsr"] = xr.DataArray(
        null_neg_cvnsrs.cpu().numpy(),
        dims=["subject_pair", f"{generalization_dim}_train", "split", "permute", f"{generalization_dim}_test"]
    )
    
    results["r"] = xr.DataArray(
        rs.cpu().numpy(),
        dims=["subject_pair", "split", pc_dims[1], generalization_dim]
    )
    results["null_r"] = xr.DataArray(
        null_rs.cpu().numpy(),
        dims=["subject_pair", "split", "permute", pc_dims[1], generalization_dim]
    )
    # results["cvnsr"] = xr.DataArray(
    #     torch.diagonal(cvnsrs, dim1=1, dim2=3).cpu().numpy(),
    #     dims=["subject_pair", "split", generalization_dim]
    # )
    # results["null_cvnsr"] = xr.DataArray(
    #     torch.diagonal(null_cvnsrs, dim1=1, dim2=4).cpu().numpy(),
    #     dims=["subject_pair", "split", "permute", generalization_dim]
        
    # )
    
    return xr.Dataset(results).assign_coords({
        generalization_dim: x_train[generalization_dim].values,
        f"{generalization_dim}_train": x_train[generalization_dim].values,
        f"{generalization_dim}_test": x_train[generalization_dim].values
    })

def tt_pca_cross_subject(
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
    
    temp_cache_path = f"main_analyses/tt_pca_cross_subject/analysis={analysis}/{generalization_id}"
    cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, None if space == "eeg" else scorer_kwargs, include_root=True)
    
    target_var = load_target_var(dataset)
    if generalization_dim == "target_var":
        generalization_dim = target_var
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
    
    df_cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    cacher = cache(
        f"{df_cache_path}/results.nc", 
        # mode="ignore"
    )
    
    return cacher(_tt_pca_cross_subject)(X_train, X_test, cache_path, pc_dims, generalization_dim, split_dim=split_dim)