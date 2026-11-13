import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN axis')

import gc
import os
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
from lib.utilities import SEED, _append_path
from lib.computation.statistics import compute_p
from lib.computation.scorers import TrainTestModelScorer

import matplotlib.colors as mcolors
import colorsys

from bonner.computation.decomposition import PCA
from bonner.computation.metrics import covariance, pearson_r
from bonner.caching import cache
from bonner.datasets.hebart2022_things_behavior import load_embeddings



def _sb_rel(x, n=2):
        return n*x/(1+(n-1)*x)


# Stringer-path precision: fp64 gesvd/covariance on GPU is ~8-32x slower than fp32; the neural .nc is
# float64. Cast the Stringer cvPCA tensors to fp32 (+ TF32 matmul) for the win. dtype-conditional
# default-ON on the Stringer path only (its caches carry the .split_mode=stringer tag, disjoint from the
# published fp64 train-basis caches). Toggle OFF for an fp64 reference: TDIM_STRINGER_FP32=0.
_STRINGER_FP32 = os.environ.get("TDIM_STRINGER_FP32", "1") == "1"


# =============================================================================
# Stringer per-image half-rep cvPCA (gated behind per_image; all of the below is
# UNUSED on the default path, so `_tt_pca` / results.nc stay byte-identical).
# =============================================================================

def _gpu_perm_null(z0, z1, n_perms, perm_batch, base_seed, maybe_sb):
    """GPU-RNG permutation null of pearson_r(z0, z1) with per-feature sample shuffles.

    z0: (n_samples, n_feat); z1: (n_gen, n_samples, n_feat). Returns (n_perms, n_gen, n_feat) on CUDA.
    Draws the shuffles directly on CUDA (one torch.Generator(device='cuda')) instead of the default
    path's ~2000 CPU argsorts -> different random stream (validated at the distribution level, #4).
    """
    n_gen = z1.size(0)
    n_samples, n_feat = z0.size(0), z0.size(1)
    gen = torch.Generator(device="cuda"); gen.manual_seed(int(base_seed))
    out = []
    for start in range(0, n_perms, perm_batch):
        b = min(perm_batch, n_perms - start)
        # per-(perm, feature) independent shuffle of the sample axis (matches the default null)
        randi = torch.argsort(torch.rand((b, n_samples, n_feat), device="cuda", generator=gen), dim=1)
        randi = randi.unsqueeze(1).expand(-1, n_gen, -1, -1)                      # (b, n_gen, n_s, n_f)
        z1_sh = z1.unsqueeze(0).expand(b, -1, -1, -1).gather(2, randi)
        z1_flat = z1_sh.reshape(b * n_gen, n_samples, n_feat)
        z0_tiled = z0.unsqueeze(0).expand(b * n_gen, -1, -1)
        r = maybe_sb(pearson_r(z0_tiled, z1_flat)).reshape(b, n_gen, n_feat)
        out.append(r)
        del randi, z1_sh, z1_flat, z0_tiled, r
    return torch.cat(out, dim=0)


_NULL_COUNT_MAX_ELEMENTS = 1_200_000_000


def _null_count_block(null_r, null_r2, n_perms, perm_batch, alpha):
    """The O(n_perm^2) null-vs-null block: count significant PCs per gen cell for each null draw.

    null_r, null_r2: (n_perms, n_gen, n_feat) CUDA (seed-mean). Returns (null_cvnsr, null_neg_cvnsr),
    each (n_perms, n_gen) on CPU. Mirrors the default path's second-null block exactly.

    Each comparison is summed over a (batch, n_perms, n_gen, n_feat) tensor that ``.sum`` promotes to
    int64, so the batch is capped at ``_NULL_COUNT_MAX_ELEMENTS`` elements (~9.6 GB) to fit the GPU when
    n_feat (the number of PCs, i.e. channels) is large. Rows are counted independently, so the cap
    changes memory only, never the counts.
    """
    null_r2_t = null_r2.unsqueeze(0)                                     # (1, n_perms, n_gen, n_feat)
    denominator = n_perms + 1
    per_row = n_perms * null_r2.shape[1] * null_r2.shape[2]
    perm_batch = max(1, min(perm_batch, _NULL_COUNT_MAX_ELEMENTS // per_row))
    pos_list, neg_list = [], []
    for start in range(0, n_perms, perm_batch):
        end = min(start + perm_batch, n_perms)
        nb = null_r[start:end].unsqueeze(1)                             # (b, 1, n_gen, n_feat)
        cg = (nb > null_r2_t).sum(dim=1)
        cl = (nb < null_r2_t).sum(dim=1)
        del nb
        pg = 1 - ((cg + 1) / denominator)
        pl = 1 - ((cl + 1) / denominator)
        pos_list.append((pg < alpha).sum(dim=-1).float().cpu())
        neg_list.append((pl < alpha).sum(dim=-1).float().cpu())
        del cg, cl, pg, pl
    return torch.cat(pos_list, dim=0), torch.cat(neg_list, dim=0)


def _scheme_x_axis_counts(r_axis, null_r_axis, alpha):
    """Scheme X (cluster@Layer-1): per-PC temporal cluster_correction, count reliable PCs per cell.

    r_axis: (n_feat, axis); null_r_axis: (n_perm, n_feat, axis) [numpy]. Returns (pos, neg), each
    (axis,). This is fig_2/4's current recipe generalised to any 1-D axis (time for the diagonal,
    t_test per t_train for the generalization). Negative side = cluster on the flipped sign.
    """
    n_feat, axis = r_axis.shape
    pos = np.zeros(axis); neg = np.zeros(axis)
    for f in range(n_feat):
        pos += (cluster_correction(r_axis[f], null_r_axis[:, f], alpha=alpha) < alpha)
        neg += (cluster_correction(-r_axis[f], -null_r_axis[:, f], alpha=alpha) < alpha)
    return pos, neg


def _signed_combine(pos, neg):
    """Per-cell max-abs-with-sign combine of the positive/negative counts (nan-safe).

    Matches load_cvpca_generalization_(streaming|per_subject): +count for positive dims, -count for
    negative, whichever has larger magnitude.
    """
    stacked = np.stack([pos, neg], axis=0)
    all_nan = np.all(np.isnan(stacked), axis=0)
    filled = np.where(np.isnan(stacked), -np.inf, stacked)
    magnitude = np.max(filled, axis=0)
    sign = (0.5 - np.argmax(filled, axis=0)) * 2
    signed = magnitude * sign
    signed[all_nan] = np.nan
    return signed


def _layer2_row_sig(count, null_count, neg_count, null_neg_count, alpha):
    """Scheme Y Layer-2 on one 1-D axis: cluster the COUNT vs the null-COUNT, sign-aware.

    count/neg_count: (axis,); null_count/null_neg_count: (perm, axis). Uses gen_null_neg for the
    negative side (the neg-null bug FIX). Returns a boolean sig mask (axis,).
    """
    pos_p = cluster_correction(count, null_count, alpha=alpha)
    neg_p = cluster_correction(neg_count, null_neg_count, alpha=alpha)
    signed = _signed_combine(count, neg_count)
    return np.where(signed > 0, pos_p < alpha, neg_p < alpha)


def _tt_pca(
    X_train: xr.DataArray,
    X_test: xr.DataArray,
    cache_path: Path,
    pc_dims: set[str, str],
    generalization_dim: str,
    split_dim=None,
    pca_source: str = "train",
) -> pd.DataFrame:
    if split_dim:
        X_test = X_test.mean("seed")
    
    rs, null_rs, cvnsrs, null_cvnsrs, neg_cvnsrs, null_neg_cvnsrs = [], [], [], [], [], []

    # Save coordinate values before loop (needed for final dataset creation)
    gen_dim_coords = X_train[generalization_dim].values

    spectrum_dict = dict()
    for g_train in gen_dim_coords:
        spectrum_dict[g_train] = []

    for subject in tqdm(X_train.subject.values, desc="subject"):
        x_train = X_train.sel(subject=subject).dropna("time")
        x_test = X_test.sel(subject=subject).dropna("time")
        x_test = x_test.transpose("split", *pc_dims, generalization_dim) if split_dim else x_test.transpose(*pc_dims, generalization_dim)
        
        subject_rs_diag, subject_null_rs_diag = [], []  # Only store diagonal elements
        subject_cvnsrs, subject_null_cvnsrs, subject_neg_cvnsrs, subject_null_neg_cvnsrs = [], [], [], []
        for i, g_train in enumerate(x_train[generalization_dim].values):
            # train mode: fit ONE basis on the train set per g_train (reused for both split orderings).
            # test mode: the basis is fit on the reference test split inside the split_pair loop below,
            # so it is fit twice per g_train (once per ordering, on a different split).
            x_train_fit = None
            if pca_source == "train":
                pca = PCA()
                x_train_fit = torch.tensor(x_train.sel({generalization_dim: g_train}).values).to("cuda")
                pca.fit(x_train_fit)

            g_rs_diag, g_null_rs_diag = [], []  # Only store diagonal (index i)
            g_cvnsrs, g_null_cvnsrs, g_neg_cvnsrs, g_null_neg_cvnsrs = [], [], [], []
            for split_pair in itertools.permutations(x_test.split.values, 2) if split_dim else [None]:
                x_test_split0 = torch.tensor(x_test.sel({"split": split_pair[0], generalization_dim: g_train}).values if split_dim else x_test.sel({generalization_dim: g_train}).values).to("cuda")

                # test mode: fit the PCA basis on the reference test split (split_pair[0]) at g_train
                # (within-test cvPCA), rather than reusing a train-set basis.
                if pca_source == "test":
                    pca = PCA()
                    pca.fit(x_test_split0)

                z_test_split0= pca.transform(x_test_split0)
                
                x_test_split1 = x_test.sel({"split": split_pair[1]}) if split_dim else x_test
                z_test_split1 = pca.transform(torch.tensor(
                    x_test_split1.stack(stack=[generalization_dim, pc_dims[0]]).transpose("stack", pc_dims[1]).values
                ).to("cuda")).reshape([x_test_split1.shape[-1], z_test_split0.shape[0], z_test_split0.shape[1]])
                
                # TODO: spearman brown here for good?
                r = _sb_rel(pearson_r(z_test_split0, z_test_split1))
                # gendim * neuroid
                
                cvev = covariance(z_test_split0, z_test_split1)
                spectrum_dict[g_train].append(pd.DataFrame({
                    "eigenvalue": cvev[i].cpu().numpy(),
                    "pc": np.arange(len(cvev[i]))+1
                }))
                
                # Batched permutation test for null distribution
                n_perms = 1000
                perm_batch_size = 100
                null_r = []
                n_gen = z_test_split1.size(0)
                for batch_start in range(0, n_perms, perm_batch_size):
                    batch_end = min(batch_start + perm_batch_size, n_perms)
                    batch_size = batch_end - batch_start

                    # Generate all random indices for this batch: (batch, n_samples, n_features)
                    all_randi = torch.stack([
                        torch.argsort(torch.rand(z_test_split0.size(), generator=torch.Generator().manual_seed(seed)), dim=0)
                        for seed in range(batch_start, batch_end)
                    ], dim=0).to("cuda")

                    # Expand for gen_dim: (batch, gen_dim, n_samples, n_features)
                    all_randi = all_randi.unsqueeze(1).expand(-1, n_gen, -1, -1)
                    z_test_split1_expanded = z_test_split1.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    z_test_split1_shuffled = z_test_split1_expanded.gather(2, all_randi)
                    del all_randi

                    # pearson_r only supports up to 3D, so flatten batch and gen_dim
                    # z_test_split1_shuffled: (batch, gen_dim, n_samples, n_features) -> (batch*gen_dim, n_samples, n_features)
                    z_shuffled_flat = z_test_split1_shuffled.reshape(batch_size * n_gen, z_test_split1.size(1), z_test_split1.size(2))
                    del z_test_split1_shuffled
                    # z_test_split0: (n_samples, n_features) -> tile to (batch*gen_dim, n_samples, n_features)
                    z0_tiled = z_test_split0.unsqueeze(0).expand(batch_size * n_gen, -1, -1)

                    # Compute correlations and reshape back: (batch*gen_dim, n_features) -> (batch, gen_dim, n_features)
                    batch_r = _sb_rel(pearson_r(z0_tiled, z_shuffled_flat)).reshape(batch_size, n_gen, -1)
                    del z_shuffled_flat, z0_tiled
                    null_r.append(batch_r.cpu())
                    del batch_r
                null_r = torch.cat(null_r, dim=0)
                # permute * gendim * neuroid
                
                r_cpu = r.cpu()
                null_r_np = null_r.numpy()
                p = compute_p(r_cpu.numpy(), null_r_np, 'greater')
                cvnsr = torch.tensor(np.sum(p < .05, axis=-1)).float()
                p = compute_p(r_cpu.numpy(), null_r_np, 'less')
                neg_cvnsr = torch.tensor(np.sum(p < .05, axis=-1)).float()
                del p
                # gendim

                # Batched permutation test for second null distribution
                null_r2 = []
                for batch_start in range(n_perms, 2 * n_perms, perm_batch_size):
                    batch_end = min(batch_start + perm_batch_size, 2 * n_perms)
                    batch_size = batch_end - batch_start

                    # Generate all random indices for this batch: (batch, n_samples, n_features)
                    all_randi = torch.stack([
                        torch.argsort(torch.rand(z_test_split0.size(), generator=torch.Generator().manual_seed(seed)), dim=0)
                        for seed in range(batch_start, batch_end)
                    ], dim=0).to("cuda")

                    # Expand for gen_dim: (batch, gen_dim, n_samples, n_features)
                    all_randi = all_randi.unsqueeze(1).expand(-1, n_gen, -1, -1)
                    z_test_split1_expanded = z_test_split1.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    z_test_split1_shuffled = z_test_split1_expanded.gather(2, all_randi)
                    del all_randi

                    # Flatten batch and gen_dim for pearson_r
                    z_shuffled_flat = z_test_split1_shuffled.reshape(batch_size * n_gen, z_test_split1.size(1), z_test_split1.size(2))
                    del z_test_split1_shuffled
                    z0_tiled = z_test_split0.unsqueeze(0).expand(batch_size * n_gen, -1, -1)

                    # Compute correlations and reshape back
                    batch_r = _sb_rel(pearson_r(z0_tiled, z_shuffled_flat)).reshape(batch_size, n_gen, -1)
                    del z_shuffled_flat, z0_tiled
                    null_r2.append(batch_r.cpu())
                    del batch_r
                null_r2 = torch.cat(null_r2, dim=0).numpy()
                
                # Batched p-value computation for null distribution
                # null_r: (n_perms, n_gen, n_features), null_r2: (n_perms, n_gen, n_features)
                # We want to compare each null_r[i] against all of null_r2
                null_r2_tensor = torch.tensor(null_r2).to("cuda").unsqueeze(0)  # (1, n_perms, n_gen, n_features)
                del null_r2
                denominator = n_perms + 1

                p_batch_size = perm_batch_size  # reuse the same batch size
                null_cvnsr_list = []
                null_neg_cvnsr_list = []
                for batch_start in range(0, n_perms, p_batch_size):
                    batch_end = min(batch_start + p_batch_size, n_perms)

                    # Get batch of null_r: (batch, n_gen, n_features) -> (batch, 1, n_gen, n_features)
                    null_r_batch = null_r[batch_start:batch_end].unsqueeze(1).to("cuda")

                    # Count comparisons against all of null_r2
                    count_greater = (null_r_batch > null_r2_tensor).sum(dim=1)
                    count_less = (null_r_batch < null_r2_tensor).sum(dim=1)
                    del null_r_batch

                    p_greater = 1 - ((count_greater + 1) / denominator)
                    p_less = 1 - ((count_less + 1) / denominator)
                    del count_greater, count_less

                    # Count significant features per gen_dim: (batch, n_gen)
                    null_cvnsr_list.append((p_greater < .05).sum(dim=-1).float().cpu())
                    null_neg_cvnsr_list.append((p_less < .05).sum(dim=-1).float().cpu())
                    del p_greater, p_less

                del null_r2_tensor
                null_cvnsr = torch.cat(null_cvnsr_list, dim=0)
                null_neg_cvnsr = torch.cat(null_neg_cvnsr_list, dim=0)
                del null_cvnsr_list, null_neg_cvnsr_list
                # permute * gendim

                # Only store diagonal element (index i) for r and null_r to save memory
                g_rs_diag.append(r_cpu[i])  # neuroid
                g_null_rs_diag.append(null_r[:, i])  # permute * neuroid
                g_cvnsrs.append(cvnsr)
                g_neg_cvnsrs.append(neg_cvnsr)
                g_null_cvnsrs.append(null_cvnsr)
                g_null_neg_cvnsrs.append(null_neg_cvnsr)

                # Free GPU memory after each split pair
                del r, r_cpu, null_r, null_r_np, cvnsr, neg_cvnsr, null_cvnsr, null_neg_cvnsr
                del x_test_split0, z_test_split0
                torch.cuda.empty_cache()
                
            # Stack diagonal elements: split * neuroid (for rs) and split * permute * neuroid (for null_rs)
            subject_rs_diag.append(torch.stack(g_rs_diag, dim=0))  # split * neuroid
            subject_null_rs_diag.append(torch.stack(g_null_rs_diag, dim=0))  # split * permute * neuroid
            subject_cvnsrs.append(torch.stack(g_cvnsrs, dim=0))
            subject_neg_cvnsrs.append(torch.stack(g_neg_cvnsrs, dim=0))
            subject_null_cvnsrs.append(torch.stack(g_null_cvnsrs, dim=0))
            subject_null_neg_cvnsrs.append(torch.stack(g_null_neg_cvnsrs, dim=0))
            # split * ...

            # Free memory after each generalization dimension
            del g_rs_diag, g_null_rs_diag, g_cvnsrs, g_neg_cvnsrs, g_null_cvnsrs, g_null_neg_cvnsrs
            del x_train_fit, pca, z_test_split1, cvev
            gc.collect()
            torch.cuda.empty_cache()

        # Stack subject results - diagonal already extracted, just stack by gen_dim
        # subject_rs_diag is list of (split * neuroid) for each gen_dim
        # We want: split * neuroid * gen_dim
        stacked_rs_diag = torch.stack(subject_rs_diag, dim=-1)  # split * neuroid * gen_dim
        stacked_null_rs_diag = torch.stack(subject_null_rs_diag, dim=-1)  # split * permute * neuroid * gen_dim

        rs.append(stacked_rs_diag)
        # split * neuroid * gen_dim
        null_rs.append(stacked_null_rs_diag)
        # split * permute * neuroid * gen_dim
        cvnsrs.append(torch.stack(subject_cvnsrs, dim=0))
        neg_cvnsrs.append(torch.stack(subject_neg_cvnsrs, dim=0))
        null_cvnsrs.append(torch.stack(subject_null_cvnsrs, dim=0))
        null_neg_cvnsrs.append(torch.stack(subject_null_neg_cvnsrs, dim=0))
        # gendim * split * ...

        # Free memory after each subject
        del subject_rs_diag, subject_null_rs_diag, subject_cvnsrs, subject_neg_cvnsrs
        del subject_null_cvnsrs, subject_null_neg_cvnsrs
        del stacked_rs_diag, stacked_null_rs_diag
        del x_train, x_test
        gc.collect()
        torch.cuda.empty_cache()
        
    rs = torch.stack(rs, dim=0)
    null_rs = torch.stack(null_rs, dim=0)
    # subject * split * ...
    cvnsrs = torch.stack(cvnsrs, dim=0)
    neg_cvnsrs = torch.stack(neg_cvnsrs, dim=0)
    null_cvnsrs = torch.stack(null_cvnsrs, dim=0)
    null_neg_cvnsrs = torch.stack(null_neg_cvnsrs, dim=0)
    # subject * gendim * split * ...
    
    # for g_train in X_train[generalization_dim].values:
    #     if generalization_dim == "behavior":
    #         g_train_alt = g_train.replace('/', '_or_')
    #     else:
    #         g_train_alt = g_train
    #     _plot_eigenspectrum(
    #         pd.concat(spectrum_dict[g_train], axis=0),
    #         cache_path / "eigenspectrum" / f"{g_train_alt}.png"
    #     )
    
    # def _to_da(x):
    #     return xr.DataArray(
    #         x.cpu(),
    #         dims=[f"{generalization_dim}_train", f"{generalization_dim}_test"],
    #         coords={
    #             f"{generalization_dim}_train": x_train[generalization_dim].values,
    #             f"{generalization_dim}_test": x_train[generalization_dim].values
    #         }
    #     )
    
    # def _cc_per_row(x, null_x):
    #     p = []
    #     for i in range(x.size(0)):
    #         p.append(cluster_correction(x[i].cpu().numpy(), null_x[:, i].cpu().numpy()))
    #     return np.stack(p, axis=0)
            
    # mean_cvnsrs = cvnsrs.mean(dim=[0, 2])
    # mean_null_cvnsrs = null_cvnsrs.mean(dim=[0, 2]).permute(1, 0, 2)
    # p = _cc_per_row(mean_cvnsrs, mean_null_cvnsrs)
    
    # mean_neg_cvnsrs = neg_cvnsrs.mean(dim=[0, 2])
    # mean_null_neg_cvnsrs = null_cvnsrs.mean(dim=[0, 2]).permute(1, 0, 2)
    # neg_p = _cc_per_row(mean_neg_cvnsrs, mean_null_neg_cvnsrs)
    # df = _to_da(mean_cvnsrs).to_dataframe("cvnsr").reset_index()
    # df["neg_cvnsr"] = _to_da(mean_neg_cvnsrs).to_dataframe("neg_cvnsr").reset_index()["neg_cvnsr"].values
    
    # temp = []
    # for i in range(len(df)):
    #     v = np.nanmax([df.loc[i, "cvnsr"], df.loc[i, "neg_cvnsr"]])
    #     if ~np.isnan(v):
    #         v *= (.5-np.nanargmax([df.loc[i, "cvnsr"], df.loc[i, "neg_cvnsr"]]))*2
    #     temp.append(v)
    # df["cvnsr"] = temp
    
    # p = p.T.flatten()
    # neg_p = neg_p.T.flatten()
    # cbp = [p[i] if c>0 else neg_p[i] for i, c in enumerate(df.cvnsr.values)]
    # df["p"] = cbp
    # df["sig"] = [p < .05 for p in cbp]
    
    # def _plots(df, subject_prefix=None, order_index=None):
    #     subject_prefix = f"{subject_prefix}_" if subject_prefix else ""
        
    #     score_df = df.pivot(index=f"{generalization_dim}_test", columns=f"{generalization_dim}_train", values="cvnsr")
    #     sig_df = df.pivot(index=f"{generalization_dim}_test", columns=f"{generalization_dim}_train", values="sig")
        
    #     def get_hue_from_color_name(color_name):
    #         # Get RGB values from matplotlib color name
    #         rgb = mcolors.to_rgb(color_name)
            
    #         # Convert RGB to HSV (hue, saturation, value)
    #         h, s, v = colorsys.rgb_to_hsv(*rgb)
            
    #         # Convert hue from [0,1] range to [0,359] degrees
    #         return h * 359
        
    #     indianred_hue = get_hue_from_color_name("indianred")
        
    #     _plot_gen_with_p(score_df, sig_df, "cvnsr", cache_path=cache_path, cbar_label="dimensionality", cmap=sns.diverging_palette(270, indianred_hue, as_cmap=True))
    #     return order_index
    
    # _plots(df, subject_prefix="significant",)
    
    results = {}
    # results["gen_r"] = xr.DataArray(
    #     rs.cpu().numpy(),
    #     dims=["subject", f"{generalization_dim}_train", "split", f"{generalization_dim}_test", pc_dims[1]]
    # )
    # results["gen_null_r"] = xr.DataArray(
    #     null_rs.cpu().numpy(),
    #     dims=["subject", f"{generalization_dim}_train", "split", "permute", f"{generalization_dim}_test", pc_dims[1]]
    # )
    results["gen_cvnsr"] = xr.DataArray(
        cvnsrs.cpu().numpy(),
        dims=["subject", f"{generalization_dim}_train", "split", f"{generalization_dim}_test",]
    )
    results["gen_null_cvnsr"] = xr.DataArray(
        null_cvnsrs.cpu().numpy(),
        dims=["subject", f"{generalization_dim}_train", "split", "permute", f"{generalization_dim}_test"]
    )
    results["gen_neg_cvnsr"] = xr.DataArray(
        neg_cvnsrs.cpu().numpy(),
        dims=["subject", f"{generalization_dim}_train", "split", f"{generalization_dim}_test",]
    )
    results["gen_null_neg_cvnsr"] = xr.DataArray(
        null_neg_cvnsrs.cpu().numpy(),
        dims=["subject", f"{generalization_dim}_train", "split", "permute", f"{generalization_dim}_test"]
    )
    
    results["r"] = xr.DataArray(
        rs.cpu().numpy(),
        dims=["subject", "split", pc_dims[1], generalization_dim]
    )
    results["null_r"] = xr.DataArray(
        null_rs.cpu().numpy(),
        dims=["subject", "split", "permute", pc_dims[1], generalization_dim]
    )
    # results["cvnsr"] = xr.DataArray(
    #     torch.diagonal(cvnsrs, dim1=1, dim2=3).cpu().numpy(),
    #     dims=["subject", "split", generalization_dim]
    # )
    # results["null_cvnsr"] = xr.DataArray(
    #     torch.diagonal(null_cvnsrs, dim1=1, dim2=4).cpu().numpy(),
    #     dims=["subject", "split", "permute", generalization_dim]
    # )
    
    return xr.Dataset(results).assign_coords({
        generalization_dim: gen_dim_coords,
        f"{generalization_dim}_train": gen_dim_coords,
        f"{generalization_dim}_test": gen_dim_coords
    })

def _tt_pca_stringer(
    X_train: xr.DataArray,
    X_test: xr.DataArray,
    pc_dims,
    generalization_dim: str,
    n_seeds: int = 1,
    spearman_brown: bool = True,
    alpha: float = 0.05,
) -> xr.Dataset:
    """Stringer per-image half-rep cvPCA with IN-LIB significance (drop the raw null).

    Averages the per-cell r / null_r / null_r2 across the `n_seeds` rep-splits into a single seed-mean
    (valid: the PCA basis is TRAIN-fit -> seed-independent, so PC #k is the same direction every seed),
    then runs the O(n^2) null-vs-null block + ALL significance ONCE on the seed-mean. Only the per-seed
    r/null compute is x n_seeds. Stores compact seed-mean observed counts + significance (BOTH scheme X
    and scheme Y, group + per-subject, diagonal + generalization) and drops r / null_r / gen_null_*.
    """
    maybe_sb = (lambda x: x) if not spearman_brown else _sb_rel
    n_perms, perm_batch = 1000, 100
    BASE_POS, BASE_NEG = 0, n_perms

    # Track-B precision: cast the fp64 neural tensors to fp32 (+ TF32 matmul) for the gesvd/covariance/
    # null speedup. Everything downstream (PCA fit/transform, pearson_r, covariance, null) inherits the
    # dtype from these inputs. Toggle via TDIM_STRINGER_FP32 (default on) for the fp64-vs-fp32 comparison.
    if _STRINGER_FP32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    _t = (lambda a: torch.tensor(a).to("cuda").float()) if _STRINGER_FP32 else (lambda a: torch.tensor(a).to("cuda"))

    gen_coords = X_train[generalization_dim].values
    n_time = len(gen_coords)
    split_vals = list(X_test.split.values)
    orderings = list(itertools.permutations(split_vals, 2))
    n_ord = len(orderings)

    # per-subject stored outputs (seed-mean)
    o_cvnsr_X, o_neg_cvnsr_X = [], []            # (subject, split, time)
    o_gen_X, o_gen_neg_X = [], []                # (subject, time_train, split, time_test)
    o_cvnsr_Y, o_neg_cvnsr_Y = [], []            # (subject, split, time)
    o_gen_Y, o_gen_neg_Y = [], []                # (subject, time_train, split, time_test)
    o_perseed_Y = []                             # (subject, seed, split, time) -- stability band
    o_sig_diag_subj, o_sig_gen_subj = [], []     # per-subject scheme-Y sig

    # group accumulators (running sum over subject x ordering)
    g_diag_c = g_diag_n = g_diag_nc = g_diag_nnc = None      # diagonal: count(time,), nullcount(perm,time)
    g_gen_c = g_gen_n = g_gen_nc = g_gen_nnc = None          # gen: count(tt,te), nullcount(tt,perm,te)
    n_grp = 0

    for subject in tqdm(X_train.subject.values, desc="subject"):
        xt = X_train.sel(subject=subject).dropna("time").transpose(*pc_dims, generalization_dim)
        x_test = X_test.sel(subject=subject).dropna("time").transpose("seed", "split", *pc_dims, generalization_dim)

        # per (ordering) collectors over g_train
        ord_gen_Y = np.zeros((n_ord, n_time, n_time))       # (ord, t_train, t_test)  scheme-Y count
        ord_gen_negY = np.zeros((n_ord, n_time, n_time))
        ord_gen_X = np.zeros((n_ord, n_time, n_time))       # scheme-X count
        ord_gen_negX = np.zeros((n_ord, n_time, n_time))
        ord_gen_nullc = np.zeros((n_ord, n_time, n_perms, n_time))   # (ord, t_train, perm, t_test)
        ord_gen_nullnegc = np.zeros((n_ord, n_time, n_perms, n_time))
        ord_rdiag, ord_nulldiag = [None] * n_ord, [None] * n_ord     # scheme-X diagonal inputs
        perseed_Y = np.zeros((n_ord, n_seeds, n_time))

        for oi, (sp0, sp1) in enumerate(orderings):
            rdiag = np.zeros((n_time, 0))   # placeholder, replaced below
            rdiag_list, nulldiag_list = [], []
            for i, g_train in enumerate(xt[generalization_dim].values):
                pca = PCA()
                pca.fit(_t(xt.sel({generalization_dim: g_train}).values))

                r_acc = null_r_acc = null_r2_acc = None
                for seed in range(n_seeds):
                    xs = x_test.sel(seed=x_test.seed.values[seed])
                    x0 = _t(xs.sel(split=sp0).sel({generalization_dim: g_train}).values)
                    z0 = pca.transform(x0)                                  # (n_samples, n_feat)
                    x1 = xs.sel(split=sp1)                                  # (pc0, pc1, gen)
                    z1 = pca.transform(_t(
                        x1.stack(stack=[generalization_dim, pc_dims[0]]).transpose("stack", pc_dims[1]).values
                    )).reshape([x1.shape[-1], z0.shape[0], z0.shape[1]])   # (n_gen, n_s, n_feat)

                    r_s = maybe_sb(pearson_r(z0, z1))                       # (n_gen, n_feat), matches _tt_pca
                    null_r_s = _gpu_perm_null(z0, z1, n_perms, perm_batch, BASE_POS, maybe_sb)
                    null_r2_s = _gpu_perm_null(z0, z1, n_perms, perm_batch, BASE_NEG, maybe_sb)

                    # per-seed diagonal scheme-Y count (cheap, no clustering) for the stability band
                    perseed_Y[oi, seed, i] = float((compute_p(r_s[i], null_r_s[:, i], 'greater') < alpha).sum().cpu())

                    r_acc = r_s if r_acc is None else r_acc + r_s
                    null_r_acc = null_r_s if null_r_acc is None else null_r_acc + null_r_s
                    null_r2_acc = null_r2_s if null_r2_acc is None else null_r2_acc + null_r2_s
                    del r_s, null_r_s, null_r2_s, z0, z1, x0
                del pca

                r = r_acc / n_seeds; null_r = null_r_acc / n_seeds; null_r2 = null_r2_acc / n_seeds
                del r_acc, null_r_acc, null_r2_acc

                # --- Scheme Y Layer-1: per-cell compute_p count (positive + negative) ---
                ord_gen_Y[oi, i] = (compute_p(r, null_r, 'greater') < alpha).sum(-1).cpu().numpy()
                ord_gen_negY[oi, i] = (compute_p(r, null_r, 'less') < alpha).sum(-1).cpu().numpy()

                # --- O(n^2) null-vs-null block -> null counts for the Layer-2 cluster ---
                nc, nnc = _null_count_block(null_r, null_r2, n_perms, perm_batch, alpha)
                ord_gen_nullc[oi, i] = nc.numpy()
                ord_gen_nullnegc[oi, i] = nnc.numpy()
                del null_r2, nc, nnc

                # --- Scheme X Layer-1: per-PC cluster over t_test ---
                r_np = r.cpu().numpy(); null_r_np = null_r.cpu().numpy()
                px, nx = _scheme_x_axis_counts(
                    np.moveaxis(r_np, 0, -1),            # (n_feat, t_test)
                    np.moveaxis(null_r_np, 1, -1),       # (perm, n_feat, t_test)
                    alpha,
                )
                ord_gen_X[oi, i] = px; ord_gen_negX[oi, i] = nx

                # diagonal inputs for the scheme-X temporal cluster (per g_train diagonal cell = i)
                rdiag_list.append(r_np[i])               # (n_feat,)
                nulldiag_list.append(null_r_np[:, i])    # (perm, n_feat)
                del r, null_r, r_np, null_r_np
                gc.collect(); torch.cuda.empty_cache()

            ord_rdiag[oi] = np.stack(rdiag_list, axis=0)       # (time, n_feat)
            ord_nulldiag[oi] = np.stack(nulldiag_list, axis=0) # (time, perm, n_feat)

        # ---- per subject: scheme-X diagonal (cluster the diagonal r over TIME, per PC) ----
        cvnsr_X = np.zeros((n_ord, n_time)); neg_cvnsr_X = np.zeros((n_ord, n_time))
        for oi in range(n_ord):
            px, nx = _scheme_x_axis_counts(
                np.moveaxis(ord_rdiag[oi], 0, -1),        # (n_feat, time)
                np.moveaxis(ord_nulldiag[oi], 0, -1),     # (perm, n_feat, time)  <- from (time,perm,feat)
                alpha,
            )
            cvnsr_X[oi] = px; neg_cvnsr_X[oi] = nx

        # ---- per subject: scheme-Y diagonal counts = diagonal of the per-cell generalization count ----
        cvnsr_Y = np.stack([np.diagonal(ord_gen_Y[oi]) for oi in range(n_ord)], axis=0)      # (ord, time)
        neg_cvnsr_Y = np.stack([np.diagonal(ord_gen_negY[oi]) for oi in range(n_ord)], axis=0)
        # diagonal null counts (perm, time) per ordering, for the diagonal Layer-2 cluster
        diag_nullc = np.stack([np.stack([ord_gen_nullc[oi, t, :, t] for t in range(n_time)], axis=-1) for oi in range(n_ord)], axis=0)     # (ord, perm, time)
        diag_nullnegc = np.stack([np.stack([ord_gen_nullnegc[oi, t, :, t] for t in range(n_time)], axis=-1) for oi in range(n_ord)], axis=0)

        # ---- per-subject significance (scheme Y), averaged over ordering ----
        sig_diag_subj = _layer2_row_sig(
            cvnsr_Y.mean(0), diag_nullc.mean(0), neg_cvnsr_Y.mean(0), diag_nullnegc.mean(0), alpha
        )
        gen_Y_m = ord_gen_Y.mean(0); gen_negY_m = ord_gen_negY.mean(0)
        gen_nullc_m = ord_gen_nullc.mean(0); gen_nullnegc_m = ord_gen_nullnegc.mean(0)
        sig_gen_subj = np.stack([
            _layer2_row_sig(gen_Y_m[t], gen_nullc_m[t], gen_negY_m[t], gen_nullnegc_m[t], alpha)
            for t in range(n_time)
        ], axis=0)

        # ---- accumulate group sums (over subject x ordering) ----
        for oi in range(n_ord):
            g_diag_c = _acc(g_diag_c, cvnsr_Y[oi]); g_diag_n = _acc(g_diag_n, neg_cvnsr_Y[oi])
            g_diag_nc = _acc(g_diag_nc, diag_nullc[oi]); g_diag_nnc = _acc(g_diag_nnc, diag_nullnegc[oi])
            g_gen_c = _acc(g_gen_c, ord_gen_Y[oi]); g_gen_n = _acc(g_gen_n, ord_gen_negY[oi])
            g_gen_nc = _acc(g_gen_nc, ord_gen_nullc[oi]); g_gen_nnc = _acc(g_gen_nnc, ord_gen_nullnegc[oi])
            n_grp += 1

        # ---- stash per-subject outputs (split axis = ordering) ----
        o_cvnsr_X.append(cvnsr_X); o_neg_cvnsr_X.append(neg_cvnsr_X)
        o_gen_X.append(np.transpose(ord_gen_X, (1, 0, 2)))          # (t_train, ord, t_test)
        o_gen_neg_X.append(np.transpose(ord_gen_negX, (1, 0, 2)))
        o_cvnsr_Y.append(cvnsr_Y); o_neg_cvnsr_Y.append(neg_cvnsr_Y)
        o_gen_Y.append(np.transpose(ord_gen_Y, (1, 0, 2)))
        o_gen_neg_Y.append(np.transpose(ord_gen_negY, (1, 0, 2)))
        o_perseed_Y.append(perseed_Y)
        o_sig_diag_subj.append(sig_diag_subj); o_sig_gen_subj.append(sig_gen_subj)
        del ord_gen_nullc, ord_gen_nullnegc, ord_rdiag, ord_nulldiag
        gc.collect(); torch.cuda.empty_cache()

    # ---- group significance (scheme Y): mean over subject x ordering, then cluster ----
    sig_diag_group = _layer2_row_sig(g_diag_c / n_grp, g_diag_nc / n_grp, g_diag_n / n_grp, g_diag_nnc / n_grp, alpha)
    gen_c_m = g_gen_c / n_grp; gen_n_m = g_gen_n / n_grp
    gen_nc_m = g_gen_nc / n_grp; gen_nnc_m = g_gen_nnc / n_grp
    sig_gen_group = np.stack([
        _layer2_row_sig(gen_c_m[t], gen_nc_m[t], gen_n_m[t], gen_nnc_m[t], alpha)
        for t in range(n_time)
    ], axis=0)

    subjects = X_train.subject.values
    split_coord = np.arange(n_ord)
    seed_coord = np.arange(n_seeds)
    common = {generalization_dim: gen_coords, f"{generalization_dim}_train": gen_coords,
              f"{generalization_dim}_test": gen_coords, "subject": subjects, "split": split_coord}

    def _diag(name, arr):   # (subject, split, time)
        return xr.DataArray(np.asarray(arr), dims=["subject", "split", generalization_dim])

    def _gen(name, arr):    # (subject, time_train, split, time_test)
        return xr.DataArray(np.asarray(arr), dims=["subject", f"{generalization_dim}_train", "split", f"{generalization_dim}_test"])

    results = {
        "cvnsr_X": _diag("cvnsr_X", o_cvnsr_X), "neg_cvnsr_X": _diag("neg_cvnsr_X", o_neg_cvnsr_X),
        "cvnsr_Y": _diag("cvnsr_Y", o_cvnsr_Y), "neg_cvnsr_Y": _diag("neg_cvnsr_Y", o_neg_cvnsr_Y),
        "gen_cvnsr_X": _gen("gen_cvnsr_X", o_gen_X), "gen_neg_cvnsr_X": _gen("gen_neg_cvnsr_X", o_gen_neg_X),
        "gen_cvnsr_Y": _gen("gen_cvnsr_Y", o_gen_Y), "gen_neg_cvnsr_Y": _gen("gen_neg_cvnsr_Y", o_gen_neg_Y),
        "cvnsr_Y_perseed": xr.DataArray(np.asarray(o_perseed_Y), dims=["subject", "split", "seed", generalization_dim]),
        "sig_diag_group": xr.DataArray(sig_diag_group, dims=[generalization_dim]),
        "sig_diag_subject": xr.DataArray(np.asarray(o_sig_diag_subj), dims=["subject", generalization_dim]),
        "sig_gen_group": xr.DataArray(sig_gen_group, dims=[f"{generalization_dim}_train", f"{generalization_dim}_test"]),
        "sig_gen_subject": xr.DataArray(np.asarray(o_sig_gen_subj), dims=["subject", f"{generalization_dim}_train", f"{generalization_dim}_test"]),
    }
    return xr.Dataset(results).assign_coords({
        **{k: v for k, v in common.items() if k != "split"},
        "split": split_coord, "seed": seed_coord,
    })


def _acc(acc, x):
    return x.copy() if acc is None else acc + x


def tt_pca(
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
    presentation_subset: str = None,
    pca_source: str = "train",
    per_image: bool = False,
    n_seeds: int = 1,
    spearman_brown: bool = True,
    **kwargs,
):
    # pca_source: "train" (default) fits the PCA basis on the train set; "test" fits it on the reference
    # TEST split (within-test cvPCA). test mode requires 2 test splits.
    assert pca_source in ("train", "test"), pca_source
    if pca_source == "test":
        assert split_dim is not None, "pca_source='test' requires split_dim (needs 2 test splits)"

    # Stringer per-image half-rep mode: seed-averaged, in-lib significance, drop the raw null. The
    # seed-averaging of the observed r is only valid on a seed-INDEPENDENT (train-fit) basis, so
    # per_image requires pca_source='train'.
    if per_image:
        assert pca_source == "train", "per_image (Stringer) requires pca_source='train' (seed-independent basis)"
        assert split_dim is not None, "per_image requires split_dim (needs 2 test splits)"

    generalization_id = f"space={space}.pc_dims={pc_dims[0]}_{pc_dims[1]}.generalization_dim={generalization_dim}"
    if split_dim is not None:
        generalization_id += f".split_dim={split_dim}.n_splits={n_splits}"
    else:
        generalization_id += ".split_dim=None"

    # Stringer cache tags (default path adds NO tag -> published caches stay byte-identical).
    if per_image:
        generalization_id += f".split_mode=stringer.n_seeds={n_seeds}"
    if not spearman_brown:
        generalization_id += ".sb=0"

    # optional presentation (≈ session) subset of the TEST data, e.g. "0-5" / "6-11".
    # Tag the generalization_id BEFORE temp_cache_path so BOTH the internal cache_path and the
    # results.nc df_cache_path carry the tag (else a session run overwrites the full band cache).
    presentation_indices = None
    if presentation_subset is not None:
        lo, hi = presentation_subset.split("-")
        presentation_indices = list(range(int(lo), int(hi) + 1))
        generalization_id += f".sessions={presentation_subset}"

    # tag the basis source (de-indented: applies to the standard test run too, else it would clobber the
    # published train-basis cache). Default "train" adds NO tag -> train paths stay byte-identical.
    if pca_source != "train":
        generalization_id += f".pca_source={pca_source}"

    temp_cache_path = f"main_analyses/tt_pca/analysis={analysis}/{generalization_id}"
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
                per_image=per_image,
                n_seeds=n_seeds if per_image else 1,
                subjects=subjects,
                presentation_indices=presentation_indices,
                **load_dataset_kwargs
            )

            if analysis not in ["behavior", "bhvpc", "bhvsubset"]:
                feature_dim = target_var
            else:
                feature_dim = "behavior"
        case "bhv_eeg_rsd":
            # ttpca dimensionality of residual signals
            
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
            
            embd = load_embeddings()
            target_train = embd.sel(object=X_train.object.values)
            target_test = embd.sel(object=X_test.object.values)
            
            if analysis not in ["behavior", "bhvpc", "bhvsubset"]:
                feature_dim = target_var
            else:
                feature_dim = "behavior"
            predictor_cache_path = f"main_analyses/{analysis}_tt_encoding.subset=False.pca=False/dataset={dataset}"
            predictor_cache_path = _append_path(
                predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs
            )
            
            X_train_residual, X_test_residual = [], []
            for subject in X_train.subject.values:
                subject_train_residual, subject_test_residual = [], []
                for time in X_train.time.values:
                    subject_data_train = X_train.sel(subject=subject, time=time)
                    
                    scorer_fn = TrainTestModelScorer
                    scorer = scorer_fn(
                        **scorer_kwargs,
                        cache_predictors=True,
                        cache_subpath=f"{predictor_cache_path}/subject={subject}/target=time={time}/predictor=behavior=all",
                    )
                    
                    def _2t(x):
                        return torch.tensor(x.values).float().to(scorer.device)
                    
                    cacher = cache(
                        f"{scorer.cache_path}.pkl",
                        mode = "normal" if scorer.cache_predictors else "ignore",
                    )
                    
                    lmodel = cacher(scorer._fit_model)(
                        x_train=_2t(target_train), y_train=_2t(subject_data_train)
                    )
                    
                    subject_train_residual.append(
                        (_2t(subject_data_train) - lmodel.predict(_2t(target_train))).cpu()
                    )
                    
                    split_test_residual = [
                        (_2t(X_test.sel(seed=0, split=split, subject=subject, time=time)) - lmodel.predict(_2t(target_test))).cpu()
                        for split in X_test.split.values
                    ]
                    subject_test_residual.append(torch.stack(split_test_residual, dim=0))
                X_train_residual.append(torch.stack(subject_train_residual, dim=-1))
                X_test_residual.append(torch.stack(subject_test_residual, dim=-1))
            X_train_residual = torch.stack(X_train_residual, dim=0)
            X_test_residual = torch.stack([torch.stack(X_test_residual, dim=1)], dim=0)
            
            X_train = X_train.copy(data=X_train_residual.numpy())
            X_test = X_test.copy(data=X_test_residual.numpy())

    
    df_cache_path = _cache_path(temp_cache_path, dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    cacher = cache(
        f"{df_cache_path}/results.nc",
        # mode="ignore"
    )

    if per_image:
        return cacher(_tt_pca_stringer)(
            X_train, X_test, pc_dims, generalization_dim,
            n_seeds=n_seeds, spearman_brown=spearman_brown, alpha=alpha,
        )

    return cacher(_tt_pca)(X_train, X_test, cache_path, pc_dims, generalization_dim, split_dim=split_dim, pca_source=pca_source)