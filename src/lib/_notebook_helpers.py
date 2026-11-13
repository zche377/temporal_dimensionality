"""
Memory-efficient helper functions for figure notebooks.

These functions are designed to handle large permutation datasets (n=1000)
by processing one subject at a time and only keeping aggregated statistics.
"""

import gc
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from lib.computation.statistics import cluster_correction
from lib.datasets import load_n_subjects
from bonner.caching import BONNER_CACHING_HOME


# =============================================================================
# Cached cluster_correction
# =============================================================================
# `cluster_correction` is a pure function of (true, null, alpha) but its cluster-mass step is
# ~560 ms/call, which dominates every dimensionality figure (a single_subject EEG panel makes
# ~340 calls => ~3 min of pure cluster_correction, recomputed on every notebook re-run). This
# disk+RAM cache keys results on a content hash of the inputs, so it is correct across notebooks
# and auto-invalidates when the inputs change. Reuse it anywhere a notebook calls cluster_correction.

_CC_MEMO: dict[str, np.ndarray] = {}  # in-process: content-hash -> result


def _cc_key(true: np.ndarray, null: np.ndarray, alpha: float) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in (np.ascontiguousarray(true), np.ascontiguousarray(null)):
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    h.update(repr(alpha).encode())
    return h.hexdigest()


def cluster_correction_cached(true: np.ndarray, null: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Disk+RAM cached `cluster_correction` (pure fn of true, null, alpha).

    Drop-in replacement for `lib.computation.statistics.cluster_correction` in notebooks. Caches to
    `BONNER_CACHING_HOME/notebook_cache/cluster_correction/{hash}.npy`, keyed on a content hash of the
    inputs (auto-invalidates if they change). First call computes + saves; re-runs load instantly.
    """
    key = _cc_key(true, null, alpha)
    if key in _CC_MEMO:
        return _CC_MEMO[key]
    path = BONNER_CACHING_HOME / "notebook_cache" / "cluster_correction" / f"{key}.npy"
    if path.exists():
        res = np.load(path)
    else:
        res = np.asarray(cluster_correction(true, null, alpha=alpha))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, res)
    _CC_MEMO[key] = res
    return res


# =============================================================================
# Configuration helpers
# =============================================================================

def get_scorer_kwargs(model_name: str):
    """Get scorer_kwargs and target_var for a given model."""
    mn_str = model_name if model_name != "en_linear" else "linear"
    scorer_kwargs = f"model_name={mn_str}"
    match model_name:
        case "linear":
            scorer_kwargs += ".l2_penalty=0.01"
            target_var = "neuroid"
        case "plssvd":
            target_var = "latent"
        case "en_linear":
            scorer_kwargs += ".l2_penalty=0.01"
            target_var = "neuroid"
        case _:
            target_var = "neuroid"
    return scorer_kwargs, target_var


def spearman_brown_correction(x, n=2):
    """Apply Spearman-Brown prophecy formula."""
    return n * x / (1 + (n - 1) * x)


# =============================================================================
# Residual feature-set alignment (fig-5 "rmv" follow-up)
# =============================================================================

def _drop_constant_columns(arr: np.ndarray, labels=None, tol: float = 1e-12):
    """Drop zero-variance columns (constant over stimuli) -> NaN-free Pearson r.

    Returns (filtered_array, kept_labels). ``labels`` is an optional 1-D array of
    column identifiers carried through the same mask.
    """
    keep = arr.std(axis=0) > tol
    out = arr[:, keep]
    if labels is None:
        return out, None
    return out, np.asarray(labels)[keep]


def load_residual_feature_sets(
    dataset: str,
    model_uid: str,
    test_objects: np.ndarray,
    test_img_files: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load the four feature sets aligned to a residual's stimulus order.

    The residual (from ``remaining_var``) has a ``presentation`` dim whose index is the
    object/concept name and which also carries an ``img_files`` coord. Behavior and LLM
    embeddings are object-keyed; the early/late model layers are image-keyed. Both are
    reindexed to the *residual's* canonical order (``test_objects`` / ``test_img_files``) so
    row i of every returned matrix is the same stimulus as row i of the residual.

    No z-scoring is applied: Pearson correlation is shift/scale invariant. Constant columns
    are dropped to avoid NaNs.

    Returns ``{"behavior", "early", "late", "llm"}`` -> ``(n_stimuli, d)`` float arrays.
    """
    from bonner.datasets.hebart2022_things_behavior import load_embeddings
    from lib.datasets.hebart2022_things_behavior import load_llm_embeddings
    from lib.models import DJModel

    test_objects = np.asarray(test_objects)
    test_img_files = np.asarray(test_img_files)

    out: dict[str, np.ndarray] = {}

    # --- object-keyed embeddings: positional reindex to the residual's object order ---
    for key, loader in (("behavior", load_embeddings), ("llm", load_llm_embeddings)):
        embd = loader().sel(object=test_objects)
        assert np.array_equal(embd.object.values, test_objects), f"{key} object order mismatch"
        out[key], _ = _drop_constant_columns(np.asarray(embd.values, dtype=float))

    # --- image-keyed model layers (first Identity node = early, last = late) ---
    model = DJModel(model_uid, hook="srp", n_components="auto", nodes="Identity", node_subset=True)
    feature_maps = model(dataset, dataloader_kwargs={"batch_size": 32})
    node_names = list(feature_maps.keys())
    test_set = set(test_img_files.tolist())
    pos = {f: i for i, f in enumerate(test_img_files.tolist())}
    for key, node in (("early", node_names[0]), ("late", node_names[-1])):
        fm = feature_maps[node]
        fm = fm.isel(presentation=np.array([f in test_set for f in fm.img_files.values]))
        order = np.argsort([pos[f] for f in fm.img_files.values])
        fm = fm.isel(presentation=order)
        assert np.array_equal(fm.img_files.values, test_img_files), f"{key} img_files order mismatch"
        out[key], _ = _drop_constant_columns(np.asarray(fm.values, dtype=float))

    return out


# =============================================================================
# Low-level data loading (single subject)
# =============================================================================

def _load_single_subject(subpath: Path, subject: int) -> xr.Dataset:
    """Load a single subject's data file."""
    return xr.open_dataset(subpath / f"subject={subject}.nc")


def _get_null_key(n_permutations: int) -> str:
    """Get the null variable name for a given number of permutations."""
    return f"null.pearsonr.n_permutations={n_permutations}"


# =============================================================================
# Memory-efficient data loading with streaming statistics
# =============================================================================

def load_decoding_data_streaming(
    subpath: str | Path,
    dataset: str,
    n_permutations: int,
    target_var: str,
    time_min: float = -0.05,
    folds: bool = False,
    has_node: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Load decoding data and compute significance by processing one subject at a time.

    This function first checks for precomputed stats.nc. If available, uses that
    for significance. Otherwise falls back to streaming computation.

    Returns:
        xdf: DataFrame with pearsonr scores for plotting
        sig_times: dict mapping node -> array of significant time points
    """
    subpath = Path(subpath)
    n_subjects = load_n_subjects(dataset)
    null_key = _get_null_key(n_permutations)

    # Check for precomputed stats
    stats_path = subpath / "stats.nc"
    use_precomputed_stats = stats_path.exists()

    # First pass: collect scores for plotting
    all_scores = []
    all_pearsonr_for_sig = []  # Only needed if no precomputed stats
    all_null_for_sig = []      # Only needed if no precomputed stats

    for s in range(1, n_subjects + 1):
        x = _load_single_subject(subpath, s)
        if folds:
            x = x.mean("fold")

        # Check if null exists in file
        has_null = null_key in x
        if has_null:
            x = x.rename({null_key: "null"})

        x = x.sel(time=x.time >= time_min)

        # Store scores for plotting (without null)
        score_df = x.pearsonr.to_dataframe("score").dropna().reset_index()
        score_df["subject"] = s
        all_scores.append(score_df)

        # Only compute streaming stats if no precomputed stats
        if not use_precomputed_stats and has_null:
            if has_node:
                pearsonr_agg = x.pearsonr.mean(target_var)  # (node, time)
                null_agg = x.null.mean(target_var).transpose("node", "permutation", "time")
            else:
                pearsonr_agg = x.pearsonr.mean(target_var)  # (time,)
                null_agg = x.null.mean(target_var).transpose("permutation", "time")

            all_pearsonr_for_sig.append(pearsonr_agg.values)
            all_null_for_sig.append(null_agg.values)

        # Get metadata from first subject
        if s == 1:
            time_values = x.time.values
            if has_node:
                node_values = x.node.values

        del x
        gc.collect()

    # Combine scores DataFrame
    xdf = pd.concat(all_scores, ignore_index=True)

    # Get significance from precomputed stats or compute it
    if use_precomputed_stats:
        stats = xr.open_dataset(stats_path)
        stats = stats.sel(time=stats.time >= time_min)

        sig_times = {}
        if has_node:
            for node in stats.node.values:
                p_vals = stats.cluster_p.sel(node=node).values
                sig_times[node] = stats.time.values[p_vals < 0.05]
        else:
            p_vals = stats.cluster_p.values
            sig_times["all"] = stats.time.values[p_vals < 0.05]

        del stats
    else:
        # Fallback: compute from streaming data
        pearsonr_stack = np.stack(all_pearsonr_for_sig, axis=-1).mean(axis=-1)
        null_stack = np.stack(all_null_for_sig, axis=-1).mean(axis=-1)

        sig_times = {}
        if has_node:
            for i, node in enumerate(node_values):
                p_vals = cluster_correction_cached(pearsonr_stack[i], null_stack[i])
                sig_times[node] = time_values[p_vals < 0.05]
        else:
            p_vals = cluster_correction_cached(pearsonr_stack, null_stack)
            sig_times["all"] = time_values[p_vals < 0.05]

        del all_pearsonr_for_sig, all_null_for_sig, pearsonr_stack, null_stack

    gc.collect()
    return xdf, sig_times


def load_bhv_decoding_data_streaming(
    subpath: str | Path,
    dataset: str,
    n_permutations: int,
    target_var: str,
    time_min: float = -0.05,
    folds: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load behavioral decoding data with streaming to save memory.

    This function first checks for precomputed stats.nc. If available, uses that
    for significance. Otherwise falls back to streaming computation.

    Returns:
        xdf: DataFrame with scores for plotting
        sxdf: DataFrame with significant time points
    """
    subpath = Path(subpath)
    n_subjects = load_n_subjects(dataset)
    null_key = _get_null_key(n_permutations)

    # Check for precomputed stats
    stats_path = subpath / "stats.nc"
    use_precomputed_stats = stats_path.exists()

    all_scores = []
    all_pearsonr = []  # Only needed if no precomputed stats
    all_null = []      # Only needed if no precomputed stats

    for s in range(1, n_subjects + 1):
        x = _load_single_subject(subpath, s)
        if folds:
            x = x.mean("fold")

        # Check if null exists in file
        has_null = null_key in x
        if has_null:
            x = x.rename({null_key: "null"})

        x = x.sel(time=x.time >= time_min)

        # For plotting
        if has_null:
            x_plot = x.copy()
            x_plot["null"] = x_plot.null.mean("permutation")
            score_df = x_plot.to_dataframe().reset_index().melt(
                id_vars=["time", target_var],
                var_name="metric",
                value_name="score"
            )
            del x_plot
        else:
            score_df = x.pearsonr.to_dataframe("score").dropna().reset_index()
            score_df["metric"] = "pearsonr"

        score_df["subject"] = s
        all_scores.append(score_df)

        # Only compute streaming stats if no precomputed stats
        if not use_precomputed_stats and has_null:
            all_pearsonr.append(x.pearsonr.mean(target_var).values)
            null_transposed = x.null.mean(target_var).transpose("permutation", "time")
            all_null.append(null_transposed.values)

        if s == 1:
            time_values = x.time.values

        del x
        gc.collect()

    xdf = pd.concat(all_scores, ignore_index=True)

    # Get significance from precomputed stats or compute it
    if use_precomputed_stats:
        stats = xr.open_dataset(stats_path)
        stats = stats.sel(time=stats.time >= time_min)

        p_vals = stats.cluster_p.values
        pearsonr_mean = stats.pearsonr_mean.values
        time_values = stats.time.values

        sig_mask = p_vals < 0.05
        sxdf = pd.DataFrame({
            "time": time_values[sig_mask],
            "pearsonr": pearsonr_mean[sig_mask]
        })

        del stats
    else:
        # Fallback: compute from streaming data
        pearsonr_mean = np.stack(all_pearsonr, axis=-1).mean(axis=-1)
        null_mean = np.stack(all_null, axis=-1).mean(axis=-1)
        p_vals = cluster_correction_cached(pearsonr_mean, null_mean)

        sig_mask = p_vals < 0.05
        sxdf = pd.DataFrame({
            "time": time_values[sig_mask],
            "pearsonr": pearsonr_mean[sig_mask]
        })

        del all_pearsonr, all_null

    gc.collect()
    return xdf, sxdf


def load_cvpca_dimensionality_streaming(
    cvpca_path: str | Path,
    dataset_label: str,
    time_min: float = -0.05,
) -> pd.DataFrame:
    """
    Load CV-PCA results and compute dimensionality with memory efficiency.

    Processes one subject/split/neuroid at a time for cluster correction.

    Returns:
        cdf: DataFrame with dimensionality (cvnsr) per subject and time
    """
    cvpca = xr.open_dataset(cvpca_path)
    cvpca = cvpca.sel(time_train=cvpca.time_train >= time_min)
    cvpca = cvpca.sel(time_test=cvpca.time_test >= time_min)
    cvpca = cvpca.sel(time=cvpca.time >= time_min)

    cdf = []
    for subject in cvpca.subject.values:
        for split in cvpca.split.values:
            epval = []
            for neuroid in cvpca.neuroid.values:
                scvpca = cvpca.sel(split=split, subject=subject, neuroid=neuroid)
                epval.append(cluster_correction_cached(scvpca.r.values, scvpca.null_r.values))

            cdf.append(pd.DataFrame({
                "subject": subject,
                "time": cvpca.time.values,
                "cvnsr": np.sum(np.stack(epval, axis=0) < 0.05, axis=0),
                "metric": f"{dataset_label} dimensionality",
            }))

    del cvpca
    gc.collect()

    return pd.concat(cdf, ignore_index=True)


def load_model_decoding_dimensionality_streaming(
    subpath: str | Path,
    dataset: str,
    n_permutations: int,
    target_var: str,
    time_min: float = -0.05,
    folds: bool = False,
) -> pd.DataFrame:
    """
    Load model decoding results and compute dimensionality (number of significant PCs).

    Checks in order:
    1. Precomputed n_decodable_dims in per-subject files (for error bars)
    2. Compute from per-subject null distributions (fallback)

    Returns:
        ndf: DataFrame with n_pc (significant dimensions) per subject, time, and node
    """
    subpath = Path(subpath)
    n_subjects = load_n_subjects(dataset)
    null_key = _get_null_key(n_permutations)

    # Try loading n_decodable_dims from per-subject files
    ndf = []
    all_have_n_decodable_dims = True
    has_node = None

    for s in range(1, n_subjects + 1):
        x = _load_single_subject(subpath, s)
        if folds:
            x = x.mean("fold")

        if has_node is None:
            has_node = "node" in x.dims

        if "n_decodable_dims" in x:
            x = x.sel(time=x.time >= time_min)
            if has_node:
                for node in x.node.values:
                    ndf.append(pd.DataFrame({
                        "subject": s,
                        "time": x.time.values,
                        "n_pc": x.n_decodable_dims.sel(node=node).values,
                        "node": node,
                    }))
            else:
                ndf.append(pd.DataFrame({
                    "subject": s,
                    "time": x.time.values,
                    "n_pc": x.n_decodable_dims.values,
                }))
        else:
            all_have_n_decodable_dims = False

        del x
        gc.collect()

    if all_have_n_decodable_dims and ndf:
        return pd.concat(ndf, ignore_index=True)

    # Fallback: compute from per-subject null distributions
    ndf = []

    for s in range(1, n_subjects + 1):
        x = _load_single_subject(subpath, s)
        if folds:
            x = x.mean("fold")

        # Check if null exists in file
        has_null = null_key in x
        if not has_null:
            raise ValueError(f"No null distribution found in subject={s} and no precomputed n_decodable_dims. "
                           f"Rerun tt_decoding to regenerate with n_decodable_dims.")

        x = x.rename({null_key: "null"})
        x = x.sel(time=x.time >= time_min)

        # Check if there's a node dimension
        has_node = "node" in x.dims

        if has_node:
            for node in x.node.values:
                sspval = []
                for latent in x[target_var].values:
                    sx = x.sel({target_var: latent, "node": node})
                    # null shape after sel: (time, permutation) -> need (permutation, time)
                    null_transposed = sx.null.transpose("permutation", "time").values
                    sspval.append(cluster_correction_cached(sx.pearsonr.values, null_transposed))

                ndf.append(pd.DataFrame({
                    "subject": s,
                    "time": x.time.values,
                    "n_pc": np.sum(np.stack(sspval, axis=0) < 0.05, axis=0),
                    "node": node,
                }))
        else:
            sspval = []
            for latent in x[target_var].values:
                sx = x.sel({target_var: latent})
                # null shape after sel: (time, permutation) -> need (permutation, time)
                null_transposed = sx.null.transpose("permutation", "time").values
                sspval.append(cluster_correction_cached(sx.pearsonr.values, null_transposed))

            ndf.append(pd.DataFrame({
                "subject": s,
                "time": x.time.values,
                "n_pc": np.sum(np.stack(sspval, axis=0) < 0.05, axis=0),
            }))

        del x
        gc.collect()

    return pd.concat(ndf, ignore_index=True)


# =============================================================================
# Generalization matrix helpers
# =============================================================================

def load_generalization_data_streaming(
    subpath: str | Path,
    dataset: str,
    n_permutations: int,
    target_var: str,
    time_min: float = -0.05,
    folds: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load temporal generalization data with streaming.

    This function first checks for precomputed stats.nc. If available, uses that
    for significance. Otherwise falls back to streaming computation.

    Returns:
        xdf: DataFrame (pivoted) with pearsonr as values, time_train as index, time_test as columns
        sig_df: DataFrame (pivoted) with significance (bool) in same format
    """
    subpath = Path(subpath)
    n_subjects = load_n_subjects(dataset)
    null_key = _get_null_key(n_permutations)

    # Check for precomputed stats
    stats_path = subpath / "stats.nc"
    use_precomputed_stats = stats_path.exists()

    all_pearsonr = []
    all_null = []  # Only needed if no precomputed stats

    for s in range(1, n_subjects + 1):
        x = _load_single_subject(subpath, s)
        if folds:
            x = x.mean("fold")

        # Check if null exists in file
        has_null = null_key in x
        if has_null:
            x = x.rename({null_key: "null"})

        x = x.sel(time=x.time >= time_min)
        x = x.sel(time_generalization=x.time_generalization >= time_min)

        all_pearsonr.append(x.pearsonr.mean(target_var).values)

        # Only compute streaming stats if no precomputed stats
        if not use_precomputed_stats and has_null:
            null_transposed = x.null.mean(target_var).transpose("permutation", "time", "time_generalization")
            all_null.append(null_transposed.values)

        if s == 1:
            time_values = x.time.values
            time_gen_values = x.time_generalization.values

        del x
        gc.collect()

    # Average over subjects for pearsonr
    pearsonr_mean = np.stack(all_pearsonr, axis=-1).mean(axis=-1)

    # Get significance from precomputed stats or compute it
    if use_precomputed_stats:
        stats = xr.open_dataset(stats_path)
        stats = stats.sel(time=stats.time >= time_min)
        stats = stats.sel(time_generalization=stats.time_generalization >= time_min)

        p = stats.cluster_p.values
        neg_p = stats.cluster_p_neg.values if "cluster_p_neg" in stats else np.ones_like(p)

        del stats
    else:
        # Fallback: compute from streaming data
        null_mean = np.stack(all_null, axis=-1).mean(axis=-1)
        p = _cc_per_row(pearsonr_mean, null_mean)
        neg_p = _cc_per_row(-pearsonr_mean, -null_mean)

        del all_null

    # Create DataFrames
    xdf_flat = pd.DataFrame({
        "time_train": np.repeat(time_values, len(time_gen_values)),
        "time_test": np.tile(time_gen_values, len(time_values)),
        "pearsonr": pearsonr_mean.flatten(),
    })

    p_flat = p.T.flatten()
    neg_p_flat = neg_p.T.flatten()
    xdf_flat["sig"] = [
        p_flat[i] < 0.05 if c > 0 else neg_p_flat[i] < 0.05
        for i, c in enumerate(xdf_flat.pearsonr.values)
    ]

    xdf = xdf_flat.pivot(index="time_train", columns="time_test", values="pearsonr")
    sig_df = xdf_flat.pivot(index="time_train", columns="time_test", values="sig")

    del all_pearsonr
    gc.collect()

    return xdf, sig_df


def _cc_per_row(x: np.ndarray, null_x: np.ndarray) -> np.ndarray:
    """Apply cluster correction per row of a 2D array."""
    p = []
    for i in range(x.shape[0]):
        p.append(cluster_correction_cached(x[i], null_x[:, i] if null_x.ndim > 1 else null_x))
    return np.stack(p, axis=0)


def load_cvpca_generalization_streaming(
    cvpca_path: str | Path,
    time_min: float = -0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CV-PCA generalization results and compute significance.

    This handles the gen_cvnsr, gen_neg_cvnsr, gen_null_cvnsr variables from CV-PCA results.

    Returns:
        score_df: DataFrame (pivoted) with cvnsr values, time_train as index, time_test as columns
        sig_df: DataFrame (pivoted) with significance (bool) in same format
    """
    import torch

    cvpca = xr.open_dataset(cvpca_path)
    cvpca = cvpca.sel(time_train=cvpca.time_train >= time_min)
    cvpca = cvpca.sel(time_test=cvpca.time_test >= time_min)
    cvpca = cvpca.sel(time=cvpca.time >= time_min)

    time_values = cvpca.time.values

    # Mean over subjects (dim 0) and splits (dim 2)
    mean_cvnsrs = torch.tensor(cvpca.gen_cvnsr.values).mean(dim=[0, 2])
    mean_null_cvnsrs = torch.tensor(cvpca.gen_null_cvnsr.values).mean(dim=[0, 2]).permute(1, 0, 2)

    mean_neg_cvnsrs = torch.tensor(cvpca.gen_neg_cvnsr.values).mean(dim=[0, 2])
    mean_null_neg_cvnsrs = torch.tensor(cvpca.gen_null_cvnsr.values).mean(dim=[0, 2]).permute(1, 0, 2)

    # Cluster correction per row
    p = _cc_per_row_torch(mean_cvnsrs, mean_null_cvnsrs)
    neg_p = _cc_per_row_torch(mean_neg_cvnsrs, mean_null_neg_cvnsrs)

    # Create DataFrame
    df = pd.DataFrame({
        "time_train": np.repeat(time_values, len(time_values)),
        "time_test": np.tile(time_values, len(time_values)),
        "cvnsr": mean_cvnsrs.cpu().numpy().flatten(),
        "neg_cvnsr": mean_neg_cvnsrs.cpu().numpy().flatten(),
    })

    # Combine cvnsr and neg_cvnsr: take max absolute value with sign
    temp = []
    for i in range(len(df)):
        v = np.nanmax([df.loc[i, "cvnsr"], df.loc[i, "neg_cvnsr"]])
        if ~np.isnan(v):
            v *= (0.5 - np.nanargmax([df.loc[i, "cvnsr"], df.loc[i, "neg_cvnsr"]])) * 2
        temp.append(v)
    df["cvnsr"] = temp

    # Compute significance
    p_flat = p.T.flatten()
    neg_p_flat = neg_p.T.flatten()
    df["sig"] = [
        p_flat[i] < 0.05 if c > 0 else neg_p_flat[i] < 0.05
        for i, c in enumerate(df.cvnsr.values)
    ]

    score_df = df.pivot(index="time_train", columns="time_test", values="cvnsr")
    sig_df = df.pivot(index="time_train", columns="time_test", values="sig")

    del cvpca
    gc.collect()

    return score_df, sig_df


def _cc_per_row_torch(x, null_x):
    """Apply cluster correction per row using torch tensors."""
    p = []
    for i in range(x.shape[0]):
        p.append(cluster_correction_cached(x[i].cpu().numpy(), null_x[:, i].cpu().numpy()))
    return np.stack(p, axis=0)


def load_cvpca_generalization_per_subject(
    cvpca_path: str | Path,
    time_min: float = -0.05,
) -> pd.DataFrame:
    """
    Load CV-PCA generalization results per subject (no subject-averaging, no significance).

    Mirrors the signed-cvnsr combine of ``load_cvpca_generalization_streaming`` (max abs of the
    positive/negative dimensionality with sign), but keeps the ``subject`` axis and only averages
    over splits. Intended for per-subject generalization heatmaps.

    Returns:
        df: long DataFrame with columns [subject, time_train, time_test, cvnsr]
    """
    cvpca = xr.open_dataset(cvpca_path)
    cvpca = cvpca.sel(time_train=cvpca.time_train >= time_min)
    cvpca = cvpca.sel(time_test=cvpca.time_test >= time_min)

    time_train_values = cvpca.time_train.values
    time_test_values = cvpca.time_test.values

    # Mean over splits, keep subject: (subject, time_train, time_test)
    pos = cvpca.gen_cvnsr.mean("split").transpose("subject", "time_train", "time_test").values
    neg = cvpca.gen_neg_cvnsr.mean("split").transpose("subject", "time_train", "time_test").values

    # Per cell: max abs with sign (+1 pos / -1 neg), nan-safe, matching the streaming loader.
    stacked = np.stack([pos, neg], axis=0)  # (2, subject, tt, te)
    all_nan = np.all(np.isnan(stacked), axis=0)
    filled = np.where(np.isnan(stacked), -np.inf, stacked)  # avoid all-NaN nanargmax errors
    magnitude = np.max(filled, axis=0)
    sign = (0.5 - np.argmax(filled, axis=0)) * 2
    signed = magnitude * sign
    signed[all_nan] = np.nan

    rows = []
    for si, subject in enumerate(cvpca.subject.values):
        rows.append(pd.DataFrame({
            "subject": subject,
            "time_train": np.repeat(time_train_values, len(time_test_values)),
            "time_test": np.tile(time_test_values, len(time_train_values)),
            "cvnsr": signed[si].flatten(),
        }))

    del cvpca
    gc.collect()

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Stringer per-image half-rep cvPCA loaders (seed-averaged, in-lib significance).
# All read the NEW results.nc schema (cvnsr_{X,Y}, gen_cvnsr_{X,Y}, sig_*); guarded by the
# "sig_diag_group" discriminator so published seedless caches keep their existing loaders.
# =============================================================================

def _assert_stringer(ds):
    assert "sig_diag_group" in ds, (
        "not a Stringer per-image cvPCA cache (no sig_diag_group); use the default loaders instead"
    )


def load_cvpca_stringer_dimensionality(
    cvpca_path: str | Path,
    dataset_label: str,
    scheme: str = "Y",
    time_min: float = -0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stringer diagonal dimensionality over time (seed-mean; significance precomputed in-lib).

    scheme "Y" (compute_p@L1 + cluster@L2; fig_2/4 value == fig_3 diagonal) or "X" (cluster@L1, the
    published fig_2/4 recipe; no significance line). Returns:
      df:  long [subject, time, cvnsr]  -- per-subject, seed-mean, averaged over the split orderings
      sig: long [time, sig_group, sig_frac_subject]  -- group sig line (Y only; all False for X)
    """
    ds = xr.open_dataset(cvpca_path)
    _assert_stringer(ds)
    ds = ds.sel(time=ds.time >= time_min)
    cv = ds[f"cvnsr_{scheme}"].mean("split").transpose("subject", "time")   # (subject, time)
    df = []
    for si, subject in enumerate(ds.subject.values):
        df.append(pd.DataFrame({
            "subject": subject,
            "time": ds.time.values,
            "cvnsr": cv.isel(subject=si).values,
            "metric": f"{dataset_label} dimensionality",
        }))
    df = pd.concat(df, ignore_index=True)

    if scheme == "Y":
        sig_group = ds["sig_diag_group"].values.astype(bool)
        sig_subj = ds["sig_diag_subject"].transpose("subject", "time").values.astype(bool)
        sig_frac = sig_subj.mean(axis=0)
    else:
        sig_group = np.zeros(len(ds.time), dtype=bool)
        sig_frac = np.zeros(len(ds.time))
    sig = pd.DataFrame({"time": ds.time.values, "sig_group": sig_group, "sig_frac_subject": sig_frac})
    ds.close()
    return df, sig


def load_cvpca_stringer_perseed(
    cvpca_path: str | Path,
    dataset_label: str,
    time_min: float = -0.05,
) -> pd.DataFrame:
    """Per-seed diagonal dimensionality (scheme-Y per-cell count, no clustering) for an across-SEED
    stability band. Long [seed, subject, time, cvnsr], seed-mean over the split orderings."""
    ds = xr.open_dataset(cvpca_path)
    _assert_stringer(ds)
    ds = ds.sel(time=ds.time >= time_min)
    cv = ds["cvnsr_Y_perseed"].mean("split").transpose("seed", "subject", "time")
    rows = []
    for sd in range(cv.sizes["seed"]):
        for si, subject in enumerate(ds.subject.values):
            rows.append(pd.DataFrame({
                "seed": sd, "subject": subject, "time": ds.time.values,
                "cvnsr": cv.isel(seed=sd, subject=si).values,
                "metric": f"{dataset_label} dimensionality",
            }))
    ds.close()
    return pd.concat(rows, ignore_index=True)


def load_cvpca_stringer_generalization(
    cvpca_path: str | Path,
    scheme: str = "Y",
    per_subject: bool = False,
    time_min: float = -0.05,
):
    """Stringer generalization dimensionality matrix (seed-mean; significance precomputed in-lib).

    Group (per_subject=False): returns (score_df, sig_df) pivoted [time_train x time_test], the
    subject+split-averaged signed dimensionality and the group scheme-Y sig mask (all False for X).
    per_subject=True: returns a long DataFrame [subject, time_train, time_test, cvnsr, sig].
    """
    ds = xr.open_dataset(cvpca_path)
    _assert_stringer(ds)
    ds = ds.sel(time_train=ds.time_train >= time_min, time_test=ds.time_test >= time_min)
    tt, te = ds.time_train.values, ds.time_test.values

    pos = ds[f"gen_cvnsr_{scheme}"]; neg = ds[f"gen_neg_cvnsr_{scheme}"]
    have_sig = scheme == "Y"

    if per_subject:
        pos = pos.mean("split").transpose("subject", "time_train", "time_test").values
        neg = neg.mean("split").transpose("subject", "time_train", "time_test").values
        signed = _signed_combine_np(pos, neg)
        sig = (ds["sig_gen_subject"].transpose("subject", "time_train", "time_test").values.astype(bool)
               if have_sig else np.zeros_like(signed, dtype=bool))
        rows = []
        for si, subject in enumerate(ds.subject.values):
            rows.append(pd.DataFrame({
                "subject": subject,
                "time_train": np.repeat(tt, len(te)),
                "time_test": np.tile(te, len(tt)),
                "cvnsr": signed[si].flatten(),
                "sig": sig[si].flatten(),
            }))
        ds.close()
        return pd.concat(rows, ignore_index=True)

    pos = pos.mean(["subject", "split"]).transpose("time_train", "time_test").values
    neg = neg.mean(["subject", "split"]).transpose("time_train", "time_test").values
    signed = _signed_combine_np(pos, neg)
    sig = ds["sig_gen_group"].transpose("time_train", "time_test").values.astype(bool) if have_sig else np.zeros_like(signed, dtype=bool)
    df = pd.DataFrame({
        "time_train": np.repeat(tt, len(te)), "time_test": np.tile(te, len(tt)),
        "cvnsr": signed.flatten(), "sig": sig.flatten(),
    })
    ds.close()
    return df.pivot(index="time_train", columns="time_test", values="cvnsr"), df.pivot(index="time_train", columns="time_test", values="sig")


def across_subject_sig(vals, alpha: float = 0.05, fdr: bool = True):
    """Group significance for a Scheme-X dimensionality (per-cell, ACROSS SUBJECTS, one-sided > 0).

    Scheme X's count is already the per-PC cluster-corrected dimensionality (Layer-1, per subject), so
    the only inference left is at the group level and it must NOT re-cluster over the axis (that would
    double-correct + reintroduce the pre-stim floor / temporal-contiguity artifacts). This does a simple
    per-cell one-sided test that the per-subject count is reliably > 0.

    `vals`: (n_subject, *cells). Returns (p, sig) each of shape `cells`. Uses Wilcoxon signed-rank when
    n>=6; below that Wilcoxon CANNOT reach alpha (min two-sided p at n=5 is 0.0625), so it falls back to
    a one-sample t-test (relevant for MEG, n=4). All-<=0 / constant cells -> p=1 (pre-stim ~0 => n.s.).
    """
    from scipy import stats
    vals = np.asarray(vals, dtype=float)
    n = vals.shape[0]
    flat = vals.reshape(n, -1)
    p = np.ones(flat.shape[1])
    use_wilcoxon = n >= 6
    for c in range(flat.shape[1]):
        x = flat[:, c]
        x = x[~np.isnan(x)]
        if len(x) < 3 or np.nanmax(x) <= 0 or np.ptp(x) == 0:
            continue  # p stays 1
        try:
            if use_wilcoxon:
                p[c] = stats.wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue
            else:
                p[c] = stats.ttest_1samp(x, 0.0, alternative="greater").pvalue
        except ValueError:
            p[c] = 1.0
    p = np.where(np.isfinite(p), p, 1.0)
    if fdr:
        from scipy.stats import false_discovery_control
        q = false_discovery_control(p)
        sig = q < alpha
    else:
        sig = p < alpha
    return p.reshape(vals.shape[1:]), sig.reshape(vals.shape[1:])


def _signed_combine_np(pos, neg):
    """max-abs-with-sign combine (nan-safe), matching tt_pca._signed_combine / the streaming loaders."""
    stacked = np.stack([pos, neg], axis=0)
    all_nan = np.all(np.isnan(stacked), axis=0)
    filled = np.where(np.isnan(stacked), -np.inf, stacked)
    signed = np.max(filled, axis=0) * ((0.5 - np.argmax(filled, axis=0)) * 2)
    signed[all_nan] = np.nan
    return signed


def load_bhv_dimensionality_streaming(
    subpath: str | Path,
    dataset: str,
    n_permutations: int,
    target_var: str,
    time_min: float = -0.05,
    folds: bool = False,
) -> pd.DataFrame:
    """
    Load behavioral decoding results and compute dimensionality (number of significant latents).

    Checks in order:
    1. Precomputed n_decodable_dims in per-subject files (for error bars)
    2. Compute from per-subject null distributions (fallback)

    This is for behavior/everything decoding where there's no node dimension.

    Returns:
        ndf: DataFrame with n_pc (significant dimensions) per subject and time
    """
    subpath = Path(subpath)
    n_subjects = load_n_subjects(dataset)
    null_key = _get_null_key(n_permutations)

    # Try loading n_decodable_dims from per-subject files
    ndf = []
    all_have_n_decodable_dims = True

    for s in range(1, n_subjects + 1):
        x = _load_single_subject(subpath, s)
        if folds:
            x = x.mean("fold")

        if "n_decodable_dims" in x:
            x = x.sel(time=x.time >= time_min)
            ndf.append(pd.DataFrame({
                "subject": s,
                "time": x.time.values,
                "n_pc": x.n_decodable_dims.values,
            }))
        else:
            all_have_n_decodable_dims = False

        del x
        gc.collect()

    if all_have_n_decodable_dims and ndf:
        return pd.concat(ndf, ignore_index=True)

    # Fallback: compute from per-subject null distributions
    ndf = []

    for s in range(1, n_subjects + 1):
        x = _load_single_subject(subpath, s)
        if folds:
            x = x.mean("fold")

        # Check if null exists in file
        has_null = null_key in x
        if not has_null:
            raise ValueError(f"No null distribution found in subject={s} and no precomputed n_decodable_dims. "
                           f"Rerun tt_decoding to regenerate with n_decodable_dims.")

        x = x.rename({null_key: "null"})
        x = x.sel(time=x.time >= time_min)

        sspval = []
        for latent in x[target_var].values:
            sx = x.sel({target_var: latent})
            # null shape: (time, permutation) -> need (permutation, time)
            null_transposed = sx.null.transpose("permutation", "time").values
            sspval.append(cluster_correction_cached(sx.pearsonr.values, null_transposed))

        ndf.append(pd.DataFrame({
            "subject": s,
            "time": x.time.values,
            "n_pc": np.sum(np.stack(sspval, axis=0) < 0.05, axis=0),
        }))

        del x
        gc.collect()

    return pd.concat(ndf, ignore_index=True)


def load_model_generalization_streaming(
    subpath: str | Path,
    dataset: str,
    target_var: str,
    node: int,
    time_min: float = -0.05,
    folds: bool = False,
) -> pd.DataFrame:
    """
    Load model decoding temporal generalization data for a specific node.

    This is for plotting model layer generalization matrices (no significance computation).

    Returns:
        xdf: DataFrame (pivoted) with pearsonr as values, time_test as index, time_train as columns
    """
    subpath = Path(subpath)
    n_subjects = load_n_subjects(dataset)

    all_pearsonr = []

    for s in range(1, n_subjects + 1):
        x = xr.open_dataset(subpath / f"subject={s}.nc")
        if folds:
            x = x.mean("fold")
        # Select specific node
        x = x.pearsonr.isel(node=node)
        x = x.sel(time=x.time >= time_min)
        x = x.sel(time_generalization=x.time_generalization >= time_min)

        all_pearsonr.append(x.mean(target_var).values)

        if s == 1:
            time_values = x.time.values
            time_gen_values = x.time_generalization.values

        del x
        gc.collect()

    # Average over subjects
    pearsonr_mean = np.stack(all_pearsonr, axis=-1).mean(axis=-1)

    # Create DataFrame
    xdf_flat = pd.DataFrame({
        "time_train": np.repeat(time_values, len(time_gen_values)),
        "time_test": np.tile(time_gen_values, len(time_values)),
        "pearsonr": pearsonr_mean.flatten(),
    })

    xdf = xdf_flat.pivot(index="time_test", columns="time_train", values="pearsonr")

    del all_pearsonr
    gc.collect()

    return xdf


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_with_sem(
    data: pd.DataFrame,
    x: str,
    y: str,
    ax,
    color: str,
    label: str = None,
    group_cols: list = None,
):
    """
    Plot line with SEM shading.

    Args:
        data: DataFrame with the data
        x: Column name for x-axis
        y: Column name for y-axis
        ax: Matplotlib axis
        color: Color for line and fill
        label: Legend label
        group_cols: Columns to group by before computing mean/SEM (default: [x, "subject"])
    """
    import seaborn as sns

    if group_cols is None:
        group_cols = [x, "subject"]

    sns.lineplot(data=data, x=x, y=y, ax=ax, color=color, label=label, errorbar=None)

    grouped = data.groupby(group_cols)[y].mean().groupby(x)
    score = grouped.mean().values
    err = grouped.sem().values

    ax.fill_between(
        data[x].unique(),
        score - err,
        score + err,
        color=color,
        alpha=0.3,
    )


def add_time_markers(ax, dataset: str, ymin: float, ymax: float):
    """Add vertical lines at key time points based on dataset."""
    ax.vlines(0, ymin, ymax, color="dimgrey", linestyle="-", linewidth=0.8, zorder=0)

    if dataset == "things_eeg_2":
        ax.vlines(0.1, ymin, ymax, color="dimgrey", linestyle="--", linewidth=0.8, zorder=0)
        ax.vlines(0.2, ymin, ymax, color="darkgrey", linestyle="-", linewidth=0.8, zorder=0)
        ax.vlines(0.3, ymin, ymax, color="darkgrey", linestyle="--", linewidth=0.8, zorder=0)
        ax.vlines(0.4, ymin, ymax, color="silver", linestyle="-", linewidth=0.8, zorder=0)
        ax.vlines(0.5, ymin, ymax, color="silver", linestyle="--", linewidth=0.8, zorder=0)
        ax.vlines(0.6, ymin, ymax, color="gainsboro", linestyle="-", linewidth=0.8, zorder=0)
        ax.vlines(0.7, ymin, ymax, color="gainsboro", linestyle="--", linewidth=0.8, zorder=0)
    else:
        ax.vlines(0.5, ymin, ymax, color="dimgrey", linestyle="--", linewidth=0.8, zorder=0)


def plot_significance_bar(
    ax,
    sig_times: np.ndarray,
    all_times: np.ndarray,
    y_pos: float,
    color: str,
    linewidth: float = 1,
):
    """Plot a horizontal bar indicating significant time points."""
    if len(sig_times) < 2:
        return

    for xposl, xposr in zip(sig_times[:-1], sig_times[1:]):
        if np.argwhere(all_times == xposr) - np.argwhere(all_times == xposl) == 1:
            ax.plot([xposl, xposr], [y_pos, y_pos], color=color, linewidth=linewidth)


def load_cvpca_per_dim_streaming(
    cvpca_path: str | Path,
    time_min: float = -0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Load CV-PCA results and return per-PC correlation vs time, plus significance per PC.

    Uses the `r` (dims: subject, split, neuroid, time) and `null_r`
    (dims: subject, split, permute, neuroid, time) variables from results.nc.
    Here "neuroid" is the PC index.

    Returns:
        df: DataFrame with columns [subject, pc, time, r] (split-averaged per subject)
        sig_times: dict mapping 1-based PC number -> array of significant time points
    """
    cvpca = xr.open_dataset(cvpca_path)
    cvpca = cvpca.sel(time=cvpca.time >= time_min)

    time_values = cvpca.time.values
    n_pcs = cvpca.sizes["neuroid"]

    # Build per-subject DataFrame (averaged over splits)
    rows = []
    for subject in cvpca.subject.values:
        r_subj = cvpca.r.sel(subject=subject).mean("split")  # (neuroid, time)
        for i in range(n_pcs):
            r_pc = r_subj.isel(neuroid=i).values  # (time,)
            for t_idx, t in enumerate(time_values):
                rows.append({
                    "subject": subject,
                    "pc": i + 1,
                    "time": t,
                    "r": float(r_pc[t_idx]),
                })

    df = pd.DataFrame(rows)

    # Compute significance: average over subjects and splits, then cluster-correct per PC
    r_mean = cvpca.r.mean(["subject", "split"])       # (neuroid, time)
    null_r_mean = cvpca.null_r.mean(["subject", "split"])  # (permute, neuroid, time)

    sig_times = {}
    for i in range(n_pcs):
        r_pc = r_mean.isel(neuroid=i).values                  # (time,)
        null_r_pc = null_r_mean.isel(neuroid=i).values        # (permute, time)
        p_vals = cluster_correction_cached(r_pc, null_r_pc)
        sig_times[i + 1] = time_values[p_vals < 0.05]

    del cvpca
    gc.collect()

    return df, sig_times


def load_cvpca_generalization_at_time_train(
    cvpca_path: str,
    time_train_value: float,
    time_min: float = -0.05,
) -> pd.DataFrame:
    """
    Load CV-PCA generalization results for a specific time_train value.

    Args:
        cvpca_path: Path to the CV-PCA results NetCDF file
        time_train_value: The time_train value to select (will find closest)
        time_min: Minimum time value to include

    Returns:
        cdf: DataFrame with gen_cvnsr and gen_neg_cvnsr per subject, split, and time_test
    """
    cvpca = xr.open_dataset(cvpca_path)
    cvpca = cvpca.sel(time_train=cvpca.time_train >= time_min)
    cvpca = cvpca.sel(time_test=cvpca.time_test >= time_min)

    # Find closest time_train value
    time_train_values = cvpca.time_train.values
    closest_idx = np.argmin(np.abs(time_train_values - time_train_value))
    actual_time_train = time_train_values[closest_idx]

    # Select the specific time_train
    cvpca_slice = cvpca.sel(time_train=actual_time_train)

    time_test_values = cvpca_slice.time_test.values

    # gen_cvnsr shape: (subject, split, time_test)
    # gen_neg_cvnsr shape: (subject, split, time_test)
    cdf = []

    for subject in cvpca_slice.subject.values:
        for split in cvpca_slice.split.values:
            scvpca = cvpca_slice.sel(subject=subject, split=split)

            cdf.append(pd.DataFrame({
                "subject": subject,
                "split": split,
                "time": time_test_values,
                "pos_cvnsr": scvpca.gen_cvnsr.values,
                "neg_cvnsr": scvpca.gen_neg_cvnsr.values,
            }))

    cdf_combined = pd.concat(cdf, ignore_index=True)

    del cvpca, cvpca_slice
    gc.collect()

    return cdf_combined
