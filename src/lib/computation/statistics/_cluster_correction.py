import numpy as np
import torch
from scipy import ndimage
from lib.computation.statistics import compute_p



def _cluster_nulls(null: np.ndarray, alpha: float) -> np.ndarray:
    # Leave-one-out max-cluster-mass null distribution.
    #
    # For each permutation i and cell t, the original code compared perm i's statistic against the
    # OTHER perms' statistics via a compute_p('greater') over the leave-one-out stack
    #   p_nulls = stack([delete(null, i, axis=0) for i in range(n_perm)])   # (n_perm-1, T, n_perm)
    # which is O(n_perm^2 * T) in memory + a full CUDA upload -- the dominant cost of the in-lib
    # significance.  compute_p('greater') reduces to
    #   count[i, t] = #{ j != i : null[i, t] > null[j, t] } = #{ j : null[j, t] < null[i, t] }
    #   p_vals[i, t] = 1 - (count[i, t] + 1) / n_perm            (denominator = (n_perm-1) + 1)
    # so p_vals depends on `null` only through the strict-less RANK of each value within its column.
    # We compute that rank in O(n_perm log n_perm) per column with a sort + searchsorted, then feed
    # the identical integer `count` through compute_p's exact final arithmetic on CUDA -> the boolean
    # mask `p_vals < alpha` (all that the max-cluster-mass loop below actually consumes) is
    # byte-identical to the original.  The mass values themselves come from `null[i]` directly (exact).
    #
    # NOTE: like the original (its `null[i][:, None]` and the swapaxes/compute_p shape assertion),
    # this supports a single 1-D cluster axis, i.e. `null` of shape (n_perm, T).
    null_t = torch.as_tensor(null).to("cuda")
    n_perm = null_t.shape[0]
    denominator = n_perm  # (n_perm - 1) leave-one-out entries + 1, matching the original compute_p

    # strict-less count per column, via a per-column sort + searchsorted (side='left').
    # NaN breaks searchsorted's sorted-boundary invariant, so replace NaN with +inf first: a finite
    # value's count then excludes the former-NaNs (+inf is never < v), matching the original `>`; a
    # NaN query (now +inf) is forced back to 0 (nothing is < NaN), also matching the original.
    is_nan = torch.isnan(null_t)
    filled = torch.where(is_nan, torch.full_like(null_t, float("inf")), null_t)
    sorted_null, _ = torch.sort(filled, dim=0)                       # (n_perm, T)
    count = torch.searchsorted(
        sorted_null.transpose(0, 1).contiguous(),
        filled.transpose(0, 1).contiguous(),
        side="left",
    ).transpose(0, 1)                                                # (n_perm, T)
    count = torch.where(is_nan, torch.zeros_like(count), count)
    p_vals = (1 - ((count + 1) / denominator)).cpu().numpy()         # matches compute_p('greater')

    # Per-perm max-cluster-mass, vectorized. The original looped `ndimage.label` over each of the
    # n_perm rows (a tiny 1-D label call each) then per-perm max — that Python loop is the real cost
    # (~80x the GPU p-value math). Instead label the whole (n_perm, T) mask ONCE with a structure that
    # connects ONLY along the cluster axis T (never across perm rows), so each global cluster is exactly
    # one perm-row's contiguous run; then reduce cluster masses and take the per-perm max. Bit-identical
    # to the loop: `sum_labels` sums each run in the same order, the consumer treats the result as an
    # unordered set, and perms with no cluster are dropped (as the original's `if num_features > 0`).
    mask = p_vals < alpha
    structure = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])          # 4-connectivity along T only
    labels, num_features = ndimage.label(mask, structure=structure)
    if num_features == 0:
        return np.array([])
    idx = np.arange(1, num_features + 1)
    masses = ndimage.sum_labels(null, labels, index=idx)            # sum of null over each cluster's cells
    perm_of = ndimage.minimum(
        np.broadcast_to(np.arange(len(p_vals))[:, None], mask.shape), labels, index=idx,
    ).astype(int)                                                    # which perm row each cluster is in
    out = np.full(len(p_vals), -np.inf)
    np.maximum.at(out, perm_of, masses)                             # per-perm max cluster mass
    return out[out > -np.inf]                                       # perms with >=1 cluster, ascending

def cluster_correction(
    true: np.ndarray,
    null: np.ndarray,
    alpha=0.05,
) -> np.ndarray:
    p_values = compute_p(true, null, 'greater')

    cluster_nulls = _cluster_nulls(null, alpha=alpha)
    cluster_ps = p_values.copy()

    labels, num_features = ndimage.label(p_values < alpha)
    if num_features > 0:
        for i in range(1, num_features + 1):
            cluster_indices = labels == i
            sum_val = np.sum(np.sum(true[cluster_indices]) > cluster_nulls)
            cluster_ps[cluster_indices] = 1 - (sum_val / len(cluster_nulls))

    return cluster_ps
