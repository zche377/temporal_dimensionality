import numpy as np
from scipy import ndimage
from lib.computation.statistics import compute_p



def _cluster_nulls(null: np.ndarray, alpha: float) -> np.ndarray:
    p_trues = null.copy().swapaxes(0, -1)
    p_nulls = np.stack([np.delete(null, i, axis=0) for i in range(len(null))], axis=-1)
    p_vals = compute_p(p_trues, p_nulls, 'greater').swapaxes(-1, 0)
    
    cluster_nulls = []
    for i in range(len(p_vals)):
        labels, num_features = ndimage.label(p_vals[i] < alpha)
        if num_features > 0:
            cluster_nulls.append(np.sum(null[i][:, None] * (labels[..., None] == np.arange(1, num_features + 1)), axis=0).max())
            
    return np.array(cluster_nulls)

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