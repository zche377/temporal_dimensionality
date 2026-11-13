import logging
logging.basicConfig(level=logging.INFO)

import os
import torch
import numpy as np
from torchmetrics.classification import (
    Accuracy, 
    F1Score
)
from torchmetrics import (
    PearsonCorrCoef, 
    CosineSimilarity
)
from lib.utilities import SEED


import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='The variance of predictions or target is close to zero.')




def compute_metric(
    metric: str,
    y_true: list[torch.Tensor],
    y_predicted: list[torch.Tensor],
    score_across_folds: bool,
    n_permutations: int = None,
    seed: int = SEED,
    device: torch.device | str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
) -> torch.Tensor:
    # Capture the metric NAME before `metric` is rebound to a torchmetrics object below, so the
    # batched-null path (B) can gate on it.
    metric_name = metric

    if score_across_folds:
        y_true = torch.cat(y_true, 0)
        y_predicted = torch.cat(y_predicted, 0)
        num_labels = y_true.size(-1)
    else:
        num_labels = y_true[0].size(-1)

    # Classification metrics
    if metric in ["accuracy", "f1"]:
        num_classes = len(y_true.unique()) if score_across_folds else len(y_true[0].unique())
        if num_labels > 1:
            match metric:
                case "accuracy":
                    metric = Accuracy(task="multilabel", num_labels=num_labels, average=None)
                case "f1":
                    metric = F1Score(task="multilabel", num_labels=num_labels, average=None)
                case _:
                    raise ValueError("metric must be 'accuracy' or 'f1'")
        elif num_classes > 2:
            match metric:
                case "accuracy":
                    metric = Accuracy(task="multiclass", num_classes=num_classes)
                case "f1":
                    metric = F1Score(task="multiclass", num_classes=num_classes)
                case _:
                    raise ValueError("metric must be 'accuracy' or 'f1'")
        else:
            match metric:
                case "accuracy":
                    metric = Accuracy(task="binary")
                case "f1":
                    metric = F1Score(task="binary")
                case _:
                    raise ValueError("metric must be 'accuracy' or 'f1'")
    
    # Regression metrics
    elif metric in ["pearsonr", "cosine_similarity"]:
        match metric:
            case "pearsonr":
                metric = PearsonCorrCoef(num_outputs=num_labels).to(device)
            case "cosine_similarity":
                metric = CosineSimilarity().to(device)
            case _:
                raise ValueError("metric must be 'pearsonr' or 'cosine_similarity'")
    
    else:
        raise ValueError("Unsupported metric specified")

    metric = metric.to(device)
            
    if n_permutations is None:
        if score_across_folds:
            return metric(y_predicted, y_true).cpu()
        else:
            return torch.stack([
                metric(y_predicted[i], y_true[i]).cpu()
                for i in range(len(y_true))
            ])
    else:
        # Pre-generate all permutations up front, preserving the exact draw order of the original
        # per-iteration loop (score_across_folds: n_permutations draws; else: nested per-fold draws).
        rng = np.random.default_rng(seed)
        if score_across_folds:
            perms = [rng.permutation(len(y_predicted)) for _ in range(n_permutations)]
        else:
            perms = [
                [rng.permutation(len(y_predicted[i])) for i in range(len(y_true))]
                for _ in range(n_permutations)
            ]

        dev = torch.device(device)
        # B — GPU-batched pearson null. Gated to pearson + score_across_folds + cuda, behind an env
        # flag (default OFF until validated; TDIM_BATCHED_NULL=0 always forces the bit-identical A
        # path). Numerically == torchmetrics up to FP reduction order (~1e-6); only the null-DERIVED
        # rank stats (cluster_p / n_decodable_dims) are saved, so this is rank-stable.
        use_batched = (
            os.environ.get("TDIM_BATCHED_NULL", "0") == "1"
            and metric_name == "pearsonr"
            and score_across_folds
            and dev.type == "cuda"
        )
        if use_batched:
            yp = y_predicted.to(dev).float()                      # (N, K)
            yt = y_true.to(dev).float()                           # (N, K)
            n, k = yp.shape[0], yp.shape[1]
            # pre-normalize y_true once: center, /std(correction=1)
            yt = yt - yt.mean(0, keepdim=True)
            yt = yt / yt.std(0, correction=1, keepdim=True)
            perm_idx = torch.stack([
                torch.as_tensor(p, device=dev, dtype=torch.long) for p in perms
            ])                                                    # (n_perm, N)
            # chunk over permutations from live free memory; peak ~ 4 * chunk * N * K * 4 bytes
            override_gb = os.environ.get("TDIM_NULL_CHUNK_MEM_GB")
            if override_gb is not None:
                budget = float(override_gb) * (1024 ** 3)
            else:
                # no-arg form → current device (dev may be a bare "cuda" without an index, which
                # mem_get_info rejects)
                free_bytes, _ = torch.cuda.mem_get_info()
                budget = 0.1 * free_bytes
            chunk = int(budget // (4 * n * k * 4))
            chunk = max(1, min(n_permutations, chunk))
            out = []
            for start in range(0, n_permutations, chunk):
                idx = perm_idx[start:start + chunk]               # (c, N)
                xp = yp[idx]                                      # (c, N, K)
                xp.sub_(xp.mean(1, keepdim=True))
                xp.div_(xp.std(1, correction=1, keepdim=True))
                out.append((xp * yt).sum(1) / (n - 1))            # (c, K)
                del xp
            return torch.cat(out, 0).cpu()                        # (n_perm, K)

        # A — always available, metric-agnostic, VALUE-identical to the original loop. Accumulate on
        # device and copy to host ONCE at the end (removes the 1000 per-iteration .cpu() syncs that
        # serialized the GPU).
        if score_across_folds:
            scores = [metric(y_predicted[perm], y_true) for perm in perms]
        else:
            scores = [
                torch.stack([
                    metric(y_predicted[i][perm[i]], y_true[i])
                    for i in range(len(y_true))
                ])
                for perm in perms
            ]
        return torch.stack(scores).cpu()