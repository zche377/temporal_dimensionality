import logging
logging.basicConfig(level=logging.INFO)

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
        scores = []
        rng = np.random.default_rng(seed)
        for _ in range(n_permutations):
            if score_across_folds:
                shuffled_indices = rng.permutation(len(y_predicted))
                scores.append(metric(y_predicted[shuffled_indices], y_true).cpu())
            else:
                temp_scores = []
                for i in range(len(y_true)):
                    shuffled_indices = rng.permutation(len(y_predicted[i]))
                    temp_scores.append(metric(y_predicted[i][shuffled_indices], y_true[i]).cpu())
                scores.append(torch.stack(temp_scores))
        return torch.stack(scores)