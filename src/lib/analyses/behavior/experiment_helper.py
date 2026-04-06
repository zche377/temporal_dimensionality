import shutil
import logging

logging.basicConfig(level=logging.INFO)

from pathlib import Path
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import torch
from joblib import Parallel, delayed
from copy import deepcopy
import umap
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from lib.models import DJModel
from lib.analyses._utilities import _cache_path
from lib.computation.scorers import TrainTestModelScorer, TrainTestPLSSVDScorer
from lib.datasets import load_dataset, load_target_var, load_stimulus_set, load_n_subjects
from lib.datasets.hebart2022_things_behavior import load_triplet_rdm
from lib.utilities import (
    _append_path,
    SEED,
)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(
    context="paper",
    style="ticks",
    palette="Set2",
    rc={
        "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.labelsize": "small",
    },
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import matplotlib.patches as patches

from sklearn.manifold import MDS
from bonner.computation.decomposition import PCA
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)
from bonner.computation.metrics import pearson_r, spearman_r, euclidean_distance

from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import spearmanr, pearsonr

def _flattend_tril(x: torch.Tensor) -> torch.Tensor:
    lower_tri_indices = torch.tril_indices(x.size(0), x.size(1), offset=-1)
    return x[lower_tri_indices[0], lower_tri_indices[1]]

def _reshape_subject_data(data, subject, target_var, average) -> xr.DataArray:
    if average:
        return (
            data.sel(subject=subject)
            .mean("presentation")
            .transpose(target_var, "neuroid", "time")
            .rename({target_var: "presentation"})
        )
    else:
        return (
            data.sel(subject=subject)
            .stack(stack_presentation=(target_var, "presentation"))
            .transpose("stack_presentation", "neuroid", "time")
            .reset_index("stack_presentation")
            .drop_vars("presentation")
            .rename({"stack_presentation": "presentation"})
        )

def find_lowest_indices(matrix, k=5):
    x = np.array(matrix)
    
    # Get indices that would sort each column in ascending order
    sorted_indices = np.argsort(x, axis=0)
    
    # Take the first k indices (lowest k values) for each column
    lowest_k_indices = sorted_indices[:k, :]
    
    return lowest_k_indices.T

def find_random_indices_above_threshold(matrix, threshold, k=5, seed=None):
    if seed is None:
        seed = SEED
    
    x = np.array(matrix)
    rng = np.random.default_rng(seed)
    
    result = []
    
    for col in range(x.shape[1]):
        column_data = x[:, col]
        below_threshold_indices = np.where(column_data > threshold)[0]
        # logging.info(f"Column {col}: Found {len(below_threshold_indices)} indices above threshold {threshold}")
        assert len(below_threshold_indices) > k
        random_indices = rng.choice(below_threshold_indices, size=k, replace=False)
        result.append(random_indices)
    
    return result

def pdistance(x, **kwargs):
    return 1 - pearson_r(x, **kwargs)

def _eeg_rdm(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    time: float,
    distance_fn: callable,
    residual: bool = True,
):
    predictor_cache_str = f"main_analyses/{analysis}_tt_encoding.subset=False.pca=False"
    predictor_cache_path = _cache_path(
        predictor_cache_str,
        dataset,
        load_dataset_kwargs,
        scorer_kwargs,
        include_root=False,
    )
    predictor_cache_path = f"{predictor_cache_str}/dataset={dataset}"
    predictor_cache_path = _append_path(
        predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs
    )
    target_var = load_target_var(f"{dataset}_train")
    
    eeg_resid_rdm = []
    eeg_rdm = []
    for subject in range(1, load_n_subjects(dataset)+1):
        data_train = load_dataset(
            f"{dataset}_train",
            subjects=subject,
            **load_dataset_kwargs,
        )
        data_test = load_dataset(
            f"{dataset}_test",
            subjects=subject,
            **load_dataset_kwargs,
        )
        subject_data_train = _reshape_subject_data(data_train, subject, target_var, average=True).sel(time=time)
        subject_data_test = _reshape_subject_data(data_test, subject, target_var, average=True).sel(time=time)
        if subject == 1:
            embd = load_embeddings()
            target_train = embd.sel(object=subject_data_train.presentation.values)
            target_test = embd.sel(object=subject_data_test.presentation.values)
        
        
        scorer = TrainTestModelScorer(
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
        y_remain = (_2t(subject_data_test) - lmodel.predict(_2t(target_test))).cpu()
        distance = distance_fn(y_remain.T, return_diagonal=False)
        eeg_resid_rdm.append(distance)
        distance = distance_fn(_2t(subject_data_test).T.cpu(), return_diagonal=False)
        eeg_rdm.append(distance)
    eeg_resid_rdm = torch.mean(torch.stack(eeg_resid_rdm), dim=0)
    eeg_rdm = torch.mean(torch.stack(eeg_rdm), dim=0)
    return eeg_resid_rdm if residual else eeg_rdm

def _lowest_distance_indices(rdm, k=5):
    rdm = torch.eye(rdm.size(0)) * 9999 + rdm
    return find_lowest_indices(rdm, k=k)

def _random_distance_indices_above_threshold(rdm, threshold_percentile, k=5, seed=None):
    if seed is None:
        seed = SEED
    
    threshold = np.percentile(_flattend_tril(rdm), threshold_percentile)
    return find_random_indices_above_threshold(rdm, threshold=threshold, k=k, seed=seed)

def _yield_trial_indices(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    time: float,
    distance_fn: callable,
    residual: bool = True,
    n_within_cluster: int = 5,
    threshold_percentile: int = 80,
    seed: int = None,
):
    if seed is None:
        seed = SEED
    
    rdm = _eeg_rdm(
        analysis=analysis,
        dataset=dataset,
        load_dataset_kwargs=load_dataset_kwargs,
        scorer_kwargs=scorer_kwargs,
        time=time,
        distance_fn=distance_fn,
        residual=residual,
    )
    n_img = rdm.size(0)
    rng = np.random.default_rng(seed)
    l_indices = _lowest_distance_indices(rdm, k=n_within_cluster)
    r_indices = _random_distance_indices_above_threshold(rdm, threshold_percentile=threshold_percentile, k=n_within_cluster, seed=seed)
    cluster_no_l = rng.choice([0, 1], size=n_img,)
    for i in range(n_img):
        cnl = cluster_no_l[i]
        yield {
            "target": i,
            "cluster_0": l_indices[i] if cnl == 0 else r_indices[i],
            "cluster_1": r_indices[i] if cnl == 0 else l_indices[i],
            "cluster_correct": cnl,
        }
        
def _load_stimulus(
    dataset: str,
    i: int,
):
    stimulus_set = load_stimulus_set(f"{dataset}_test")
    return stimulus_set.__getitem__(i)

def yield_trial_images(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    time: float,
    distance_fn: callable,
    residual: bool = True,
    n_within_cluster: int = 5,
    threshold_percentile: int = 80,
    seed: int = None,
):
    for trial in _yield_trial_indices(
        analysis=analysis,
        dataset=dataset,
        load_dataset_kwargs=load_dataset_kwargs,
        scorer_kwargs=scorer_kwargs,
        time=time,
        distance_fn=distance_fn,
        residual=residual,
        n_within_cluster=n_within_cluster,
        threshold_percentile=threshold_percentile,
        seed=seed,
    ):
        target_img = _load_stimulus(dataset, trial["target"])
        cluster_0_imgs = [_load_stimulus(dataset, i) for i in trial["cluster_0"]]
        cluster_1_imgs = [_load_stimulus(dataset, i) for i in trial["cluster_1"]]
        yield {
            "target": target_img,
            "cluster_0": cluster_0_imgs,
            "cluster_1": cluster_1_imgs,
            "cluster_correct": trial["cluster_correct"],
            # ADD THESE STIMULUS INDEX FIELDS:
            "target_idx": trial["target"], 
            "cluster_0_indices": trial["cluster_0"],
            "cluster_1_indices": trial["cluster_1"],
        }