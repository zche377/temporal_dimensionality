import shutil
import logging
import gc

logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import torch
from joblib import Parallel, delayed
from copy import deepcopy

from lib.analyses._utilities import _cache_path
from lib.computation.scorers import TrainTestModelScorer, TrainTestPLSSVDScorer
from lib.computation.statistics import cluster_correction
from lib.datasets import load_dataset, load_target_var, load_n_subjects
from lib.utilities import (
    _append_path,
    SEED,
)
from lib.datasets.hebart2022_things_behavior import sort_embeddings
from lib.models import DJModel

from bonner.computation.decomposition import PCA
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)


def _reshape_subject_data(data, subject, target_var, average=False,) -> xr.DataArray:
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

def _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=None):
    subject_data_train = _reshape_subject_data(data_train, subject, target_var, average)
    subject_data_test = _reshape_subject_data(data_test, subject, target_var, average)
    if reconstruct_with_pc:
        assert average
        if num_pc is not None:
            re_train, re_test = [], []
            for time in subject_data_train.time.values:
                pmodel = PCA()
                pmodel.fit(torch.tensor(subject_data_train.sel(time=time).values))
                transformed = pmodel.transform(torch.tensor(subject_data_train.sel(time=time).values))
                transformed[:, num_pc:] = 0
                re_train.append(
                    subject_data_train.sel(time=time).copy(
                        data=pmodel.inverse_transform(transformed)
                    ).assign_coords({"pc": num_pc})
                )
                transformed = pmodel.transform(torch.tensor(subject_data_test.sel(time=time).values))
                transformed[:, num_pc:] = 0
                re_test.append(
                    subject_data_test.sel(time=time).copy(
                        data=pmodel.inverse_transform(transformed)
                    ).assign_coords({"pc": num_pc})
                )
            subject_data_train = xr.concat(re_train, dim="time")
            subject_data_test = xr.concat(re_test, dim="time")
    return subject_data_train.sortby("img_files"), subject_data_test.sortby("img_files")

def _model_srpz_decoding_per_node(
    dataset: str,
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    pca: bool,
    reconstruct_with_pc: bool,
    model_uid: str,
    n_components,
    nodes: str = "Identity",
    num_pc: int = None,
    node_subset: bool = True,
):
    """Generator that yields one node's score at a time to reduce memory usage."""
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=num_pc)

    model = DJModel(model_uid, hook="srp", n_components=n_components, nodes=nodes, node_subset=node_subset)
    target = model(dataset, dataloader_kwargs={"batch_size": 32})

    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer

    for node, feature_map in target.items():
        feature_map_train = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_train.img_files))).sortby("img_files")
        feature_map_test = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_test.img_files))).sortby("img_files")

        # z-score the feature maps
        temp = torch.tensor(feature_map_train.values, device="cuda")
        mean = temp.mean(dim=0)
        std = temp.std(dim=0)
        temp = (temp-mean)/std
        feature_map_train = feature_map_train.copy(data=temp.cpu().numpy())
        temp = torch.tensor(feature_map_test.values, device="cuda")
        temp = (temp-mean)/std
        feature_map_test = feature_map_test.copy(data=temp.cpu().numpy())

        assert np.all(feature_map_train.img_files.values==subject_data_train.img_files.values)

        if pca:
            pmodel = PCA()
            pmodel.fit(torch.tensor(feature_map_train.values))
            feature_map_train = feature_map_train.copy(data=pmodel.transform(torch.tensor(feature_map_train.values)))
            feature_map_test = feature_map_test.copy(data=pmodel.transform(torch.tensor(feature_map_test.values)))
            feature_map_train["neuroid"] = np.arange(len(feature_map_train.neuroid))+1
            feature_map_test["neuroid"] = np.arange(len(feature_map_test.neuroid))+1

        time_scores = Parallel(n_jobs=3, backend="loky")(
            delayed(scorer_fn(
                **scorer_kwargs,
                cache_predictors=True,
                cache_subpath=f"{predictor_cache_path}/target=node={node}/predictor=time={t}",
            ))(
                predictor_train=subject_data_train.sel(time=t),
                target_train=feature_map_train,
                predictor_test=subject_data_test.sel(time=t),
                target_test=feature_map_test,
                target_dim="neuroid",
            ) for t in subject_data_test.time.values)

        time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data_test.time.values)]
        node_score = xr.concat(time_scores, dim="time").assign_coords({"node": node, "subject": subject})

        # Clear feature maps before yielding
        del feature_map_train, feature_map_test, time_scores
        gc.collect()

        yield node, node_score


def _model_srpz_decoding(
    dataset: str,
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    pca: bool,
    reconstruct_with_pc: bool,
    model_uid: str,
    n_components,
    nodes: str = "Identity",
    num_pc: int = None,
    node_subset: bool = True,
) -> xr.Dataset:
    """Original interface - collects all nodes and returns concatenated result."""
    scores = []
    for node, node_score in _model_srpz_decoding_per_node(
        dataset=dataset,
        data_train=data_train,
        data_test=data_test,
        subject=subject,
        target_var=target_var,
        scorer_kwargs=scorer_kwargs,
        predictor_cache_path=predictor_cache_path,
        pca=pca,
        reconstruct_with_pc=reconstruct_with_pc,
        model_uid=model_uid,
        n_components=n_components,
        nodes=nodes,
        num_pc=num_pc,
        node_subset=node_subset,
    ):
        scores.append(node_score)

    return xr.concat(scores, dim="node")

def _model_srp_decoding(
    dataset: str,
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    pca: bool,
    reconstruct_with_pc: bool,
    model_uid: str,
    n_components,
    nodes: str = "Identity",
    num_pc: int = None,
    node_subset: bool = True,
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=num_pc)

    model = DJModel(model_uid, hook="srp", n_components=n_components, nodes=nodes, node_subset=node_subset)
    target = model(dataset, dataloader_kwargs={"batch_size": 32})
    
    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer
        
    scores = []
    for node, feature_map in target.items():
        feature_map_train = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_train.img_files))).sortby("img_files")
        feature_map_test = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_test.img_files))).sortby("img_files")
        
        assert np.all(feature_map_train.img_files.values==subject_data_train.img_files.values)
        
        if pca:
            pmodel = PCA()
            pmodel.fit(torch.tensor(feature_map_train.values))
            feature_map_train = feature_map_train.copy(data=pmodel.transform(torch.tensor(feature_map_train.values)))
            feature_map_test = feature_map_test.copy(data=pmodel.transform(torch.tensor(feature_map_test.values)))
            feature_map_train["neuroid"] = np.arange(len(feature_map_train.neuroid))+1
            feature_map_test["neuroid"] = np.arange(len(feature_map_test.neuroid))+1
            
        time_scores = Parallel(n_jobs=3, backend="loky")(
            delayed(scorer_fn(
                **scorer_kwargs,
                cache_predictors=True,
                cache_subpath=f"{predictor_cache_path}/target=node={node}/predictor=time={t}",
            ))(
                predictor_train=subject_data_train.sel(time=t),
                target_train=feature_map_train,
                predictor_test=subject_data_test.sel(time=t),
                target_test=feature_map_test,
                target_dim="neuroid",
            ) for t in subject_data_test.time.values)
    
        time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data_test.time.values)]
        scores.append(xr.concat(time_scores, dim="time").assign_coords({"node": node}))

    return xr.concat(scores, dim="node").assign_coords({"subject": subject})

def _behavior_decoding(
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    subset: bool,
    pca: bool,
    reconstruct_with_pc: bool,
    num_pc: int = None,
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=num_pc)

    if subset:
        embd = sort_embeddings()
    else:
        embd = load_embeddings()
    target_train = embd.sel(object=subject_data_train.presentation.values)
    target_test = embd.sel(object=subject_data_test.presentation.values)
    
    # z-score
    temp =  torch.tensor(target_train.values, device="cuda")
    mean = temp.mean(dim=0)
    std = temp.std(dim=0)
    target_train = xr.DataArray(
        ((temp-mean)/std).cpu().numpy(),
        dims=["presentation", "neuroid"],
    )
    temp =  torch.tensor(target_test.values, device="cuda")
    target_test = xr.DataArray(
        ((temp-mean)/std).cpu().numpy(),
        dims=["presentation", "neuroid"],
    )
    target_train["neuroid"] = np.arange(len(target_train.neuroid))+1
    target_test["neuroid"] = np.arange(len(target_test.neuroid))+1
    
    if pca:
        logging.info("Applying PCA to target data")
        pmodel = PCA()
        pmodel.fit(torch.tensor(target_train.values))
        target_train = target_train.copy(data=pmodel.transform(torch.tensor(target_train.values)))
        target_test = target_test.copy(data=pmodel.transform(torch.tensor(target_test.values)))
        target_train["neuroid"] = np.arange(len(target_train.neuroid))+1
        target_test["neuroid"] = np.arange(len(target_test.neuroid))+1
    
    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer
    
    # logging.info(subject_data_train)
    # logging.info(subject_data_test)
    # logging.info(target_train)
    # logging.info(target_test)
    # quit()
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=time={t}",
        ))(
            predictor_train=subject_data_train.sel(time=t),
            target_train=target_train,
            predictor_test=subject_data_test.sel(time=t),
            target_test=target_test,
            target_dim="neuroid",
        ) for t in subject_data_test.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data_test.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})

def _everything_decoding(
    dataset: str,
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    subset: bool,
    pca: bool,
    reconstruct_with_pc: bool,
    model_uid: str,
    n_components,
    nodes: str = "Identity",
    num_pc: int = None,
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=num_pc)
    
    target_train, target_test = dict(), dict()
    if subset:
        embd = sort_embeddings()
    else:
        embd = load_embeddings()
        
    target_train["behavior"] = embd.sel(object=subject_data_train.presentation.values).rename({"object": "presentation", "behavior": "neuroid"}).values
    target_test["behavior"] = embd.sel(object=subject_data_test.presentation.values).rename({"object": "presentation", "behavior": "neuroid"}).values
    
    model = DJModel(model_uid, hook="srp", n_components=n_components, nodes=nodes)
    model_target = model(dataset, dataloader_kwargs={"batch_size": 32})
    
    i = 0
    for node, feature_map in model_target.items():
        i+=1
        if i not in [1, len(model_target)]:
            continue
        target_train[node] = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_train.img_files))).sortby("img_files")
        assert np.all(target_train[node].img_files.values==subject_data_train.img_files.values)
        target_train[node] = target_train[node].values
        target_test[node] = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_test.img_files))).sortby("img_files").values
    del model_target
    
    # z-score
    temp =  torch.tensor(np.concatenate([v for v in target_train.values()], axis=1), device="cuda")
    mean = temp.mean(dim=0)
    std = temp.std(dim=0)
    target_train = xr.DataArray(
        ((temp-mean)/std).cpu().numpy(),
        dims=["presentation", "neuroid"],
    )
    temp =  torch.tensor(np.concatenate([v for v in target_test.values()], axis=1), device="cuda")
    target_test = xr.DataArray(
        ((temp-mean)/std).cpu().numpy(),
        dims=["presentation", "neuroid"],
    )
    target_train["neuroid"] = np.arange(len(target_train.neuroid))+1
    target_test["neuroid"] = np.arange(len(target_test.neuroid))+1
    
    if pca:
        pmodel = PCA()
        pmodel.fit(torch.tensor(target_train.values))
        target_train = target_train.copy(data=pmodel.transform(torch.tensor(target_train.values)))
        target_test = target_test.copy(data=pmodel.transform(torch.tensor(target_test.values)))
        target_train["neuroid"] = np.arange(len(target_train.neuroid))+1
        target_test["neuroid"] = np.arange(len(target_test.neuroid))+1
    
    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer
    
    time_scores = Parallel(n_jobs=1, backend="loky")(
        delayed(scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=everything/predictor=time={t}",
        ))(
            predictor_train=subject_data_train.sel(time=t),
            target_train=target_train,
            predictor_test=subject_data_test.sel(time=t),
            target_test=target_test,
            target_dim="neuroid",
        ) for t in subject_data_test.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data_test.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})
    
def _get_null_key(n_permutations: int) -> str:
    """Get the null variable name for a given number of permutations."""
    return f"null.pearsonr.n_permutations={n_permutations}"


def _compute_n_decodable_dims(score: xr.Dataset, n_permutations: int, target_var: str = "neuroid") -> xr.DataArray:
    """
    Compute number of decodable dimensions at each time point for a single subject.

    For each neuroid/dimension, run cluster correction across time using its null distribution,
    then count how many are significant at each time point.

    Args:
        score: Dataset with pearsonr (time, neuroid) and null (time, permutation, neuroid)
        n_permutations: Number of permutations used
        target_var: Name of the target dimension (default: "neuroid")

    Returns:
        DataArray with n_decodable_dims at each time point
    """
    null_key = _get_null_key(n_permutations)

    # pearsonr shape: (time, neuroid) or (neuroid, time) depending on concat order
    pearsonr = score.pearsonr
    null = score[null_key]

    # Ensure consistent ordering: (neuroid, time)
    if pearsonr.dims[0] == "time":
        pearsonr = pearsonr.transpose(target_var, "time")

    # null shape: (time, permutation, neuroid) -> (neuroid, permutation, time)
    null = null.transpose(target_var, "permutation", "time")

    time_values = pearsonr.time.values
    neuroid_values = pearsonr[target_var].values
    n_neuroids = len(neuroid_values)
    n_time = len(time_values)

    # Compute cluster-corrected p-values for each neuroid
    neuroid_p_vals = np.zeros((n_neuroids, n_time))
    for i, neuroid in enumerate(neuroid_values):
        # pearsonr for this neuroid: (time,)
        # null for this neuroid: (permutation, time)
        neuroid_p_vals[i] = cluster_correction(
            pearsonr.sel({target_var: neuroid}).values,
            null.sel({target_var: neuroid}).values
        )

    # Count number of significant neuroids at each time point
    n_decodable_dims = np.sum(neuroid_p_vals < 0.05, axis=0)

    return xr.DataArray(
        n_decodable_dims,
        dims=("time",),
        coords={"time": time_values},
        attrs={"n_permutations": n_permutations, "n_neuroids": n_neuroids}
    )


def _compute_n_decodable_dims_per_node(score: xr.Dataset, n_permutations: int, target_var: str = "latent") -> xr.DataArray:
    """
    Compute number of decodable dimensions at each time point for a single subject with node dimension.

    For each node and latent, run cluster correction across time using its null distribution,
    then count how many are significant at each time point per node.

    Args:
        score: Dataset with pearsonr (node, time, latent) and null (node, time, permutation, latent)
        n_permutations: Number of permutations used
        target_var: Name of the target dimension (default: "latent")

    Returns:
        DataArray with n_decodable_dims at each (node, time)
    """
    null_key = _get_null_key(n_permutations)

    nodes = score.node.values
    time_values = score.time.values
    n_nodes = len(nodes)
    n_time = len(time_values)

    n_decodable_dims = np.zeros((n_nodes, n_time))

    for i, node in enumerate(nodes):
        node_score = score.sel(node=node)
        pearsonr = node_score.pearsonr
        null = node_score[null_key]

        # Ensure consistent ordering: (latent, time)
        if pearsonr.dims[0] == "time":
            pearsonr = pearsonr.transpose(target_var, "time")

        # null shape: (time, permutation, latent) -> (latent, permutation, time)
        null = null.transpose(target_var, "permutation", "time")

        latent_values = pearsonr[target_var].values
        n_latents = len(latent_values)

        # Compute cluster-corrected p-values for each latent
        latent_p_vals = np.zeros((n_latents, n_time))
        for j, latent in enumerate(latent_values):
            latent_p_vals[j] = cluster_correction(
                pearsonr.sel({target_var: latent}).values,
                null.sel({target_var: latent}).values
            )

        # Count number of significant latents at each time point for this node
        n_decodable_dims[i] = np.sum(latent_p_vals < 0.05, axis=0)

    return xr.DataArray(
        n_decodable_dims,
        dims=("node", "time"),
        coords={"node": nodes, "time": time_values},
        attrs={"n_permutations": n_permutations}
    )


class _IncrementalStatsAccumulator:
    """
    Incrementally accumulate statistics for computing cluster-corrected p-values.

    Uses Welford's online algorithm to compute running mean and variance
    without storing all data in memory.

    Also accumulates per-latent data for computing n_decodable_dims.
    """
    def __init__(self, n_permutations: int, has_node: bool = False, compute_dimensionality: bool = True):
        self.n_permutations = n_permutations
        self.has_node = has_node
        self.null_key = _get_null_key(n_permutations)
        self.compute_dimensionality = compute_dimensionality

        self.n_subjects = 0
        self.pearsonr_sum = None
        self.pearsonr_sq_sum = None  # For computing variance
        self.null_sum = None
        self.time_values = None
        self.node_values = None
        self.target_var = None  # Will be inferred from first score

        # Per-latent accumulators for dimensionality
        self.latent_pearsonr_sum = None
        self.latent_null_sum = None
        self.latent_values = None

    def add_subject(self, score: xr.Dataset) -> None:
        """Add a subject's data to the running statistics."""
        # Infer target_var from score dimensions (exclude 'time' and 'node')
        if self.target_var is None:
            score_dims = set(score.pearsonr.dims)
            non_target_dims = {"time", "node"}
            target_dims = score_dims - non_target_dims
            if len(target_dims) == 1:
                self.target_var = target_dims.pop()
            else:
                raise ValueError(f"Cannot infer target_var from dimensions {score_dims}")

        # Average over target_var for significance testing
        pearsonr_agg = score.pearsonr.mean(self.target_var).values

        # Handle null distribution shape
        null_data = score[self.null_key].mean(self.target_var)
        if self.has_node:
            # Shape: (node, time, permutation) -> (node, permutation, time)
            null_agg = null_data.transpose("node", "permutation", "time").values
        else:
            # Shape: (time, permutation) -> (permutation, time)
            null_agg = null_data.transpose("permutation", "time").values

        # Initialize accumulators on first subject
        if self.n_subjects == 0:
            self.pearsonr_sum = np.zeros_like(pearsonr_agg)
            self.pearsonr_sq_sum = np.zeros_like(pearsonr_agg)
            self.null_sum = np.zeros_like(null_agg)
            self.time_values = score.time.values
            if self.has_node:
                self.node_values = score.node.values

        # Update running sums
        self.pearsonr_sum += pearsonr_agg
        self.pearsonr_sq_sum += pearsonr_agg ** 2
        self.null_sum += null_agg
        self.n_subjects += 1

        # Accumulate per-latent data for dimensionality computation
        if self.compute_dimensionality:
            # Keep per-latent data: shape (latent, time)
            pearsonr_latent = score.pearsonr.transpose(self.target_var, "time").values
            # null shape: (latent, permutation, time)
            null_latent = score[self.null_key].transpose(self.target_var, "permutation", "time").values

            if self.latent_values is None:
                self.latent_values = score[self.target_var].values
                self.latent_pearsonr_sum = np.zeros_like(pearsonr_latent)
                self.latent_null_sum = np.zeros_like(null_latent)

            self.latent_pearsonr_sum += pearsonr_latent
            self.latent_null_sum += null_latent

    def compute_stats(self) -> xr.Dataset:
        """Compute final statistics and return as xr.Dataset."""
        if self.n_subjects == 0:
            raise ValueError("No subjects added")

        # Compute means
        pearsonr_mean = self.pearsonr_sum / self.n_subjects
        null_mean = self.null_sum / self.n_subjects

        # Compute SEM using variance formula: var = E[X^2] - E[X]^2
        pearsonr_var = (self.pearsonr_sq_sum / self.n_subjects) - (pearsonr_mean ** 2)
        pearsonr_sem = np.sqrt(pearsonr_var / self.n_subjects)

        # Compute cluster-corrected p-values
        if self.has_node:
            cluster_p = np.zeros_like(pearsonr_mean)
            for i in range(pearsonr_mean.shape[0]):  # iterate over nodes
                cluster_p[i] = cluster_correction(pearsonr_mean[i], null_mean[i])

            dims = ("node", "time")
            coords = {"node": self.node_values, "time": self.time_values}
        else:
            cluster_p = cluster_correction(pearsonr_mean, null_mean)

            dims = ("time",)
            coords = {"time": self.time_values}

        # Create stats dataset
        stats = xr.Dataset({
            "pearsonr_mean": xr.DataArray(pearsonr_mean, dims=dims, coords=coords),
            "pearsonr_sem": xr.DataArray(pearsonr_sem, dims=dims, coords=coords),
            "cluster_p": xr.DataArray(cluster_p, dims=dims, coords=coords),
        })

        # Compute n_decodable_dims
        if self.compute_dimensionality and self.latent_pearsonr_sum is not None:
            lat_pearsonr_mean = self.latent_pearsonr_sum / self.n_subjects  # (latent, time)
            lat_null_mean = self.latent_null_sum / self.n_subjects  # (latent, permutation, time)

            n_latents = lat_pearsonr_mean.shape[0]
            n_time = lat_pearsonr_mean.shape[1]
            latent_p_vals = np.zeros((n_latents, n_time))
            for j in range(n_latents):
                latent_p_vals[j] = cluster_correction(lat_pearsonr_mean[j], lat_null_mean[j])

            n_decodable_dims = np.sum(latent_p_vals < 0.05, axis=0)
            stats["n_decodable_dims"] = xr.DataArray(n_decodable_dims, dims=("time",), coords={"time": self.time_values})

        stats.attrs["n_subjects"] = self.n_subjects
        stats.attrs["n_permutations"] = self.n_permutations

        return stats


class _BhvDimensionalityAccumulator:
    """
    Accumulate per-neuroid statistics for computing number of decodable behavioral dimensions.

    Instead of averaging across neuroids first, this accumulator keeps per-neuroid data
    to compute cluster-corrected p-values for each neuroid independently, then counts
    how many are significant at each time point.
    """
    def __init__(self, n_permutations: int):
        self.n_permutations = n_permutations
        self.null_key = _get_null_key(n_permutations)

        self.n_subjects = 0
        # Per-neuroid accumulators: shape (n_neuroids, n_time)
        self.pearsonr_sum = None
        self.null_sum = None  # shape (n_neuroids, n_permutations, n_time)
        self.time_values = None
        self.neuroid_values = None
        self.target_var = None

    def add_subject(self, score: xr.Dataset) -> None:
        """Add a subject's data to the running statistics (per-neuroid)."""
        # Infer target_var from score dimensions (exclude 'time')
        if self.target_var is None:
            score_dims = set(score.pearsonr.dims)
            non_target_dims = {"time"}
            target_dims = score_dims - non_target_dims
            if len(target_dims) == 1:
                self.target_var = target_dims.pop()
            else:
                raise ValueError(f"Cannot infer target_var from dimensions {score_dims}")

        # Keep per-neuroid data: shape (neuroid, time)
        pearsonr_data = score.pearsonr.transpose(self.target_var, "time").values

        # null shape: (permutation, neuroid) per time -> need (neuroid, permutation, time)
        null_data = score[self.null_key].transpose(self.target_var, "permutation", "time").values

        # Initialize accumulators on first subject
        if self.n_subjects == 0:
            self.pearsonr_sum = np.zeros_like(pearsonr_data)
            self.null_sum = np.zeros_like(null_data)
            self.time_values = score.time.values
            self.neuroid_values = score[self.target_var].values

        # Update running sums
        self.pearsonr_sum += pearsonr_data
        self.null_sum += null_data
        self.n_subjects += 1

    def compute_stats(self) -> xr.Dataset:
        """Compute final statistics: n_decodable_dims per time point."""
        if self.n_subjects == 0:
            raise ValueError("No subjects added")

        # Compute means
        pearsonr_mean = self.pearsonr_sum / self.n_subjects  # (neuroid, time)
        null_mean = self.null_sum / self.n_subjects  # (neuroid, permutation, time)

        # Compute cluster-corrected p-values for each neuroid
        n_neuroids = pearsonr_mean.shape[0]
        n_time = pearsonr_mean.shape[1]

        neuroid_p_vals = np.zeros((n_neuroids, n_time))
        for i in range(n_neuroids):
            # null_mean[i] has shape (permutation, time)
            neuroid_p_vals[i] = cluster_correction(pearsonr_mean[i], null_mean[i])

        # Count number of significant neuroids at each time point
        n_decodable_dims = np.sum(neuroid_p_vals < 0.05, axis=0)

        # Also compute mean pearsonr (averaged across neuroids) for backwards compatibility
        pearsonr_mean_avg = pearsonr_mean.mean(axis=0)
        null_mean_avg = null_mean.mean(axis=0)
        cluster_p = cluster_correction(pearsonr_mean_avg, null_mean_avg)

        # Create stats dataset
        stats = xr.Dataset({
            "pearsonr_mean": xr.DataArray(pearsonr_mean_avg, dims=("time",), coords={"time": self.time_values}),
            "cluster_p": xr.DataArray(cluster_p, dims=("time",), coords={"time": self.time_values}),
            "n_decodable_dims": xr.DataArray(n_decodable_dims, dims=("time",), coords={"time": self.time_values}),
            "neuroid_cluster_p": xr.DataArray(neuroid_p_vals, dims=(self.target_var, "time"),
                                               coords={self.target_var: self.neuroid_values, "time": self.time_values}),
        })

        stats.attrs["n_subjects"] = self.n_subjects
        stats.attrs["n_permutations"] = self.n_permutations
        stats.attrs["n_neuroids"] = n_neuroids

        return stats


class _PerNodeStatsAccumulator:
    """
    Accumulate statistics per-node to reduce memory usage.

    Instead of storing all nodes at once, this accumulator handles one node at a time
    and stores per-node accumulators in a dictionary.

    Also accumulates per-latent data for computing n_decodable_dims per node.
    """
    def __init__(self, n_permutations: int, compute_dimensionality: bool = True):
        self.n_permutations = n_permutations
        self.null_key = _get_null_key(n_permutations)
        self.target_var = None
        self.compute_dimensionality = compute_dimensionality

        # Per-node accumulators: {node_name: {pearsonr_sum, pearsonr_sq_sum, null_sum, n_subjects}}
        self.node_accumulators = {}
        # Per-node, per-latent accumulators for dimensionality: {node_name: {pearsonr_sum, null_sum}}
        self.node_latent_accumulators = {}
        self.time_values = None
        self.latent_values = None

    def add_node_score(self, node: str, score: xr.Dataset) -> None:
        """Add a single node's score from one subject."""
        # Infer target_var from score dimensions (exclude 'time')
        if self.target_var is None:
            score_dims = set(score.pearsonr.dims)
            non_target_dims = {"time"}
            target_dims = score_dims - non_target_dims
            if len(target_dims) == 1:
                self.target_var = target_dims.pop()
            else:
                raise ValueError(f"Cannot infer target_var from dimensions {score_dims}")

        # Average over target_var for significance testing
        pearsonr_agg = score.pearsonr.mean(self.target_var).values

        # Handle null distribution: (time, permutation) -> (permutation, time)
        null_data = score[self.null_key].mean(self.target_var)
        null_agg = null_data.transpose("permutation", "time").values

        # Store time values
        if self.time_values is None:
            self.time_values = score.time.values

        # Initialize or update node accumulator
        if node not in self.node_accumulators:
            self.node_accumulators[node] = {
                "pearsonr_sum": np.zeros_like(pearsonr_agg),
                "pearsonr_sq_sum": np.zeros_like(pearsonr_agg),
                "null_sum": np.zeros_like(null_agg),
                "n_subjects": 0,
            }

        acc = self.node_accumulators[node]
        acc["pearsonr_sum"] += pearsonr_agg
        acc["pearsonr_sq_sum"] += pearsonr_agg ** 2
        acc["null_sum"] += null_agg
        acc["n_subjects"] += 1

        # Accumulate per-latent data for dimensionality computation
        if self.compute_dimensionality:
            # Keep per-latent data: shape (latent, time)
            pearsonr_latent = score.pearsonr.transpose(self.target_var, "time").values
            # null shape: (latent, permutation, time)
            null_latent = score[self.null_key].transpose(self.target_var, "permutation", "time").values

            if self.latent_values is None:
                self.latent_values = score[self.target_var].values

            if node not in self.node_latent_accumulators:
                self.node_latent_accumulators[node] = {
                    "pearsonr_sum": np.zeros_like(pearsonr_latent),
                    "null_sum": np.zeros_like(null_latent),
                }

            lat_acc = self.node_latent_accumulators[node]
            lat_acc["pearsonr_sum"] += pearsonr_latent
            lat_acc["null_sum"] += null_latent

    def compute_stats(self) -> xr.Dataset:
        """Compute final statistics across all nodes."""
        if not self.node_accumulators:
            raise ValueError("No nodes added")

        node_names = list(self.node_accumulators.keys())
        n_nodes = len(node_names)
        n_time = len(self.time_values)

        # Get n_subjects from first node (should be same for all)
        n_subjects = self.node_accumulators[node_names[0]]["n_subjects"]

        pearsonr_mean = np.zeros((n_nodes, n_time))
        pearsonr_sem = np.zeros((n_nodes, n_time))
        cluster_p = np.zeros((n_nodes, n_time))
        n_decodable_dims = np.zeros((n_nodes, n_time)) if self.compute_dimensionality else None

        for i, node in enumerate(node_names):
            acc = self.node_accumulators[node]

            # Compute means
            node_pearsonr_mean = acc["pearsonr_sum"] / acc["n_subjects"]
            null_mean = acc["null_sum"] / acc["n_subjects"]

            # Compute SEM
            node_var = (acc["pearsonr_sq_sum"] / acc["n_subjects"]) - (node_pearsonr_mean ** 2)
            node_sem = np.sqrt(node_var / acc["n_subjects"])

            # Compute cluster-corrected p-values
            node_cluster_p = cluster_correction(node_pearsonr_mean, null_mean)

            pearsonr_mean[i] = node_pearsonr_mean
            pearsonr_sem[i] = node_sem
            cluster_p[i] = node_cluster_p

            # Compute n_decodable_dims for this node
            if self.compute_dimensionality and node in self.node_latent_accumulators:
                lat_acc = self.node_latent_accumulators[node]
                lat_pearsonr_mean = lat_acc["pearsonr_sum"] / n_subjects  # (latent, time)
                lat_null_mean = lat_acc["null_sum"] / n_subjects  # (latent, permutation, time)

                n_latents = lat_pearsonr_mean.shape[0]
                latent_p_vals = np.zeros((n_latents, n_time))
                for j in range(n_latents):
                    latent_p_vals[j] = cluster_correction(lat_pearsonr_mean[j], lat_null_mean[j])

                n_decodable_dims[i] = np.sum(latent_p_vals < 0.05, axis=0)

        dims = ("node", "time")
        coords = {"node": node_names, "time": self.time_values}

        stats = xr.Dataset({
            "pearsonr_mean": xr.DataArray(pearsonr_mean, dims=dims, coords=coords),
            "pearsonr_sem": xr.DataArray(pearsonr_sem, dims=dims, coords=coords),
            "cluster_p": xr.DataArray(cluster_p, dims=dims, coords=coords),
        })

        if self.compute_dimensionality and n_decodable_dims is not None:
            stats["n_decodable_dims"] = xr.DataArray(n_decodable_dims, dims=dims, coords=coords)

        stats.attrs["n_subjects"] = n_subjects
        stats.attrs["n_permutations"] = self.n_permutations

        return stats


def _drop_null_from_score(score: xr.Dataset, n_permutations: int) -> xr.Dataset:
    """Remove null distribution from a score dataset."""
    null_key = _get_null_key(n_permutations)
    if null_key in score:
        return score.drop_vars(null_key)
    return score


def tt_decoding(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    subset: bool = False,
    pca: bool = False,
    reconstruct_with_pc: bool = False,
    subset_method: str = "cumulative",
    model_kwargs: dict = None,
    save_null: bool = False,
    subject: int = None,  # deprecated, kept for backwards compatibility
) -> xr.Dataset:
    
    # TEMP
    cache_str = f"main_analyses/{analysis}_tt_decoding.subset={subset}.pca={pca}"
    if reconstruct_with_pc:
        cache_str += f".reconstruct_with_pc={reconstruct_with_pc}.subset_method={subset_method}"
    cache_path = _cache_path(
        cache_str,
        dataset,
        load_dataset_kwargs,
        scorer_kwargs,
        include_root=False,
    )
    predictor_cache_path = f"{cache_str}/dataset={dataset}"
    predictor_cache_path = _append_path(
        predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs
    )

    match analysis:
        case "behaviorz":
            analysis_fn = _behavior_decoding
            cache_kwargs = {}
            cache_kwargs["subset"] = subset
        case "model_srpz" | "model_srpz_full":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srpz_decoding
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            cache_kwargs["node_subset"] = analysis[-5:]!="_full"
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
        case "model_srp" | "model_srp_full":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srp_decoding
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            cache_kwargs["node_subset"] = analysis[-5:]!="_full"
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
        case "model_srpz_relu":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srpz_decoding
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            cache_kwargs["nodes"] = "ReLU"
            cache_kwargs["node_subset"] = True
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
        # change to z for feature-wise z-score
        case "everythingz":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _everything_decoding
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
            
            cache_kwargs["subset"] = subset
        case "everything_relu":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _everything_decoding
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            cache_kwargs["nodes"] = "ReLU"
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
            
            cache_kwargs["subset"] = subset
        case _:
            raise ValueError(
                "analysis not implemented"
            )

    # Determine if analysis has node dimension
    has_node = analysis in ["model_srpz", "model_srpz_full", "model_srp", "model_srp_full", "model_srpz_relu"]

    # Get number of subjects and target_var
    n_subjects = load_n_subjects(dataset)
    target_var = load_target_var(f"{dataset}_train")

    # Get n_permutations from scorer_kwargs (default 1000)
    n_permutations = scorer_kwargs.get("n_permutations", 1000)

    # Check if stats.nc already exists
    stats_path = f"{cache_path}/stats.nc"
    stats_full_path = BONNER_CACHING_HOME / stats_path
    if stats_full_path.exists():
        logging.info(f"Stats already computed: {stats_path}")
        # Still need to return the scores for backwards compatibility
        all_scores = []
        for s in range(1, n_subjects + 1):
            subject_path = f"{cache_path}/subject={s}.nc"
            subject_full_path = BONNER_CACHING_HOME / subject_path
            if subject_full_path.exists():
                all_scores.append(xr.open_dataset(subject_full_path))
        if all_scores:
            return xr.concat(all_scores, dim="subject")
        return None

    # Process all subjects with incremental stats accumulation
    all_scores = []

    # Use per-node accumulator for has_node analyses to reduce memory
    if has_node:
        stats_accumulator = _PerNodeStatsAccumulator(n_permutations)
    else:
        stats_accumulator = _IncrementalStatsAccumulator(n_permutations, has_node=False)

    for s in tqdm(range(1, n_subjects + 1), desc="subject"):
        logging.info(f"Processing subject={s}")

        data_train = load_dataset(
            f"{dataset}_train",
            subjects=s,
            **load_dataset_kwargs,
        )
        data_test = load_dataset(
            f"{dataset}_test",
            subjects=s,
            **load_dataset_kwargs,
        )

        if 'freq' not in data_train.coords and not reconstruct_with_pc:
            subject_path = f"{cache_path}/subject={s}.nc"
            subject_full_path = BONNER_CACHING_HOME / subject_path

            # For has_node analyses, use per-node generator to save memory
            if has_node:
                # Check if subject file already exists
                if subject_full_path.exists():
                    # Load existing file
                    score = xr.open_dataset(subject_full_path)
                    null_key = _get_null_key(n_permutations)

                    # Check if we need to recompute for n_decodable_dims
                    needs_recompute = "n_decodable_dims" not in score

                    if null_key in score:
                        # Has null, accumulate stats per node
                        for node in score.node.values:
                            node_score = score.sel(node=node)
                            stats_accumulator.add_node_score(node, node_score)

                        # Compute per-subject n_decodable_dims if not present
                        if needs_recompute:
                            dim_target_var = "latent" if scorer_kwargs.get("model_name") == "plssvd" else "neuroid"
                            n_decodable_dims = _compute_n_decodable_dims_per_node(score, n_permutations, target_var=dim_target_var)
                            score["n_decodable_dims"] = n_decodable_dims
                            logging.info(f"Computed per-subject n_decodable_dims for subject={s}")

                            # Re-save with n_decodable_dims
                            score_to_save = _drop_null_from_score(score, n_permutations)
                            cacher = cache(subject_path)
                            cacher(lambda x: x)(score_to_save)
                            logging.info(f"Re-saved subject={s} with n_decodable_dims")

                    all_scores.append(_drop_null_from_score(score, n_permutations))
                    del score
                else:
                    # Compute fresh using per-node generator
                    # Compute n_decodable_dims per node as we iterate (memory efficient)
                    subject_node_scores = []
                    node_n_decodable_dims = {}
                    dim_target_var = "latent" if scorer_kwargs.get("model_name") == "plssvd" else "neuroid"

                    for node, node_score in _model_srpz_decoding_per_node(
                        data_train=data_train,
                        data_test=data_test,
                        subject=s,
                        target_var=target_var,
                        scorer_kwargs=scorer_kwargs,
                        predictor_cache_path=f"{predictor_cache_path}/subject={s}",
                        pca=pca,
                        reconstruct_with_pc=reconstruct_with_pc,
                        **cache_kwargs,
                    ):
                        logging.info(f"  Processing node={node}")
                        # Accumulate stats for this node
                        null_key = _get_null_key(n_permutations)
                        if null_key in node_score:
                            stats_accumulator.add_node_score(node, node_score)

                            # Compute n_decodable_dims for this node immediately
                            node_dims = _compute_n_decodable_dims(node_score, n_permutations, target_var=dim_target_var)
                            node_n_decodable_dims[node] = node_dims.values

                        # Keep score without null
                        subject_node_scores.append(_drop_null_from_score(node_score, n_permutations))
                        del node_score
                        gc.collect()

                    # Combine all nodes for this subject
                    subject_score = xr.concat(subject_node_scores, dim="node")
                    del subject_node_scores

                    # Add n_decodable_dims as (node, time) array
                    if node_n_decodable_dims:
                        nodes = subject_score.node.values
                        time_values = subject_score.time.values
                        n_decodable_dims_arr = np.stack([node_n_decodable_dims[n] for n in nodes], axis=0)
                        subject_score["n_decodable_dims"] = xr.DataArray(
                            n_decodable_dims_arr,
                            dims=("node", "time"),
                            coords={"node": nodes, "time": time_values}
                        )
                        logging.info(f"Computed per-subject n_decodable_dims for subject={s}")

                    # Save per-subject file (with n_decodable_dims)
                    cacher = cache(subject_path)
                    cacher(lambda x: x)(subject_score)
                    logging.info(f"Saved subject={s} to {subject_path}")

                    all_scores.append(subject_score)
                    del subject_score
                    gc.collect()
            else:
                # Non-node analyses (behaviorz, everythingz): use original approach
                if subject_full_path.exists():
                    score = xr.open_dataset(subject_full_path)
                    null_key = _get_null_key(n_permutations)
                    # Check if we need to recompute:
                    # 1. No null and stats don't exist yet
                    # 2. No n_decodable_dims in the file
                    needs_recompute = (
                        (null_key not in score and not (BONNER_CACHING_HOME / stats_path).exists())
                        or "n_decodable_dims" not in score
                    )
                    if needs_recompute:
                        logging.info(f"Recomputing subject={s} to get null for stats/n_decodable_dims...")
                        score = analysis_fn(
                            data_train=data_train,
                            data_test=data_test,
                            subject=s,
                            target_var=target_var,
                            scorer_kwargs=scorer_kwargs,
                            predictor_cache_path=f"{predictor_cache_path}/subject={s}",
                            pca=pca,
                            reconstruct_with_pc=reconstruct_with_pc,
                            **cache_kwargs,
                        )
                else:
                    score = analysis_fn(
                        data_train=data_train,
                        data_test=data_test,
                        subject=s,
                        target_var=target_var,
                        scorer_kwargs=scorer_kwargs,
                        predictor_cache_path=f"{predictor_cache_path}/subject={s}",
                        pca=pca,
                        reconstruct_with_pc=reconstruct_with_pc,
                        **cache_kwargs,
                    )

                null_key = _get_null_key(n_permutations)
                if null_key in score:
                    # Accumulator will compute both stats and pooled n_decodable_dims
                    stats_accumulator.add_subject(score)

                    # Also compute per-subject n_decodable_dims for error bar plotting
                    # Determine target_var based on scorer type
                    if scorer_kwargs.get("model_name") == "plssvd" and scorer_kwargs.get("score_space", "latent") == "latent":
                        dim_target_var = "latent"
                    else:
                        dim_target_var = "neuroid"
                    n_decodable_dims = _compute_n_decodable_dims(score, n_permutations, target_var=dim_target_var)
                    score["n_decodable_dims"] = n_decodable_dims
                    logging.info(f"Computed per-subject n_decodable_dims for subject={s}")

                # Save per-subject file if it doesn't exist or has new n_decodable_dims
                needs_save = (
                    not subject_full_path.exists()
                    or save_null
                    or "n_decodable_dims" in score
                )
                if needs_save:
                    score_to_save = score if save_null else _drop_null_from_score(score, n_permutations)
                    cacher = cache(subject_path)
                    cacher(lambda x: x)(score_to_save)
                    logging.info(f"Saved subject={s} to {subject_path}")

                all_scores.append(_drop_null_from_score(score, n_permutations))
                del score

        elif reconstruct_with_pc:
            raise NotImplementedError("reconstruct_with_pc not yet implemented for multi-subject processing")

        else:
            raise NotImplementedError("freq case not yet implemented for multi-subject processing")

        # Clear memory
        del data_train, data_test
        gc.collect()

    # Compute and save stats from accumulated data
    logging.info("Computing aggregated statistics...")
    stats = stats_accumulator.compute_stats()
    stats_cacher = cache(stats_path)
    stats_cacher(lambda x: x)(stats)
    logging.info(f"Saved stats to {stats_path}")

    # Clear accumulator
    del stats_accumulator
    gc.collect()

    return xr.concat(all_scores, dim="subject")
