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

from lib.models import DJModel
from lib.analyses._utilities import _cache_path
from lib.computation.scorers import TrainTestModelGeneralizationScorer
from lib.computation.statistics import cluster_correction
from lib.datasets import load_dataset, load_target_var, load_n_subjects
from lib.utilities import (
    _append_path,
    SEED,
)
from lib.datasets.hebart2022_things_behavior import sort_embeddings

from bonner.computation.decomposition import PCA
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)


def _reshape_subject_data(data, subject, target_var, average=True) -> xr.DataArray:
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
    

def _reconstruct_data_with_pc_subject(
    subset_method: str,
    pc_timepoint: float,
    target_var: str,
    data_train: xr.DataArray,
    data_test: xr.DataArray,
):
    new_train, new_test = [], []
    for subject in data_train.subject.values:
        subject_new_train, subject_new_test = [], []
        
        pmodel = PCA()
        pmodel.fit(torch.tensor(data_train.sel(subject=subject, time=pc_timepoint).mean("presentation").values))
        
        x_train = _reshape_subject_data(data_train, subject, target_var).transpose("time", "presentation", "neuroid")
        x_test = _reshape_subject_data(data_test, subject, target_var).transpose("time", "presentation", "neuroid")
        
        z_train_stack = pmodel.transform(torch.tensor(
            x_train.stack(stack=["time", "presentation"]).transpose("stack", "neuroid").values
        ))
        z_test_stack = pmodel.transform(torch.tensor(
            x_test.stack(stack=["time", "presentation"]).transpose("stack", "neuroid").values
        ))
        for i in range(z_train_stack.size(-1)):
            temp_train, temp_test = torch.zeros_like(z_train_stack), torch.zeros_like(z_test_stack)
            match subset_method:
                case "single":
                    temp_train[:,i] = z_train_stack[:,i]
                    temp_test[:,i] = z_test_stack[:,i]
                case "cumulative":
                    temp_train[:,:i+1] = z_train_stack[:,:i+1]
                    temp_test[:,:i+1] = z_test_stack[:,:i+1]
                case _:
                    raise ValueError(
                        "subset method not implemented"
                    )
            subject_new_train.append(x_train.copy(data=pmodel.inverse_transform(temp_train).reshape([x_train.sizes["time"], x_train.sizes["presentation"], x_train.sizes["neuroid"]])).assign_coords({"pc": i+1}))
            subject_new_test.append(x_test.copy(data=pmodel.inverse_transform(temp_test).reshape([x_test.sizes["time"], x_test.sizes["presentation"], x_test.sizes["neuroid"]])).assign_coords({"pc": i+1}))
        new_train.append(xr.concat(subject_new_train, dim="pc").assign_coords({"subject": subject}))
        new_test.append(xr.concat(subject_new_test, dim="pc").assign_coords({"subject": subject}))
    return xr.concat(new_train, dim="subject"), xr.concat(new_test, dim="subject") 

def _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=False):
    if reconstruct_with_pc:
        subject_data_train = data_train.sel(subject=subject)
        subject_data_test = data_test.sel(subject=subject)
    else:
        subject_data_train = _reshape_subject_data(data_train, subject, target_var, average)
        subject_data_test = _reshape_subject_data(data_test, subject, target_var, average)
    return subject_data_train, subject_data_test

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
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True)

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
        pmodel = PCA()
        pmodel.fit(torch.tensor(target_train.values))
        target_train = target_train.copy(data=pmodel.transform(torch.tensor(target_train.values)))
        target_test = target_test.copy(data=pmodel.transform(torch.tensor(target_test.values)))
        target_train["behavior"] = np.arange(len(target_train.behavior))+1
        target_test["behavior"] = np.arange(len(target_test.behavior))+1
    
    assert scorer_kwargs["model_name"] != "plssvd"
    scorer_fn = TrainTestModelGeneralizationScorer
    
    return scorer_fn(
        **scorer_kwargs,
        cache_predictors=True,
        cache_subpath=f"{predictor_cache_path}/target=behavior=all",
    )(
        predictor_train=subject_data_train,
        target_train=target_train,
        predictor_test=subject_data_test,
        target_test=target_test,
        target_dim="neuroid",
        predictor_dim="time",
    ).assign_coords({"subject": subject})
    
def _model_srp_decoding(
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
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True)
    
    model = DJModel(model_uid, hook="srp", n_components=n_components)
    target = model(dataset, dataloader_kwargs={"batch_size": 32})
    
    assert scorer_kwargs["model_name"] != "plssvd"
    scorer_fn = TrainTestModelGeneralizationScorer

    scores = []
    for node, feature_map in target.items():
        feature_map_train = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_train.img_files)))
        feature_map_test = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_test.img_files)))
    
        if pca:
            pmodel = PCA()
            pmodel.fit(torch.tensor(feature_map_train.values))
            feature_map_train = feature_map_train.copy(data=pmodel.transform(torch.tensor(feature_map_train.values)))
            feature_map_test = feature_map_test.copy(data=pmodel.transform(torch.tensor(feature_map_test.values)))
            feature_map_train["neuroid"] = np.arange(len(feature_map_train.neuroid))+1
            feature_map_test["neuroid"] = np.arange(len(feature_map_test.neuroid))+1
    
        scores.append(scorer_fn(
                **scorer_kwargs,
                cache_predictors=True,
                cache_subpath=f"{predictor_cache_path}/target=node={node}",
            )(
                predictor_train=subject_data_train,
                target_train=feature_map_train,
                predictor_test=subject_data_test,
                target_test=feature_map_test,
                target_dim="neuroid",
                predictor_dim="time",
            ).assign_coords({"subject": subject})
        )
    return xr.concat(scores, dim="node").assign_coords({"subject": subject})

def _get_null_key(n_permutations: int) -> str:
    """Get the null variable name for a given number of permutations."""
    return f"null.pearsonr.n_permutations={n_permutations}"


class _IncrementalStatsAccumulatorGeneralization:
    """
    Incrementally accumulate statistics for computing cluster-corrected p-values
    for temporal generalization analyses.

    Uses running sums to compute mean without storing all data in memory.
    """
    def __init__(self, n_permutations: int, has_node: bool = False):
        self.n_permutations = n_permutations
        self.has_node = has_node
        self.null_key = _get_null_key(n_permutations)

        self.n_subjects = 0
        self.pearsonr_sum = None
        self.pearsonr_sq_sum = None  # For computing variance
        self.null_sum = None
        self.time_values = None
        self.time_gen_values = None
        self.target_var = None  # Will be inferred from first score

    def add_subject(self, score: xr.Dataset) -> None:
        """Add a subject's data to the running statistics."""
        # Infer target_var from score dimensions (exclude 'time', 'time_generalization', 'node')
        if self.target_var is None:
            score_dims = set(score.pearsonr.dims)
            non_target_dims = {"time", "time_generalization", "node"}
            target_dims = score_dims - non_target_dims
            if len(target_dims) == 1:
                self.target_var = target_dims.pop()
            else:
                raise ValueError(f"Cannot infer target_var from dimensions {score_dims}")

        # Average over target_var for significance testing
        pearsonr_agg = score.pearsonr.mean(self.target_var).values

        # Handle null distribution shape for generalization
        null_data = score[self.null_key].mean(self.target_var)
        # Shape: (time, time_generalization, permutation) -> (permutation, time, time_generalization)
        null_agg = null_data.transpose("permutation", "time", "time_generalization").values

        # Initialize accumulators on first subject
        if self.n_subjects == 0:
            self.pearsonr_sum = np.zeros_like(pearsonr_agg)
            self.pearsonr_sq_sum = np.zeros_like(pearsonr_agg)
            self.null_sum = np.zeros_like(null_agg)
            self.time_values = score.time.values
            self.time_gen_values = score.time_generalization.values

        # Update running sums
        self.pearsonr_sum += pearsonr_agg
        self.pearsonr_sq_sum += pearsonr_agg ** 2
        self.null_sum += null_agg
        self.n_subjects += 1

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

        # Compute cluster-corrected p-values per row (across time_generalization)
        cluster_p = np.zeros_like(pearsonr_mean)
        cluster_p_neg = np.zeros_like(pearsonr_mean)
        for i in range(pearsonr_mean.shape[0]):  # iterate over time
            # null_mean[:, i] has shape (permutation, time_generalization)
            cluster_p[i] = cluster_correction(pearsonr_mean[i], null_mean[:, i])
            cluster_p_neg[i] = cluster_correction(-pearsonr_mean[i], -null_mean[:, i])

        dims = ("time", "time_generalization")
        coords = {"time": self.time_values, "time_generalization": self.time_gen_values}

        # Create stats dataset
        stats = xr.Dataset({
            "pearsonr_mean": xr.DataArray(pearsonr_mean, dims=dims, coords=coords),
            "pearsonr_sem": xr.DataArray(pearsonr_sem, dims=dims, coords=coords),
            "cluster_p": xr.DataArray(cluster_p, dims=dims, coords=coords),
            "cluster_p_neg": xr.DataArray(cluster_p_neg, dims=dims, coords=coords),
        })

        stats.attrs["n_subjects"] = self.n_subjects
        stats.attrs["n_permutations"] = self.n_permutations

        return stats


def _drop_null_from_score(score: xr.Dataset, n_permutations: int) -> xr.Dataset:
    """Remove null distribution from a score dataset."""
    null_key = _get_null_key(n_permutations)
    if null_key in score:
        return score.drop_vars(null_key)
    return score


def tt_decoding_generalization(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    subset: bool = False,
    pca: bool = False,
    reconstruct_with_pc: bool = False,
    subset_method: str = "single",
    pc_timepoint: float = .37,
    model_kwargs: dict = None,
    save_null: bool = False,
    subject: int = None,  # deprecated, kept for backwards compatibility
) -> xr.Dataset:
    cache_str = f"main_analyses/{analysis}_tt_decoding_generalization.subset={subset}.pca={pca}"
    if reconstruct_with_pc:
        cache_str += f".reconstruct_with_pc={reconstruct_with_pc}.subset_method={subset_method}.pc_timepoint={pc_timepoint}"
    cache_path = _cache_path(
        cache_str,
        dataset,
        load_dataset_kwargs,
        scorer_kwargs,
        include_root=False,
    )
    predictor_cache_path = f"main_analyses/{analysis}_tt_decoding.subset={subset}.pca={pca}/dataset={dataset}"
    predictor_cache_path = _append_path(
        predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs
    )

    match analysis:
        case "behaviorz":
            analysis_fn = _behavior_decoding
            cache_kwargs = {}
        case "model_srp":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srp_decoding
            
            # TEST: MEM SIZE ISSUE
            # model_kwargs["n_components"] = "auto"
            model_kwargs["n_components"] = 1000
            
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
        case _:
            raise ValueError(
                "analysis not implemented"
            )

    # Determine if analysis has node dimension
    has_node = analysis in ["model_srp"]

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
    stats_accumulator = _IncrementalStatsAccumulatorGeneralization(n_permutations, has_node=has_node)

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

        if reconstruct_with_pc:
            data_train, data_test = _reconstruct_data_with_pc_subject(subset_method, pc_timepoint, target_var, data_train, data_test)

        if 'freq' not in data_train.coords and not reconstruct_with_pc:
            subject_path = f"{cache_path}/subject={s}.nc"

            # Check if subject file exists (with or without null)
            subject_full_path = BONNER_CACHING_HOME / subject_path
            if subject_full_path.exists():
                # Load existing file
                score = xr.open_dataset(subject_full_path)
                # Check if it has null - if not, we need to recompute for stats
                null_key = _get_null_key(n_permutations)
                if null_key not in score and not (BONNER_CACHING_HOME / stats_path).exists():
                    # Need to recompute with null for stats
                    logging.info(f"Recomputing subject={s} to get null for stats...")
                    score = analysis_fn(
                        data_train=data_train,
                        data_test=data_test,
                        subject=s,
                        target_var=target_var,
                        scorer_kwargs=scorer_kwargs,
                        predictor_cache_path=f"{predictor_cache_path}/subject={s}",
                        subset=subset,
                        pca=pca,
                        reconstruct_with_pc=reconstruct_with_pc,
                        **cache_kwargs,
                    )
            else:
                # Compute fresh
                score = analysis_fn(
                    data_train=data_train,
                    data_test=data_test,
                    subject=s,
                    target_var=target_var,
                    scorer_kwargs=scorer_kwargs,
                    predictor_cache_path=f"{predictor_cache_path}/subject={s}",
                    subset=subset,
                    pca=pca,
                    reconstruct_with_pc=reconstruct_with_pc,
                    **cache_kwargs,
                )

            # Incrementally accumulate stats (only needs null temporarily)
            null_key = _get_null_key(n_permutations)
            if null_key in score:
                stats_accumulator.add_subject(score)

            # Save per-subject file (without null unless save_null=True)
            if not subject_full_path.exists() or save_null:
                score_to_save = score if save_null else _drop_null_from_score(score, n_permutations)
                cacher = cache(subject_path)
                cacher(lambda x: x)(score_to_save)
                logging.info(f"Saved subject={s} to {subject_path}")

            # Keep score without null for return value
            all_scores.append(_drop_null_from_score(score, n_permutations))

            # Clear this subject's data from memory immediately
            del score

        elif reconstruct_with_pc:
            # TODO: implement reconstruct_with_pc case
            raise NotImplementedError("reconstruct_with_pc not yet implemented for multi-subject processing")

        else:
            # freq case - TODO: implement
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
