import shutil
import logging
import gc
import os

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
from lib.datasets import load_dataset, load_target_var, load_n_subjects, per_image_rep_split
from lib.utilities import (
    _append_path,
    SEED,
)
from lib.datasets.hebart2022_things_behavior import sort_embeddings
from lib.datasets.doerig_rcnn import load_llm_rcnn_embeddings
from lib.models import DJModel

from bonner.computation.decomposition import PCA
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)

# Memo for subject-independent SRP feature maps: (model_uid, dataset, nodes, node_subset,
# str(n_components)) -> target dict. Populated once per run in _model_srpz_decoding_per_node so the
# torchinfo trace + pkl reload don't repeat for every subject.
_MODEL_TARGET_CACHE = {}

# joblib backend for the per-timepoint decode loops: "loky" (default, byte-identical published path)
# or "threading" (Stringer path; single CUDA context). Set by tt_decoding() based on per_image.
_PARALLEL_BACKEND = "loky"


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

def _presentation_split(X: xr.DataArray, n_splits: int = 2, seed: int = 0) -> list[xr.DataArray]:
    """Split a raw dataset along `presentation` into n_splits halves (mirrors
    load_presentation_reshaped_data in datasets/_loader.py: same default_rng(seed), shuffle without
    replacement, npps = n // n_splits). Used for within-test decoding (fit_source='test_split'): the
    two halves are later rep-averaged by _reshape_subject_data (skipna mean over presentation)."""
    rng = np.random.default_rng(seed)
    n = len(X["presentation"].values)
    idx = rng.choice(np.arange(n), size=n, replace=False)
    npps = n // n_splits
    return [X.isel(presentation=idx[i * npps:(i + 1) * npps]) for i in range(n_splits)]


def _running_mean_score(prev, score, count):
    """Incremental running mean of score Datasets (pearsonr + null), to avoid holding 2*n_seeds copies.

    Returns (running_sum, count+1); divide by the final count for the mean. Datasets add elementwise on
    their (identical) coords, so this averages BOTH the observed pearsonr and the permutation null.
    """
    return (score if prev is None else prev + score), count + 1


def _stringer_nonnode(analysis_fn, data_train, data_test, subject, target_var, n_seeds, call_kwargs):
    """Seed-mean decoding score for a non-node analysis under the Stringer per-image split.

    For each of n_seeds random per-image rep-splits, decode BOTH halves (fit on TRAIN, eval on the
    half's rep-average) and average all 2*n_seeds scores -> seed-mean pearsonr + seed-mean null. The
    train fit is seed-invariant, so the per-(target,time) predictor cache is hit after the first eval.
    """
    acc, n = None, 0
    for seed in range(n_seeds):
        for half in per_image_rep_split(data_test, target_var, "presentation", n_splits=2, seed=seed):
            sc = analysis_fn(data_train=data_train, data_test=half, subject=subject, target_var=target_var, **call_kwargs)
            acc, n = _running_mean_score(acc, sc, n)
            del sc
            gc.collect()
    return acc / n


def _run_nonnode(analysis_fn, data_train, data_test, subject, target_var, per_image, n_seeds,
                 scorer_kwargs, predictor_cache_path, pca, reconstruct_with_pc, cache_kwargs):
    """Dispatch a non-node decode: Stringer seed-mean (per_image) or the single full-test decode."""
    call_kwargs = dict(scorer_kwargs=scorer_kwargs, predictor_cache_path=predictor_cache_path,
                       pca=pca, reconstruct_with_pc=reconstruct_with_pc, **cache_kwargs)
    if per_image:
        return _stringer_nonnode(analysis_fn, data_train, data_test, subject, target_var, n_seeds, call_kwargs)
    return analysis_fn(data_train=data_train, data_test=data_test, subject=subject, target_var=target_var, **call_kwargs)


def _stringer_node(gen_fn, data_train, data_test, subject, target_var, n_seeds, call_kwargs):
    """Seed-mean generator for a node analysis (model_srpz): yields (node, seed-mean node_score).

    Runs the per-node generator over both halves x n_seeds, accumulating a running mean per node, then
    yields each node's seed-mean score (drop-in for the raw per-node generator in tt_decoding's loop).
    """
    from collections import OrderedDict
    node_acc, node_n, node_order = {}, {}, []
    for seed in range(n_seeds):
        for half in per_image_rep_split(data_test, target_var, "presentation", n_splits=2, seed=seed):
            for node, ns in gen_fn(data_train=data_train, data_test=half, subject=subject, target_var=target_var, **call_kwargs):
                if node not in node_acc:
                    node_order.append(node)
                node_acc[node], node_n[node] = _running_mean_score(node_acc.get(node), ns, node_n.get(node, 0))
                del ns
            gc.collect()
    for node in node_order:
        yield node, node_acc[node] / node_n[node]


def _residualize_behavior(subject_data_train, subject_data_test, l2):
    """Regress THINGS behavior out of the neural signal at EVERY timepoint.

    Fits a linear ridge `behavior -> neural(t)` on TRAIN and subtracts the prediction from BOTH train
    and test (train-fit weights), per timepoint. Mirrors `_behavior_remaining_var` (remaining_var.py),
    generalized to loop over all timepoints. Uses RAW (non-z-scored) behavior + neural, matching the
    residual definition there. `subject_data_*` are `(presentation, neuroid, time)`; the returned
    residuals keep the same dims/coords (incl. `img_files`, `presentation`=object).
    """
    scorer = TrainTestModelScorer(
        model_name="linear", l2_penalty=l2, n_permutations=None, cache_predictors=False,
    )

    def _2t(x):
        return torch.tensor(x.values).float().to(scorer.device)

    embd = load_embeddings()
    beh_train = _2t(embd.sel(object=subject_data_train.presentation.values))
    beh_test = _2t(embd.sel(object=subject_data_test.presentation.values))

    res_train, res_test = [], []
    for t in subject_data_train.time.values:
        y_train = _2t(subject_data_train.sel(time=t))
        model = scorer._fit_model(x_train=beh_train, y_train=y_train)
        res_train.append((y_train - model.predict(beh_train)).cpu().numpy())
        y_test = _2t(subject_data_test.sel(time=t))
        res_test.append((y_test - model.predict(beh_test)).cpu().numpy())

    train_res = subject_data_train.copy(data=np.stack(res_train, axis=-1))
    test_res = subject_data_test.copy(data=np.stack(res_test, axis=-1))
    return train_res, test_res

def _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=None, regress_out=None, regress_out_l2=1e-2):
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
    if regress_out == "behavior":
        assert average, "regress_out=behavior requires average=True"
        subject_data_train, subject_data_test = _residualize_behavior(
            subject_data_train, subject_data_test, regress_out_l2
        )
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
    regress_out: str = None,
):
    """Generator that yields one node's score at a time to reduce memory usage."""
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=num_pc, regress_out=regress_out)

    # The SRP feature maps are subject-independent, but this generator is called once per subject in
    # tt_decoding's loop -> memoize `model(dataset)` so the torchinfo trace + pkl reload run only once
    # per (model, dataset, nodes, node_subset, n_components). Safe: each feature_map is deepcopy'd
    # before z-scoring below, so the cached target is never mutated.
    _mkey = (model_uid, dataset, nodes, node_subset, str(n_components))
    if _mkey not in _MODEL_TARGET_CACHE:
        model = DJModel(model_uid, hook="srp", n_components=n_components, nodes=nodes, node_subset=node_subset)
        _MODEL_TARGET_CACHE[_mkey] = model(dataset, dataloader_kwargs={"batch_size": 32})
    target = _MODEL_TARGET_CACHE[_mkey]

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

        time_scores = Parallel(n_jobs=3, backend=_PARALLEL_BACKEND)(
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
            
        time_scores = Parallel(n_jobs=3, backend=_PARALLEL_BACKEND)(
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
    
    time_scores = Parallel(n_jobs=(4 if _PARALLEL_BACKEND == "threading" else -1), backend=_PARALLEL_BACKEND)(
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

def _llm_rcnn_decoding(
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    pca: bool,
    reconstruct_with_pc: bool,
    dataset: str,
    num_pc: int = None,
    regress_out: str = None,
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=num_pc, regress_out=regress_out)

    # Doerig LLM-trained RCNN feature: pre-readout last conv layer / last timestep, 512-d PER IMAGE.
    # Join by img_files (per-image), mirroring the model-node pattern in _everything_decoding.
    embd = load_llm_rcnn_embeddings(dataset)
    target_train = deepcopy(embd.sel(presentation=embd.img_files.isin(subject_data_train.img_files))).sortby("img_files")
    assert np.all(target_train.img_files.values == subject_data_train.img_files.values)
    target_test = deepcopy(embd.sel(presentation=embd.img_files.isin(subject_data_test.img_files))).sortby("img_files")
    assert np.all(target_test.img_files.values == subject_data_test.img_files.values)

    # guard against any ~constant feature dims: 0 std -> divide-by-0 in the z-score -> Inf ->
    # PLSSVD SVD fails to converge. Drop them (kept from the MPNet path as a safety net).
    keep = (target_train.std("presentation") > 1e-6) & (target_test.std("presentation") > 1e-6)
    target_train = target_train.isel(neuroid=keep.values)
    target_test = target_test.isel(neuroid=keep.values)

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

    time_scores = Parallel(n_jobs=(4 if _PARALLEL_BACKEND == "threading" else -1), backend=_PARALLEL_BACKEND)(
        delayed(scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=llm_rcnn=all/predictor=time={t}",
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
    include_llm_rcnn: bool = False,
    all_layer: bool = False,
    n_layers: int = 7,
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True, num_pc=num_pc)

    target_train, target_test = dict(), dict()
    if subset:
        embd = sort_embeddings()
    else:
        embd = load_embeddings()

    target_train["behavior"] = embd.sel(object=subject_data_train.presentation.values).rename({"object": "presentation", "behavior": "neuroid"}).values
    target_test["behavior"] = embd.sel(object=subject_data_test.presentation.values).rename({"object": "presentation", "behavior": "neuroid"}).values

    if include_llm_rcnn:
        # Doerig RCNN feature (512-d per image); join by img_files like the model nodes below.
        llm_embd = load_llm_rcnn_embeddings(dataset)
        llm_train = deepcopy(llm_embd.sel(presentation=llm_embd.img_files.isin(subject_data_train.img_files))).sortby("img_files")
        assert np.all(llm_train.img_files.values == subject_data_train.img_files.values)
        llm_test = deepcopy(llm_embd.sel(presentation=llm_embd.img_files.isin(subject_data_test.img_files))).sortby("img_files")
        assert np.all(llm_test.img_files.values == subject_data_test.img_files.values)
        # guard against any ~constant dims (would divide-by-~0 in the joint z-score -> Inf)
        keep = (llm_train.std("presentation") > 1e-6) & (llm_test.std("presentation") > 1e-6)
        target_train["llm_rcnn"] = llm_train.isel(neuroid=keep.values).values
        target_test["llm_rcnn"] = llm_test.isel(neuroid=keep.values).values

    # all_layer=False -> node_subset=True extracts only first+last (legacy everythingz);
    # all_layer=True  -> node_subset=False extracts all Identity nodes (reuses the model_srpz_full
    # SRP cache, same hash), then keep_idx subsamples n_layers evenly across depth below.
    # Memoize the subject-independent SRP maps: under Stringer this fn runs 2*n_seeds times per subject,
    # so without the memo the torchinfo trace + pkl reload would repeat ~20x. Each feature_map is
    # deepcopy'd below before use, so the cached target is never mutated.
    _mkey = (model_uid, dataset, nodes, not all_layer, str(n_components))
    if _mkey not in _MODEL_TARGET_CACHE:
        model = DJModel(model_uid, hook="srp", n_components=n_components, nodes=nodes, node_subset=not all_layer)
        _MODEL_TARGET_CACHE[_mkey] = model(dataset, dataloader_kwargs={"batch_size": 32})
    model_target = _MODEL_TARGET_CACHE[_mkey]

    n_total = len(model_target)
    if all_layer:
        keep_idx = set(np.unique(np.round(np.linspace(1, n_total, n_layers)).astype(int)).tolist())
        assert len(keep_idx) == n_layers and 1 in keep_idx and n_total in keep_idx, (
            f"expected {n_layers} evenly-spaced layers incl. first & last, got {sorted(keep_idx)} "
            f"of n_total={n_total}")
    else:
        keep_idx = {1, n_total}

    i = 0
    for node, feature_map in model_target.items():
        i+=1
        if i not in keep_idx:
            continue
        target_train[node] = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_train.img_files))).sortby("img_files")
        assert np.all(target_train[node].img_files.values==subject_data_train.img_files.values)
        target_train[node] = target_train[node].values
        target_test[node] = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_test.img_files))).sortby("img_files").values
    del model_target
    
    # z-score
    # float32: behavior embeddings are float64 and would upcast the whole concat, doubling GPU
    # memory (OOM with many layers); SRP maps are already float32 and PLSSVD casts to .float().
    temp =  torch.tensor(np.concatenate([v for v in target_train.values()], axis=1), device="cuda", dtype=torch.float32)
    mean = temp.mean(dim=0)
    std = temp.std(dim=0)
    target_train = xr.DataArray(
        ((temp-mean)/std).cpu().numpy(),
        dims=["presentation", "neuroid"],
    )
    temp =  torch.tensor(np.concatenate([v for v in target_test.values()], axis=1), device="cuda", dtype=torch.float32)
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
    
    time_scores = Parallel(n_jobs=1, backend=_PARALLEL_BACKEND)(
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
    regress_out: str = None,
    fit_source: str = "train",
    test_split_half: int = None,
    per_image: bool = False,
    n_seeds: int = 1,
) -> xr.Dataset:

    # fit_source: "train" (default) fits the decoder on the TRAIN images and evaluates on the TEST images;
    # "test_split" fits on one TEST presentation-half and evaluates on the other (within-test decoding).
    assert fit_source in ("train", "test_split"), fit_source
    if fit_source == "test_split":
        assert not reconstruct_with_pc and regress_out is None, (
            "fit_source='test_split' does not support reconstruct_with_pc / regress_out"
        )

    # test_split_half: restrict the TEST set to one presentation(rep)-half before the rep-average, to
    # SNR-match ttpca's cvPCA dimensionality (which averages ~half the reps per split). Train side is
    # untouched (full). Only meaningful for the standard train-fit decoding path.
    assert test_split_half in (None, 0, 1), test_split_half
    if test_split_half is not None:
        assert fit_source == "train", "test_split_half is incompatible with fit_source='test_split'"

    # per_image (Stringer): fit on TRAIN (seed-invariant), evaluate on BOTH per-image rep-halves of the
    # TEST set, average the two scores, over n_seeds random splits -> seed-mean score, then the existing
    # n_decodable_dims / stats path runs ONCE. Requires the seed-invariant train fit, so it is mutually
    # exclusive with the within-test / half-split flags (which change the fit set per seed and would
    # collide on seed-0's cached predictor).
    if per_image:
        assert fit_source == "train", "per_image requires fit_source='train' (seed-invariant fit reused across seeds)"
        assert test_split_half is None, "per_image is incompatible with test_split_half (both restrict the test reps)"
    # loky spawns one CUDA context per worker (context exhaustion on many-core boxes); the per-timepoint
    # work is GPU-bound, so threading (single context, GIL released during CUDA ops) is value-identical
    # and drops the LOKY_MAX_CPU_COUNT hack. Gate on per_image so the default path stays loky (byte-id).
    global _PARALLEL_BACKEND
    _PARALLEL_BACKEND = "threading" if per_image else "loky"
    if per_image and os.environ.get("TDIM_STRINGER_FP32", "1") == "1":
        # Track-B TF32 matmul on the Stringer path. The everything target is already float32
        # (see _everything_decoding), so this is a small matmul-precision win, not an fp64->fp32 switch.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if regress_out is not None:
        # Guard against writing a cache dir tagged `.regress_out=...` that was never residualized:
        # only these two modes set `cache_kwargs["regress_out"]` below and route through the
        # residualizing `_subject_data` path.
        assert analysis in {"llm_rcnnz", "model_srpz"}, (
            f"regress_out={regress_out} only supported for llm_rcnnz / model_srpz, not {analysis}"
        )
        assert not pca and not reconstruct_with_pc, (
            "regress_out requires pca=False and reconstruct_with_pc=False"
        )

    # TEMP
    cache_str = f"main_analyses/{analysis}_tt_decoding.subset={subset}.pca={pca}"
    if reconstruct_with_pc:
        cache_str += f".reconstruct_with_pc={reconstruct_with_pc}.subset_method={subset_method}"
    if regress_out is not None:
        cache_str += f".regress_out={regress_out}"
    # tag within-test runs so they don't clobber the published train caches (default "train" adds no tag);
    # cache_str feeds BOTH cache_path and predictor_cache_path below.
    if fit_source != "train":
        cache_str += f".fit_source={fit_source}"
    # tag half-split-test runs (default None adds no tag -> published full-test caches byte-identical);
    # cache_str feeds BOTH cache_path and predictor_cache_path, so predictors don't collide either.
    if test_split_half is not None:
        cache_str += f".test_split_half={test_split_half}"
    # tag Stringer per-image runs (default False adds no tag -> published caches byte-identical). Feeds
    # both cache_path and predictor_cache_path; n_seeds is constant per run and NOT part of the per-
    # (target,time) predictor key, so the seed-invariant train fit is still reused across all seeds/halves.
    if per_image:
        cache_str += f".split_mode=stringer.n_seeds={n_seeds}"
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
        case "llm_rcnnz":
            analysis_fn = _llm_rcnn_decoding
            # dataset is needed to load the per-image RCNN feature; it carries no cache-path segment
            # (like behaviorz, this single-feature mode has no model_kwargs).
            cache_kwargs = {"dataset": dataset}
            cache_kwargs["regress_out"] = regress_out
        case "model_srpz" | "model_srpz_full":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srpz_decoding

            model_kwargs["n_components"] = "auto"

            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            cache_kwargs["node_subset"] = analysis[-5:]!="_full"
            cache_kwargs["regress_out"] = regress_out

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
        case "everything_llm_rcnnz_all_layer":
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
            # "everything" now means the 7-layer definition (behavior + 7 evenly-spaced model layers,
            # matching everythingz_all_layer / revision fig_4), plus the RCNN feature. all_layer,
            # n_layers, include_llm_rcnn are forwarded to _everything_decoding only (not in
            # model_kwargs), so they don't enter any cache-path segment.
            cache_kwargs["all_layer"] = True
            cache_kwargs["n_layers"] = 7
            cache_kwargs["include_llm_rcnn"] = True
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
        case "everythingz_all_layer":
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
            # all_layer/n_layers forwarded to _everything_decoding only (not in model_kwargs),
            # so they don't enter any cache-path segment; the analysis name already gives this
            # mode its own cache dir.
            cache_kwargs["all_layer"] = True
            cache_kwargs["n_layers"] = 7
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
    stats_subjects = []  # subjects whose null reached the accumulator, for the partial-stats guard

    # Use per-node accumulator for has_node analyses to reduce memory
    if has_node:
        stats_accumulator = _PerNodeStatsAccumulator(n_permutations)
    else:
        stats_accumulator = _IncrementalStatsAccumulator(n_permutations, has_node=False)

    for s in tqdm(range(1, n_subjects + 1), desc="subject"):
        logging.info(f"Processing subject={s}")

        if fit_source == "test_split":
            # within-test: fit on one TEST presentation-half, evaluate on the other (same 200 images).
            data_test_full = load_dataset(
                f"{dataset}_test",
                subjects=s,
                **load_dataset_kwargs,
            )
            data_train, data_test = _presentation_split(data_test_full, n_splits=2, seed=0)
            del data_test_full
        else:
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
            if test_split_half is not None:
                # keep only one rep-half of the test set (seed=0, npps=n_reps//2, same split as ttpca);
                # the downstream .mean("presentation") then averages ~half the reps -> SNR-matched.
                data_test = _presentation_split(data_test, n_splits=2, seed=0)[test_split_half]

        if 'freq' not in data_train.coords and not reconstruct_with_pc:
            subject_path = f"{cache_path}/subject={s}.nc"
            subject_full_path = BONNER_CACHING_HOME / subject_path

            # For has_node analyses, use per-node generator to save memory
            if has_node:
                null_key = _get_null_key(n_permutations)
                score = xr.open_dataset(subject_full_path) if subject_full_path.exists() else None
                if score is not None and n_permutations is not None and null_key not in score:
                    # stats.nc is being built (an existing one returns early above), and a cached
                    # subject without its null cannot reach the accumulator: recompute it, or
                    # stats.nc would silently cover fewer subjects than the per-subject files.
                    logging.info(f"Recomputing subject={s}: cached file has no null")
                    score.close()
                    score = None

                if score is not None:
                    # Check if we need to recompute for n_decodable_dims
                    needs_recompute = "n_decodable_dims" not in score

                    if null_key in score:
                        # Has null, accumulate stats per node
                        for node in score.node.values:
                            node_score = score.sel(node=node)
                            stats_accumulator.add_node_score(node, node_score)
                        stats_subjects.append(s)

                        # Compute per-subject n_decodable_dims if not present
                        if needs_recompute:
                            dim_target_var = "latent" if scorer_kwargs.get("model_name") == "plssvd" else "neuroid"
                            n_decodable_dims = _compute_n_decodable_dims_per_node(score, n_permutations, target_var=dim_target_var)
                            score["n_decodable_dims"] = n_decodable_dims
                            logging.info(f"Computed per-subject n_decodable_dims for subject={s}")

                            # Re-save with n_decodable_dims. The default cacher mode loads an existing
                            # file instead of writing, so overwrite, with the read handle released first.
                            score = score.load()
                            score.close()
                            score_to_save = _drop_null_from_score(score, n_permutations)
                            cacher = cache(subject_path, mode="overwrite")
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

                    _gen_kwargs = dict(
                        scorer_kwargs=scorer_kwargs,
                        predictor_cache_path=f"{predictor_cache_path}/subject={s}",
                        pca=pca,
                        reconstruct_with_pc=reconstruct_with_pc,
                        **cache_kwargs,
                    )
                    _node_gen = (
                        _stringer_node(_model_srpz_decoding_per_node, data_train, data_test, s, target_var, n_seeds, _gen_kwargs)
                        if per_image else
                        _model_srpz_decoding_per_node(data_train=data_train, data_test=data_test, subject=s, target_var=target_var, **_gen_kwargs)
                    )
                    for node, node_score in _node_gen:
                        logging.info(f"  Processing node={node}")
                        # Accumulate stats for this node
                        if null_key in node_score:
                            stats_accumulator.add_node_score(node, node_score)
                            if not stats_subjects or stats_subjects[-1] != s:
                                stats_subjects.append(s)

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

                    # Save per-subject file (with n_decodable_dims). Overwrite: this branch also runs
                    # for a cached file being recomputed, which the default cacher mode would keep.
                    cacher = cache(subject_path, mode="overwrite")
                    cacher(lambda x: x)(subject_score)
                    logging.info(f"Saved subject={s} to {subject_path}")

                    all_scores.append(subject_score)
                    del subject_score
                    gc.collect()
            else:
                # Non-node analyses (behaviorz, everythingz): use original approach
                recomputed = False
                if subject_full_path.exists():
                    score = xr.open_dataset(subject_full_path)
                    null_key = _get_null_key(n_permutations)
                    # Check if we need to recompute:
                    # 1. No null and stats don't exist yet
                    # 2. No n_decodable_dims in the file
                    # (n_permutations=None runs intentionally have no null/n_decodable_dims -> an
                    # existing subject file is complete; never re-decode chasing a null that won't come.)
                    needs_recompute = n_permutations is not None and (
                        (null_key not in score and not (BONNER_CACHING_HOME / stats_path).exists())
                        or "n_decodable_dims" not in score
                    )
                    if needs_recompute:
                        logging.info(f"Recomputing subject={s} to get null for stats/n_decodable_dims...")
                        score.close()
                        score = _run_nonnode(analysis_fn, data_train, data_test, s, target_var, per_image, n_seeds,
                                             scorer_kwargs, f"{predictor_cache_path}/subject={s}", pca, reconstruct_with_pc, cache_kwargs)
                        recomputed = True
                else:
                    score = _run_nonnode(analysis_fn, data_train, data_test, s, target_var, per_image, n_seeds,
                                         scorer_kwargs, f"{predictor_cache_path}/subject={s}", pca, reconstruct_with_pc, cache_kwargs)

                null_key = _get_null_key(n_permutations)
                if null_key in score:
                    # Accumulator will compute both stats and pooled n_decodable_dims
                    stats_accumulator.add_subject(score)
                    stats_subjects.append(s)

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
                    # a recomputed subject replaces its cached file; the default cacher mode would keep it
                    cacher = cache(subject_path, mode="overwrite" if recomputed else "normal")
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

    # Compute and save stats from accumulated data -- but only if the accumulator actually received
    # data. With n_permutations=None there is no null, so nothing was accumulated and compute_stats()
    # would raise ("No subjects/nodes added"). (Branch on has_node: _PerNodeStatsAccumulator has no
    # .n_subjects attribute.)
    has_stats_data = bool(stats_accumulator.node_accumulators) if has_node else stats_accumulator.n_subjects > 0
    if has_stats_data:
        if has_node:
            counts = {node: acc["n_subjects"] for node, acc in stats_accumulator.node_accumulators.items()}
        else:
            counts = {"all": stats_accumulator.n_subjects}
        if any(count != n_subjects for count in counts.values()):
            raise RuntimeError(
                f"refusing to write a partial {stats_path}: accumulated subject counts {counts} != "
                f"n_subjects={n_subjects}; subjects that reached the accumulator: {stats_subjects}"
            )
        logging.info("Computing aggregated statistics...")
        stats = stats_accumulator.compute_stats()
        stats_cacher = cache(stats_path)
        stats_cacher(lambda x: x)(stats)
        logging.info(f"Saved stats to {stats_path}")
    else:
        logging.info("No null distribution accumulated -> skipping stats.nc (n_permutations=None)")

    # Clear accumulator
    del stats_accumulator
    gc.collect()

    return xr.concat(all_scores, dim="subject")
