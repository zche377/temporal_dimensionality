import shutil
import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import torch
from joblib import Parallel, delayed
import time

from lib.analyses._utilities import _cache_path
from lib.datasets import (
    load_dataset,
    load_target_var,
    load_presentation_reshaped_data
)
from lib.datasets.hebart2022_things_behavior import sort_embeddings
from lib.analyses.loaders import load_significant_times
from lib.computation.scorers import ModelScorer
from lib.datasets import load_n_subjects
from lib.utilities import (
    _append_path,
    SEED,
)
from bonner.computation.decomposition import PCA

from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)
from bonner.datasets.hebart2022_things_behavior import load_embeddings

N_JOBS = 6
MAX_RETRIES = 10


def _subjects(subjects, dataset) -> list[int]:
    if subjects == "all" or subjects == "intersection" or subjects == "group":
        subjects = list(range(1, load_n_subjects(dataset) + 1))
    if isinstance(subjects, int):
        subjects = [subjects]
    return subjects

def _load_weights(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    significant_only: bool,
    target_var: str,
    feature_dim: str,
    feature_coords: list[str],
    neuroid_coords: list[str],
    times: xr.DataArray,
    subjects: (int | list[int] | str),
) -> xr.DataArray:
    predictor_cache_path = f"main_analyses/{analysis}_decoding/dataset={dataset}"
    predictor_cache_path = _append_path(predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    
    weights = []
    
    match analysis:
        case "multiclass":
            target_id = f"{target_var}=multiclass"
        case "behavior":
            target_id = "behavior=all"
        case "bhvsubset":
            target_id = "behavior=sortby=std.ntop=8"
    
    for subject in _subjects(subjects, dataset):
        # try:
        subject_weights = []
        subject_times = times.sel(subject=subject).dropna("time").values if significant_only else times
        for time in subject_times:
            scorer = ModelScorer(
                **scorer_kwargs,
                cache_predictors=True,
                cache_subpath=f"{predictor_cache_path}/subject={subject}/target={target_id}/predictor=time={time}",
            )
            path = BONNER_CACHING_HOME / scorer.cache_path / "split=full.pkl"
            with path.open("rb") as f:
                model = pickle.load(f,)
            model.to("cpu")
            subject_weights.append(
                xr.DataArray(
                    model.weights(),
                    dims=("neuroid", feature_dim),
                    coords={
                        "neuroid": neuroid_coords,
                        feature_dim: feature_coords,
                    },
                ).assign_coords({"time": time})
            )
        if len(subject_times) > 0:
            weights.append(xr.concat(subject_weights, dim="time").assign_coords({"subject": subject}))
        # except:
        #     pass
        
    return xr.concat(weights, dim="subject")

def _split_weights(analysis, scorer_kwargs, data, split_dim, split_index, target_var, n_splits):
    split_weights = {}
    len_split_dim = len(data[split_dim].values)
    npps = len_split_dim//n_splits
    for ns in range(n_splits):
        temp_split_index = split_index[ns*npps:(ns+1)*npps]
        scorer = ModelScorer(**scorer_kwargs)
        
        temp_data = (data
            .isel({split_dim: temp_split_index})
            .stack(stack_presentation=(target_var, "presentation"))
            .transpose("stack_presentation", "neuroid",)
            .reset_index("stack_presentation")
            .drop_vars("presentation")
            .rename({"stack_presentation": "presentation"})
        )
        var_values = temp_data[target_var].values
        
        match analysis:
            case "multiclass":
                target = xr.DataArray(
                    np.expand_dims(
                        pd.Series(var_values).factorize()[0], axis=-1
                    ),
                    dims=("presentation", target_var),
                    coords={target_var: ['multiclass']},
                )
            case "behavior":
                embd = load_embeddings()
                target = embd.sel(object=var_values)
            case "bhvsubset":
                embd = sort_embeddings()
                target = embd.sel(object=var_values)
            case "bhvpc":
                embd = load_embeddings()
                target = embd.sel(object=var_values)
                pmodel = PCA()
                pmodel.fit(torch.tensor(target.values))

                target = target.copy(data=pmodel.transform(torch.tensor(target.values)).numpy())
                target["behavior"] = np.arange(len(target.behavior))+1
            case _:
                raise ValueError("analysis must be 'multiclass' or 'behavior'")
                
        model = scorer._fit_model(
            x_train=torch.tensor(temp_data.values).float().to(scorer.device),
            y_train=torch.tensor(target.values).float().to(scorer.device)
        )
        model.to("cpu")
        split_weights[ns] = model.weights()
    return split_weights

def _fit_aux(seed, subject_data, split_dim, target_var, analysis, n_splits, scorer_kwargs, feature_dim, neuroid_coords, feature_coords, t):
    retries = 0
    max_retries = MAX_RETRIES
    while retries < max_retries:
        try:
            rng = np.random.default_rng(seed)
            len_split_dim = len(subject_data[split_dim].values)
            
            if split_dim == target_var:
                split_weights = {}
                for i in range (n_splits):
                    split_index = rng.choice(np.arange(len_split_dim), size=len_split_dim, replace=False)
                    temp_split_weights = _split_weights(
                        analysis, 
                        scorer_kwargs, 
                        subject_data.sel(time=t), 
                        split_dim, split_index,
                        target_var, n_splits,
                    )
                    temp_split_weights = torch.concat(list(temp_split_weights.values()), dim=-1)
                    split_weights[i] = temp_split_weights[:, torch.argsort(split_index)]
            else:
                split_index = rng.choice(np.arange(len_split_dim), size=len_split_dim, replace=False)
                split_weights = _split_weights(
                    analysis, 
                    scorer_kwargs, 
                    subject_data.sel(time=t), 
                    split_dim, split_index,
                    target_var, n_splits,
                )
            return xr.DataArray(
                np.stack(list(split_weights.values()), axis=0),
                dims=("split", "neuroid", feature_dim),
                coords={
                    "neuroid": neuroid_coords,
                    feature_dim: feature_coords,
                },
            ).assign_coords({"time": t})
        except Exception as e:
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            print(f"Error: {e}. Retrying {retries}/{max_retries} in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # Raise an error if all retries fail
    raise RuntimeError(f"Max retries reached for _fit_aux at time={t}")

def _fit_and_load_split_weights(
    analysis: str,
    dataset: str,
    data: xr.DataArray,
    scorer_kwargs: dict,
    significant_only: bool,
    split_dim: str,
    n_splits: int,
    seed: int,
    target_var: str,
    feature_dim: str,
    feature_coords: list[str],
    neuroid_coords: list[str],
    times: xr.DataArray,
    subjects: (int | list[int] | str),
):
    subject_scores = []
    for subject in tqdm(_subjects(subjects, dataset), desc="subject"):
        try:
            subject_data = data.sel(subject=subject)
            subject_times = times.sel(subject=subject).dropna("time").values if significant_only else times
            
            time_scores = Parallel(n_jobs=-1, backend="loky")(delayed(_fit_aux)(seed, subject_data, split_dim, target_var, analysis, n_splits, scorer_kwargs, feature_dim, neuroid_coords, feature_coords, t) for t in subject_times)
            
            subject_scores.append(xr.concat(time_scores, dim="time").assign_coords({"subject": subject}))
        except:
            pass
    return xr.concat(subject_scores, dim="subject")
    
def load_weights(
    analysis: str, # "multiclass"
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    significant_only: bool,
    subjects: (int | list[int] | str) = None,
    alpha: float = 0.05,
    split_dim: str = None,
    n_splits: int = 2,
    n_seeds: int = 1,
) -> xr.DataArray:
    assert False
    # TODO: need to check compatibility with tt_decoding and load_significant_times
    
    loader_id = f"loaders/weights/analysis={analysis}"
    if split_dim is not None:
        loader_id += f"/split_dim={split_dim}.n_splits={n_splits}"
    else:
        loader_id += "/split_dim=None"
    
    if subjects is None:
        subjects = "intersection" if significant_only else "all"
    
    data = load_dataset(dataset, subjects="all" if subjects == "intersection" or subjects == "group" else subjects, **load_dataset_kwargs)
        
    cache_path = _cache_path(loader_id, dataset, load_dataset_kwargs, scorer_kwargs, include_root=False)
    if significant_only:
        cache_path += f"/significant_only={significant_only}.alpha={alpha}"
    else:
        cache_path += f"/significant_only={significant_only}"
    
    target_var = load_target_var(dataset)
    neuroid_coords = data.neuroid.values
    
    match analysis:
        case "behavior":
            feature_dim = "behavior"
            embd = load_embeddings()
            feature_coords = embd.behavior.values
        case "bhvsubset":
            feature_dim = "behavior"
            embd = sort_embeddings()
            feature_coords = embd.behavior.values
        case "bhvpc":
            feature_dim = "behavior"
            embd = load_embeddings()
            feature_coords =  np.arange(len(embd.behavior))+1
        case _:
            feature_dim = target_var
            feature_coords = data[feature_dim].values
        
    if significant_only:
        times = load_significant_times(analysis, dataset, load_dataset_kwargs, scorer_kwargs, feature_dim, alpha, subjects)
    else:
        times = data.time.values
    
    if split_dim is None:
        cacher = cache(cache_path + f"/subject={subjects}.nc")
        X = cacher(_load_weights)(analysis, dataset, load_dataset_kwargs, scorer_kwargs, significant_only, target_var, feature_dim, feature_coords, neuroid_coords, times, subjects)
    else:
        seed_scores = []
        for seed in tqdm(range(n_seeds), desc="seed"):
            cacher = cache(cache_path + f"/seed={seed}.subject={subjects}.nc")
            seed_scores.append(cacher(_fit_and_load_split_weights)(analysis, dataset, data, scorer_kwargs, significant_only, split_dim, n_splits, seed, target_var, feature_dim, feature_coords, neuroid_coords, times, subjects).assign_coords({"seed": seed}))
        X = xr.concat(seed_scores, dim="seed")
        
    return X
                



