import logging

logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import xarray as xr
import torch
from joblib import Parallel, delayed

from lib.analyses._utilities import _model_analyses_cache_path
from lib.computation.scorers import ModelScorer
from lib.models import DJModel
from lib.datasets import load_metadata
from lib.datasets.hebart2022_things_behavior import sort_embeddings

from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)

# note: here it's decoding behavior from model, not decoding model from eeg

# compared to decoding.py, everything are cross-target (object) here


def _cross_target_behavior_decoding(
    feature_maps: dict[str, xr.DataArray],
    dataset: str,
    model_uid: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    embd = load_embeddings()
    metadata = load_metadata(dataset)
    target = embd.sel(object=metadata.object.values)
    
    node_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=node={node}",
        ))(
            predictor=feature_map,
            target=target,
            target_dim="behavior",
        ) for node, feature_map in feature_maps.items()
    )
    
    node_scores = [node_scores[i].assign_coords({"node": node}) for i, node in enumerate(feature_maps.keys())]
    return xr.concat(node_scores, dim="node").assign_coords({"model": model_uid})


def model_decoding(
    analysis: str,
    dataset: str,
    model_uid: str,
    seed: int,
    scorer_kwargs: dict,
    **kwargs,
) -> xr.Dataset:
    cache_path = _model_analyses_cache_path(
        f"model_analyses/{analysis}_decoding",
        scorer_kwargs,
        include_root=False,
    )
    predictor_cache_path = f"model_analyses/{analysis}_decoding"
    
    match analysis:
        case "ctbhv":
            analysis_fn = _cross_target_behavior_decoding
            cache_kwargs = {"dataset": dataset}
        case _:
            raise ValueError(
                "analysis not implemented"
            )
            
    model = DJModel(model_uid, seed=seed)
    feature_maps = model(dataset, dataloader_kwargs={"batch_size": 64})
    
    model_path = f"{cache_path}/model={model_uid}.seed={seed}.nc"
    cacher = cache(
        model_path,
        # mode="ignore",
    )
    score = cacher(analysis_fn)(
        feature_maps=feature_maps,
        model_uid=model_uid,
        scorer_kwargs=scorer_kwargs,
        predictor_cache_path=f"{predictor_cache_path}/model={model_uid}.seed={seed}",
        **cache_kwargs,
    )
    
    return score
