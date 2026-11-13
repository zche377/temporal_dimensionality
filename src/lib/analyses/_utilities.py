import logging
logging.basicConfig(level=logging.INFO)

from lib.utilities import _append_path

from bonner.caching import BONNER_CACHING_HOME


        
def _cache_path(identifier, dataset, load_dataset_kwargs, scorer_kwargs, include_root):
    cache_path = f"{identifier}/dataset={dataset}"
    cache_path = _append_path(cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    if scorer_kwargs is not None:
        cache_path = _append_path(cache_path, "scorer_kwargs", scorer_kwargs)
    if include_root:
        return BONNER_CACHING_HOME / cache_path
    else:
        return cache_path
    
def _model_analyses_cache_path(identifier, scorer_kwargs, include_root):
    cache_path = _append_path(identifier, "scorer_kwargs", scorer_kwargs)
    if include_root:
        return BONNER_CACHING_HOME / cache_path
    else:
        return cache_path
        