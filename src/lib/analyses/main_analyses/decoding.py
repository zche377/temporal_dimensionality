import shutil
import logging

logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import torch
from joblib import Parallel, delayed

from lib.analyses._utilities import _cache_path
from lib.computation.scorers import ModelScorer
from lib.datasets import load_dataset, load_target_var
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


def _reshape_subject_data(data, subject, target_var) -> xr.DataArray:
    return (
        data.sel(subject=subject)
        .stack(stack_presentation=(target_var, "presentation"))
        .transpose("stack_presentation", "neuroid", "time")
        .reset_index("stack_presentation")
        .drop_vars("presentation")
        .rename({"stack_presentation": "presentation"})
    )
    
def _average_presentation_data(data, subject, target_var) -> xr.DataArray:
    return (
        data.sel(subject=subject)
        .mean("presentation")
        .transpose(target_var, "neuroid", "time")
    )


def _object_pairwise_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    cache_path: str,
) -> xr.Dataset:
    subject_data = data.sel(subject=subject)
    pair_scores = []
    pairs = list(itertools.combinations(subject_data[target_var].values, 2))

    presentation_len = len(subject_data.presentation)
    target = xr.DataArray(
        np.expand_dims(
            np.concatenate(
                [
                    np.ones(presentation_len, dtype=int),
                    np.zeros(presentation_len, dtype=int),
                ]
            ),
            axis=-1,
        ),
        dims=("presentation", target_var),
    )
    cache_path += f"/temp/subject={subject}"

    def _for_cache(data, target, target_var, var_predictor_cache_path, pair):
        obj1, obj2 = pair
        pair_identifier = f"{obj1}-{obj2}"
        data = (
            data.sel({target_var: [obj1, obj2]})
            .stack(stack_presentation=(target_var, "presentation"))
            .transpose("stack_presentation", "neuroid", "time")
            .reset_index("stack_presentation")
            .drop_vars("presentation")
            .rename({"stack_presentation": "presentation"})
        )

        target = target.assign_coords({target_var: [pair_identifier]})
        time_scores = [
            ModelScorer(
                **scorer_kwargs,
                cache_predictors=True,
                cache_subpath=f"{var_predictor_cache_path}/predictor=time={t}",
            )(
                predictor=data.sel(time=t),
                target=target,
                target_dim=target_var,
            ).assign_coords(
                {"time": t}
            )
            for t in data.time.values
        ]

        logging.info(f"{pair_identifier}")

        return xr.concat(time_scores, dim="time")

    pair_scores = [
        cache(cache_path + f"/{pair[0]}_{pair[1]}.nc")(_for_cache)(
            subject_data,
            target,
            target_var,
            f"{predictor_cache_path}/target={target_var}={pair[0]}-{pair[1]}",
            pair,
        )
        for pair in pairs
    ]

    return xr.concat(pair_scores, dim=target_var).assign_coords({"subject": subject})


def _object_randompair_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
    cache_path: str,
) -> xr.Dataset:
    subject_data = data.sel(subject=subject)
    reshaped_data = (
        subject_data.stack(stack_presentation=(target_var, "presentation"))
        .transpose("stack_presentation", "neuroid", "time")
        .reset_index("stack_presentation")
        .drop_vars("presentation")
        .rename({"stack_presentation": "presentation"})
    )
    scores = []
    rng = np.random.default_rng(SEED)
    cache_path += f"/temp/subject={subject}"

    def _for_cache(predictor, target, var_predictor_cache_path, target_var):
        return xr.concat(
            [
                ModelScorer(
                    **scorer_kwargs,
                    cache_predictors=True,
                    cache_subpath=f"{var_predictor_cache_path}/predictor=time={t}",
                )(
                    predictor=predictor.sel(time=t),
                    target=target,
                    target_dim=target_var,
                ).assign_coords(
                    {"time": t}
                )
                for t in subject_data.time.values
            ],
            dim="time",
        )

    for var in subject_data[target_var].values:
        var_indices = reshaped_data.where(
            reshaped_data[target_var] == var, drop=True
        ).presentation.values
        non_var_indices = reshaped_data.where(
            reshaped_data[target_var] != var, drop=True
        ).presentation.values
        non_var_indices = rng.choice(non_var_indices, len(var_indices), replace=False)
        predictor = reshaped_data.sel(
            presentation=np.concatenate([var_indices, non_var_indices])
        )
        target = xr.DataArray(
            np.expand_dims(
                np.concatenate(
                    [
                        np.ones(len(var_indices), dtype=int),
                        np.zeros(len(var_indices), dtype=int),
                    ]
                ),
                axis=-1,
            ),
            dims=("presentation", target_var),
        )
        var_predictor_cache_path = f"{predictor_cache_path}/target={target_var}={var}"

        scores.append(
            cache(cache_path + f"/{var}.nc")(_for_cache)(
                predictor, target, var_predictor_cache_path, target_var
            )
        )

    return xr.concat(scores, dim=target_var).assign_coords({"subject": subject})


def _object_binary_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values
    var_values = data[target_var].values

    target = np.zeros((len(subject_var_values), len(var_values)), dtype=int)
    if subject_var_values.dtype != np.dtype("int64"):
        one_hot_index = pd.Series(subject_var_values).factorize()[0]
    else:
        one_hot_index = subject_var_values
    target[np.arange(len(subject_var_values)), one_hot_index] = 1
    target = xr.DataArray(
        target,
        dims=("presentation", target_var),
        coords={target_var: var_values},
    )
    time_scores = [
        ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target={target_var}=binary/predictor=time={t}",
        )(
            predictor=subject_data.sel(time=t),
            target=target,
            target_dim=target_var,
        ).assign_coords(
            {"time": t}
        )
        for t in subject_data.time.values
    ]
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})


def _object_multiclass_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values

    target = xr.DataArray(
        np.expand_dims(pd.Series(subject_var_values).factorize()[0], axis=-1),
        dims=("presentation", target_var),
        coords={target_var: ["multiclass"]},
    )
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target={target_var}=multiclass/predictor=time={t}",
        ))(
            predictor=subject_data.sel(time=t),
            target=target,
            target_dim=target_var,
        ) for t in subject_data.time.values)
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data.time.values)]

    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})


def _behavior_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values

    embd = load_embeddings()
    target = embd.sel(object=subject_var_values)
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=time={t}",
        ))(
            predictor=subject_data.sel(time=t),
            target=target,
            target_dim="behavior",
        ) for t in subject_data.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})


def _behavior_subset_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values

    embd = sort_embeddings()
    target = embd.sel(object=subject_var_values)
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=sortby=std.ntop=8/predictor=time={t}",
        ))(
            predictor=subject_data.sel(time=t),
            target=target,
            target_dim="behavior",
        ) for t in subject_data.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})


def _cross_target_behavior_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _average_presentation_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values

    embd = load_embeddings()
    target = embd.sel(object=subject_var_values)
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=time={t}",
        ))(
            predictor=subject_data.sel(time=t),
            target=target,
            target_dim="behavior",
        ) for t in subject_data.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})


def _cross_target_behavior_subset_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _average_presentation_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values

    embd = sort_embeddings()
    target = embd.sel(object=subject_var_values)
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=time={t}",
        ))(
            predictor=subject_data.sel(time=t),
            target=target,
            target_dim="behavior",
        ) for t in subject_data.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})


def _behavior_pc_decoding(
    data: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    predictor_cache_path: str,
) -> xr.Dataset:
    subject_data = _reshape_subject_data(data, subject, target_var)
    subject_var_values = subject_data[target_var].values

    embd = load_embeddings()
    target = embd.sel(object=subject_var_values)
    
    pmodel = PCA()
    pmodel.fit(torch.tensor(target.values))
    
    target = target.copy(data=pmodel.transform(torch.tensor(target.values)).numpy())
    target["behavior"] = np.arange(len(target.behavior))+1
    
    time_scores = Parallel(n_jobs=-1, backend="loky")(
        delayed(ModelScorer(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=behavior=all/predictor=time={t}",
        ))(
            predictor=subject_data.sel(time=t),
            target=target,
            target_dim="behavior",
        ) for t in subject_data.time.values)
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})


def decoding(
    analysis: str,
    dataset: str,
    subject: int,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
) -> xr.Dataset:
    cache_path = _cache_path(
        f"main_analyses/{analysis}_decoding",
        dataset,
        load_dataset_kwargs,
        scorer_kwargs,
        include_root=False,
    )
    predictor_cache_path = f"main_analyses/{analysis}_decoding/dataset={dataset}"
    predictor_cache_path = _append_path(
        predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs
    )

    match analysis:
        case "pairwise":
            analysis_fn = _object_pairwise_decoding
            cache_kwargs = {"cache_path": cache_path}
        case "randompair":
            analysis_fn = _object_randompair_decoding
            cache_kwargs = {"cache_path": cache_path}
        case "binary":
            analysis_fn = _object_binary_decoding
            cache_kwargs = {}
        case "multiclass":
            analysis_fn = _object_multiclass_decoding
            cache_kwargs = {}
        case "behavior":
            analysis_fn = _behavior_decoding
            cache_kwargs = {}
        case "bhvsubset":
            analysis_fn = _behavior_subset_decoding
            cache_kwargs = {}
        case "ctbehavior":
            analysis_fn = _cross_target_behavior_decoding
            cache_kwargs = {}
        case "ctbhvsubset":
            analysis_fn = _cross_target_behavior_subset_decoding
            cache_kwargs = {}
        case "bhvpc":
            analysis_fn = _behavior_pc_decoding
            cache_kwargs = {}
        case _:
            raise ValueError(
                "analysis not implemented"
            )

    data = load_dataset(
        dataset,
        subjects=subject,
        **load_dataset_kwargs,
    )
    if data is None:
        return xr.DataArray(None)
    
    if 'freq' not in data.coords:
        subject_path = f"{cache_path}/subject={subject}.nc"
        cacher = cache(
            subject_path,
            # mode="ignore",
        )
        score = cacher(analysis_fn)(
            data=data,
            subject=subject,
            target_var=load_target_var(dataset),
            scorer_kwargs=scorer_kwargs,
            predictor_cache_path=f"{predictor_cache_path}/subject={subject}",
            **cache_kwargs,
        )

        if analysis in ("pairwise", "randompair"):
            shutil.rmtree(BONNER_CACHING_HOME / cache_path / "temp" / f"subject={subject}")

        return xr.concat([score], dim="subject")
    else:
        scores = []
        for f in tqdm(data.freq.values, desc="freq"):
            subject_path = f"{cache_path}/freq={f}/subject={subject}.nc"
            cacher = cache(
                subject_path,
                # mode="ignore",
            )
            scores.append(cacher(analysis_fn)(
                data=data.sel(freq=f),
                subject=subject,
                target_var=load_target_var(dataset),
                scorer_kwargs=scorer_kwargs,
                predictor_cache_path=f"{predictor_cache_path}/freq={f}/subject={subject}",
                **cache_kwargs,
            ))

            if analysis in ("pairwise", "randompair"):
                shutil.rmtree(BONNER_CACHING_HOME / cache_path / "temp" / f"subject={subject}")

        return xr.concat([xr.concat(scores, dim="freq")], dim="subject")
