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
from copy import deepcopy

from lib.analyses._utilities import _cache_path
from lib.computation.scorers import TrainTestModelScorer, TrainTestPLSSVDScorer
from lib.datasets import load_dataset, load_target_var
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

def _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True,):
    if reconstruct_with_pc:
        subject_data_train = data_train.sel(subject=subject)
        subject_data_test = data_test.sel(subject=subject)
    else:
        subject_data_train = _reshape_subject_data(data_train, subject, target_var, average)
        subject_data_test = _reshape_subject_data(data_test, subject, target_var, average)
    return subject_data_train.sortby("img_files"), subject_data_test.sortby("img_files")

def _behavior_encoding(
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
    
    temp = torch.tensor(target_train.values, device="cuda")
    mean = temp.mean(dim=0)
    std = temp.std(dim=0)
    target_train = xr.DataArray(
        ((temp-mean)/std).cpu().numpy(),
        dims=["presentation", "neuroid"],
    )
    temp = torch.tensor(target_test.values, device="cuda")
    target_test = xr.DataArray(
        ((temp-mean)/std).cpu().numpy(),
        dims=["presentation", "neuroid"],
    )
    
    if pca:
        temp_train, temp_test = [], []
        for t in subject_data_test.time.values:
            pmodel = PCA()
            pmodel.fit(torch.tensor(subject_data_train.sel(time=t).values))
            temp_train.append(pmodel.transform(torch.tensor(subject_data_train.sel(time=t).values)))
            temp_test.append(pmodel.transform(torch.tensor(subject_data_test.sel(time=t).values)))
        subject_data_train = subject_data_train.copy(data=torch.stack(temp_train, dim=-1).cpu().numpy())
        subject_data_test = subject_data_test.copy(data=torch.stack(temp_test, dim=-1).cpu().numpy())
        subject_data_train["neuroid"] = np.arange(len(subject_data_train.neuroid))+1
        subject_data_test["neuroid"] = np.arange(len(subject_data_test.neuroid))+1
    
    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer
    
    time_scores = Parallel(n_jobs=1, backend="loky")(
        delayed(scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=time={t}/predictor=behavior=all",
        ))(
            predictor_train=target_train,
            target_train=subject_data_train.sel(time=t),
            predictor_test=target_test,
            target_test=subject_data_test.sel(time=t),
            target_dim="neuroid",
        ) for t in subject_data_test.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data_test.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})

def _model_srp_encoding(
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
    nodes: str = "Identity"
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True)
    
    if pca:
        temp_train, temp_test = [], []
        for t in subject_data_test.time.values:
            pmodel = PCA()
            pmodel.fit(torch.tensor(subject_data_train.sel(time=t).values))
            temp_train.append(pmodel.transform(torch.tensor(subject_data_train.sel(time=t).values)))
            temp_test.append(pmodel.transform(torch.tensor(subject_data_test.sel(time=t).values)))
        subject_data_train = subject_data_train.copy(data=torch.stack(temp_train, dim=-1).cpu().numpy())
        subject_data_test = subject_data_test.copy(data=torch.stack(temp_test, dim=-1).cpu().numpy())
        subject_data_train["neuroid"] = np.arange(len(subject_data_train.neuroid))+1
        subject_data_test["neuroid"] = np.arange(len(subject_data_test.neuroid))+1
    
    model = DJModel(model_uid, hook="srp", n_components=n_components, nodes=nodes)
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
            
        time_scores = Parallel(n_jobs=1, backend="loky")(
            delayed(scorer_fn(
                **scorer_kwargs,
                cache_predictors=True,
                cache_subpath=f"{predictor_cache_path}/target=time={t}/predictor=node={node}",
                # device="cpu"
            ))(
                predictor_train=feature_map_train,
                target_train=subject_data_train.sel(time=t),
                predictor_test=feature_map_test,
                target_test=subject_data_test.sel(time=t),
                target_dim="neuroid",
            ) for t in subject_data_test.time.values)
    
        time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data_test.time.values)]
        scores.append(xr.concat(time_scores, dim="time").assign_coords({"node": node}))

    return xr.concat(scores, dim="node").assign_coords({"subject": subject})

def _everything_encoding(
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
    nodes: str = "Identity"
) -> xr.Dataset:
    subject_data_train, subject_data_test = _subject_data(data_train, data_test, subject, target_var, reconstruct_with_pc, average=True)
    
    if pca:
        temp_train, temp_test = [], []
        for t in subject_data_test.time.values:
            pmodel = PCA()
            pmodel.fit(torch.tensor(subject_data_train.sel(time=t).values))
            temp_train.append(pmodel.transform(torch.tensor(subject_data_train.sel(time=t).values)))
            temp_test.append(pmodel.transform(torch.tensor(subject_data_test.sel(time=t).values)))
        subject_data_train = subject_data_train.copy(data=torch.stack(temp_train, dim=-1).cpu().numpy())
        subject_data_test = subject_data_test.copy(data=torch.stack(temp_test, dim=-1).cpu().numpy())
        subject_data_train["neuroid"] = np.arange(len(subject_data_train.neuroid))+1
        subject_data_test["neuroid"] = np.arange(len(subject_data_test.neuroid))+1
    
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
    
    if scorer_kwargs["model_name"] == "plssvd":
        scorer_fn = TrainTestPLSSVDScorer
    else:
        scorer_fn = TrainTestModelScorer
    
    time_scores = Parallel(n_jobs=1, backend="loky")(
        delayed(scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=time={t}/predictor=everything",
        ))(
            predictor_train=target_train,
            target_train=subject_data_train.sel(time=t),
            predictor_test=target_test,
            target_test=subject_data_test.sel(time=t),
            target_dim="neuroid",
        ) for t in subject_data_test.time.values)
    
    time_scores = [time_scores[i].assign_coords({"time": t}) for i, t in enumerate(subject_data_test.time.values)]
    
    return xr.concat(time_scores, dim="time").assign_coords({"subject": subject})

def tt_encoding(
    analysis: str,
    dataset: str,
    subject: int,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    subset: bool = False,
    pca: bool = False,
    reconstruct_with_pc: bool = False,
    subset_method: str = "single",
    pc_timepoint: float = .37,
    model_kwargs: dict = None,
) -> xr.Dataset:
    assert not reconstruct_with_pc # TODO: require re-examination
    
    cache_str = f"main_analyses/{analysis}_tt_encoding.subset={subset}.pca={pca}"
    if reconstruct_with_pc:
        cache_str += f".reconstruct_with_pc={reconstruct_with_pc}.subset_method={subset_method}.pc_timepoint={pc_timepoint}"
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
            analysis_fn = _behavior_encoding
            cache_kwargs = {}
            cache_kwargs["subset"] = subset
        case "model_srp":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srp_encoding
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            cache_kwargs["dataset"] = dataset
            
            cache_path = _append_path(
                cache_path, "model_kwargs", model_kwargs
            )
            predictor_cache_path = _append_path(
                predictor_cache_path, "model_kwargs", model_kwargs
            )
        case "model_srp_relu":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srp_encoding
            
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
        # change to z for feature-wise z-score
        case "everythingz":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _everything_encoding
            
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
            analysis_fn = _everything_encoding
            
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
    target_var = load_target_var(f"{dataset}_train")
    
    if reconstruct_with_pc:
        data_train, data_test = _reconstruct_data_with_pc_subject(subset_method, pc_timepoint, target_var, data_train, data_test)
    
    if 'freq' not in data_train.coords and not reconstruct_with_pc:
        subject_path = f"{cache_path}/subject={subject}.nc"
        cacher = cache(
            subject_path,
            # mode="ignore",
        )
        score = cacher(analysis_fn)(
            data_train=data_train,
            data_test=data_test,
            subject=subject,
            target_var=target_var,
            scorer_kwargs=scorer_kwargs,
            predictor_cache_path=f"{predictor_cache_path}/subject={subject}",
            pca=pca,
            reconstruct_with_pc=reconstruct_with_pc,
            **cache_kwargs,
        )
        
        return xr.concat([score], dim="subject")
    elif reconstruct_with_pc:
        scores = []
        for pc in tqdm(data_train.pc.values, desc="pc"):
            subject_path = f"{cache_path}/pc={pc}/subject={subject}.nc"
            cacher = cache(
                subject_path,
                # mode="ignore",
            )
            scores.append(cacher(analysis_fn)(
                data_train=data_train.sel(pc=pc),
                data_test=data_test.sel(pc=pc),
                subject=subject,
                target_var=target_var,
                scorer_kwargs=scorer_kwargs,
                predictor_cache_path=f"{predictor_cache_path}/pc={pc}/subject={subject}",
                pca=pca,
                reconstruct_with_pc=reconstruct_with_pc,
                **cache_kwargs,
            ))

        return xr.concat([xr.concat(scores, dim="pc")], dim="subject")

    else:
        scores = []
        for f in tqdm(data_train.freq.values, desc="freq"):
            subject_path = f"{cache_path}/freq={f}/subject={subject}.nc"
            cacher = cache(
                subject_path,
                # mode="ignore",
            )
            scores.append(cacher(analysis_fn)(
                data_train=data_train.sel(freq=f),
                data_test=data_test.sel(freq=f),
                subject=subject,
                target_var=target_var,
                scorer_kwargs=scorer_kwargs,
                predictor_cache_path=f"{predictor_cache_path}/freq={f}/subject={subject}",
                subset=subset,
                pca=pca,
                reconstruct_with_pc=reconstruct_with_pc,
                **cache_kwargs,
            ))

        return xr.concat([xr.concat(scores, dim="freq")], dim="subject")
