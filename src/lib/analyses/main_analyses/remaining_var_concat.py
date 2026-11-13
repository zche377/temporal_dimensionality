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
from lib.datasets import load_dataset, load_target_var, load_stimulus_set,load_n_subjects
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
        "figure.dpi": 500, "savefig.dpi": 500,
        "savefig.bbox": "tight",
        "figure.labelsize": "small",
    },
)

from sklearn.manifold import MDS
from bonner.computation.decomposition import PCA
from bonner.computation.metrics import euclidean_distance
from bonner.datasets.hebart2022_things_behavior import load_embeddings
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)


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

def _plot_umap(X, dataset, file_path, batch_size=32):
    x, y = X[:, 0], X[:, 1]
    stimulus_set = load_stimulus_set(dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    # model.eval()
    
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Process stimuli in batches
    for batch_start in tqdm(range(0, len(X), batch_size), desc="batch"):
        batch_end = min(batch_start + batch_size, len(X))
        
        # Load batch of stimuli
        batch_images = []
        batch_positions = []
        original_images = []
        
        for i, idx in enumerate(range(batch_start, batch_end)):
            img = stimulus_set.__getitem__(idx)
            batch_images.append(F.to_tensor(img))
            batch_positions.append((x[batch_start + i], y[batch_start + i]))
            original_images.append(img)
        
        # # Stack and move to device
        # img_tensors = torch.stack(batch_images).to(device)
        
        # # Perform inference
        # with torch.no_grad():
        #     batch_outputs = model(img_tensors)
        
        # # Process each image in the batch
        # for i, (img, output) in enumerate(zip(original_images, batch_outputs)):
        #     for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        #         if label == 1 and score > 0.8:  # "person" category and confidence > 0.8
        #             x1, y1, x2, y2 = map(int, box.tolist())
                    
        #             # Crop and blur the face
        #             face_region = img.crop((x1, y1, x2, y2))
        #             blurred_face = face_region.filter(ImageFilter.GaussianBlur(10))
                    
        #             # Paste the blurred face back into the image
        #             img.paste(blurred_face, (x1, y1))
            
            # Convert the image back to a numpy array
            stim_val = np.array(img)
            
            # Add the image to the plot
            image_box = OffsetImage(stim_val, zoom=0.05)
            image_box.image.axes = ax
            ab = AnnotationBbox(
                image_box,
                xy=batch_positions[i],
                xycoords="data",
                frameon=False,
                pad=0,
            )
            ax.add_artist(ab)
    
    # Finalize the plot
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.axis("off")
    
    # Save the figure
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    
    return None    

def _all_var(
    dataset: str,
    load_dataset_kwargs: dict,
    n_subject: int,
    target_var: str,
    scorer_kwargs: dict,
    cache_path: str,
    predictor_cache_path: str,
    target_timepoint: float,
    pca: bool,
    clustering_target: str,
    test_images: bool,
) -> xr.Dataset:
    distances = []
    for subject in range(1, n_subject+1):
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
    
        subject_data_train = _reshape_subject_data(data_train, subject, target_var, average=True).sel(time=target_timepoint)
        subject_data_test = _reshape_subject_data(data_test, subject, target_var, average=True).sel(time=target_timepoint)
    
        if pca:
            pmodel = PCA()
            pmodel.fit(torch.tensor(subject_data_train.values))
            temp_train = pmodel.transform(torch.tensor(subject_data_train.values))
            temp_test = pmodel.transform(torch.tensor(subject_data_test.values))
            subject_data_train = subject_data_train.copy(data=temp_train.cpu().numpy())
            subject_data_test = subject_data_test.copy(data=temp_test.cpu().numpy())
            subject_data_train["neuroid"] = np.arange(len(subject_data_train.neuroid))+1
            subject_data_test["neuroid"] = np.arange(len(subject_data_test.neuroid))+1

        if test_images:
            y_remain = torch.tensor(subject_data_test.values)
        else:
            y_remain = torch.tensor(subject_data_train.values)
            
        distance = euclidean_distance(y_remain.T, return_diagonal=False)
        
        match clustering_target:
            case "all":
                mapper = umap.UMAP(random_state=SEED, metric="precomputed")  
                y_umap = mapper.fit_transform(distance)
                _plot_umap(
                    y_umap,
                    f"{dataset}_test" if test_images else f"{dataset}_train",
                    file_path=f"{cache_path}/single_subject/subject={subject}.target_timepoint={target_timepoint}.png",
                    batch_size=32,
                )
            case _:
                raise ValueError("clustering_target not implemented")

        distances.append(distance)
        
    distances = torch.mean(torch.stack(distances,), dim=0).numpy()
    
    match clustering_target:
        case "all":
            mapper = umap.UMAP(random_state=SEED, metric="precomputed")  
            y_umap = mapper.fit_transform(distances)
            _plot_umap(
                y_umap,
                f"{dataset}_test" if test_images else f"{dataset}_train",
                file_path=f"{cache_path}/target_timepoint={target_timepoint}.png",
                batch_size=32,
            )
        case _:
            raise ValueError("clustering_target not implemented")
    
def _behavior_remaining_var_concat(
    dataset: str,
    load_dataset_kwargs: dict,
    n_subject: int,
    target_var: str,
    scorer_kwargs: dict,
    cache_path: str,
    predictor_cache_path: str,
    target_timepoint: float,
    pca: bool,
    clustering_target: str,
    test_images: bool,
) -> xr.Dataset:
    fo = []
    for subject in range(1, n_subject+1):
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
    
        subject_data_train = _reshape_subject_data(data_train, subject, target_var, average=True).sel(time=target_timepoint)
        subject_data_test = _reshape_subject_data(data_test, subject, target_var, average=True).sel(time=target_timepoint)
        
        if subject == 1:
            embd = load_embeddings()
            target_train = embd.sel(object=subject_data_train.presentation.values)
            target_test = embd.sel(object=subject_data_test.presentation.values)
    
        if pca:
            pmodel = PCA()
            pmodel.fit(torch.tensor(subject_data_train.values))
            temp_train = pmodel.transform(torch.tensor(subject_data_train.values))
            temp_test = pmodel.transform(torch.tensor(subject_data_test.values))
            subject_data_train = subject_data_train.copy(data=temp_train.cpu().numpy())
            subject_data_test = subject_data_test.copy(data=temp_test.cpu().numpy())
            subject_data_train["neuroid"] = np.arange(len(subject_data_train.neuroid))+1
            subject_data_test["neuroid"] = np.arange(len(subject_data_test.neuroid))+1
    
        # TEMP
        assert scorer_kwargs["model_name"] != "plssvd"
        scorer_fn = TrainTestModelScorer
        scorer = scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/subject={subject}/target=time={target_timepoint}/predictor=behavior=all",
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
        if test_images:
            y_remain = (_2t(subject_data_test) - lmodel.predict(_2t(target_test))).cpu()
        else:
            y_remain = (_2t(subject_data_train) - lmodel.predict(_2t(target_train))).cpu()
        
        fo.append(y_remain)
    
    fo = torch.concat(fo, dim=-1)
    distances = euclidean_distance(fo.T, return_diagonal=False).numpy()
    
    match clustering_target:
        case "all":
            mapper = umap.UMAP(random_state=SEED, metric="precomputed")  
            y_umap = mapper.fit_transform(distances)
            _plot_umap(
                y_umap,
                f"{dataset}_test" if test_images else f"{dataset}_train",
                file_path=f"{cache_path}/target_timepoint={target_timepoint}.png",
                batch_size=32,
            )
        case _:
            raise ValueError("clustering_target not implemented")
 
def _model_srp_remaining_var_concat(
    dataset: str,
    data_train: xr.DataArray,
    data_test: xr.DataArray,
    subject: int,
    target_var: str,
    scorer_kwargs: dict,
    cache_path: str,
    predictor_cache_path: str,
    target_timepoint: float,
    pca: bool,
    clustering_target: str,
    test_images: bool,
    model_uid: str,
    n_components,
    nodes: str = "Identity"
):
    assert False, "update to subject average"
    subject_data_train = _reshape_subject_data(data_train, subject, target_var, average=True).sel(time=target_timepoint)
    subject_data_test = _reshape_subject_data(data_test, subject, target_var, average=True).sel(time=target_timepoint)

    model = DJModel(model_uid, hook="srp", n_components=n_components, nodes=nodes)
    target = model(dataset, dataloader_kwargs={"batch_size": 32})
    
    if pca:
        pmodel = PCA()
        pmodel.fit(torch.tensor(subject_data_train.values))
        temp_train = pmodel.transform(torch.tensor(subject_data_train.values))
        temp_test = pmodel.transform(torch.tensor(subject_data_test.values))
        subject_data_train = subject_data_train.copy(data=temp_train.cpu().numpy())
        subject_data_test = subject_data_test.copy(data=temp_test.cpu().numpy())
        subject_data_train["neuroid"] = np.arange(len(subject_data_train.neuroid))+1
        subject_data_test["neuroid"] = np.arange(len(subject_data_test.neuroid))+1
    
    def _2t(x):
        return torch.tensor(x.values).float().to(scorer.device)
    
    # TEMP
    assert scorer_kwargs["model_name"] != "plssvd"
    scorer_fn = TrainTestModelScorer
    
    for node, feature_map in target.items():
        feature_map_train = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_train.img_files))).sortby("img_files")
        feature_map_test = deepcopy(feature_map.sel(presentation=feature_map.img_files.isin(subject_data_test.img_files))).sortby("img_files")
        
        # z-score the feature maps
        temp = torch.tensor(feature_map_train.values, device="cuda")
        temp = (temp-temp.mean(dim=0))/temp.std(dim=0)
        feature_map_train = feature_map_train.copy(data=temp.cpu().numpy())
        temp = torch.tensor(feature_map_test.values, device="cuda")
        temp = (temp-temp.mean(dim=0))/temp.std(dim=0)
        feature_map_test = feature_map_test.copy(data=temp.cpu().numpy())
    
        scorer = scorer_fn(
            **scorer_kwargs,
            cache_predictors=True,
            cache_subpath=f"{predictor_cache_path}/target=time={target_timepoint}/predictor=node={node}",
        )
    
        cacher = cache(
            f"{scorer.cache_path}.pkl",
            mode = "normal" if scorer.cache_predictors else "ignore",
        )
        
        lmodel = cacher(scorer._fit_model)(
            x_train=_2t(feature_map_train), y_train=_2t(subject_data_train)
        )
        if test_images:
            y_remain = _2t(subject_data_test) - lmodel.predict(_2t(feature_map_test))
            y_remain = subject_data_test.copy(data=y_remain.cpu().numpy())
        else:
            y_remain = _2t(subject_data_train) - lmodel.predict(_2t(feature_map_train))
            y_remain = subject_data_train.copy(data=y_remain.cpu().numpy())
        
        logging.info(subject_data_test.values.std(axis=0))
        logging.info(y_remain.values.std(axis=0))
        quit()
        
        match clustering_target:
            case "all":
                mapper = umap.UMAP(random_state=SEED)  
                y_umap = mapper.fit_transform(y_remain.values)
                # mapper = MDS()
                # y_umap = mapper.fit_transform(y_remain.values)
                _plot_umap(
                    y_umap,
                    f"{dataset}_test" if test_images else f"{dataset}_train",
                    file_path=f"{cache_path}/node={node}.target_timepoint={target_timepoint}.png",
                    batch_size=32,
                )
            case _:
                raise ValueError("clustering_target not implemented")

def remaining_var_concat(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    target_timepoint: float,
    pca: bool = False,
    clustering_target: str = "all", # "all", "single"
    test_images: bool = True,
    model_kwargs: dict = None,
) -> xr.Dataset:
    cache_str = f"main_analyses/remaining_var_concat.pca={pca}.test_images={test_images}/regress_out={analysis}"
    # cache_str = f"main_analyses/{analysis}_remaining_var_concat_mds.pca={pca}.test_images={test_images}"
    cache_path = _cache_path(
        cache_str,
        dataset,
        load_dataset_kwargs,
        scorer_kwargs,
        include_root=True,
    )
    predictor_cache_path = f"main_analyses/{analysis}_tt_encoding.subset=False.pca={pca}/dataset={dataset}"
    predictor_cache_path = _append_path(
        predictor_cache_path, "load_dataset_kwargs", load_dataset_kwargs
    )

    match analysis:
        case "none":
            analysis_fn = _all_var
            cache_kwargs = {}
        case "behavior":
            analysis_fn = _behavior_remaining_var_concat
            cache_kwargs = {}
        case "model_srpz":
            assert "model_uid" in list(model_kwargs.keys())
            analysis_fn = _model_srp_remaining_var_concat
            
            model_kwargs["n_components"] = "auto"
            
            cache_kwargs = deepcopy(model_kwargs)
            
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
    
    n_subject = load_n_subjects(dataset)
    target_var = load_target_var(f"{dataset}_train")
    
    analysis_fn(
        dataset=dataset,
        load_dataset_kwargs=load_dataset_kwargs,
        n_subject=n_subject,
        target_var=target_var,
        scorer_kwargs=scorer_kwargs,
        cache_path=cache_path,
        predictor_cache_path=predictor_cache_path,
        target_timepoint=target_timepoint,
        pca=pca,
        clustering_target=clustering_target,
        test_images=test_images,
        **cache_kwargs,
    )
