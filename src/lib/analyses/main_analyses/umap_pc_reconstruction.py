import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
from tqdm import tqdm
import numpy as np
import xarray as xr
import torch
import umap

from PIL import Image
from torchvision.transforms import functional as F
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from lib.analyses._utilities import _cache_path
from lib.datasets import load_dataset, load_target_var, load_stimulus_set, load_n_subjects
from lib.utilities import SEED

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

from bonner.computation.decomposition import PCA
from bonner.computation.metrics import pearson_r
from bonner.caching import cache


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

    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))

    for batch_start in tqdm(range(0, len(X), batch_size), desc="batch"):
        batch_end = min(batch_start + batch_size, len(X))

        batch_positions = []
        original_images = []

        for i, idx in enumerate(range(batch_start, batch_end)):
            img = stimulus_set.__getitem__(idx)
            batch_positions.append((x[batch_start + i], y[batch_start + i]))
            original_images.append(img)

        for i, img in enumerate(original_images):
            stim_val = np.array(img)

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

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.axis("off")

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)

    return None


def _reconstruct_with_pcs(pmodel, data_train, data_test, start_pc, end_pc):
    """
    Reconstruct data by keeping only PCs from start_pc to end_pc (1-indexed, inclusive).

    PCA is fit on train, then transform test, zero out PCs outside [start_pc, end_pc],
    then inverse transform back.

    Args:
        pmodel: Fitted PCA model
        data_train: Training data (for reference, already used to fit pmodel)
        data_test: Test data to transform and reconstruct
        start_pc: Starting PC index (1-indexed, inclusive)
        end_pc: Ending PC index (1-indexed, inclusive), or None to keep all PCs from start_pc

    Returns:
        Reconstructed test data
    """
    transformed = pmodel.transform(torch.tensor(data_test.values))

    # Zero out PCs before start_pc (0-indexed: indices 0 to start_pc-2)
    if start_pc > 1:
        transformed[:, :start_pc-1] = 0

    # Zero out PCs after end_pc if specified
    if end_pc is not None and end_pc < transformed.shape[1]:
        transformed[:, end_pc:] = 0

    reconstructed = pmodel.inverse_transform(transformed)
    return reconstructed


def _umap_pc_reconstruction_single_k(
    dataset: str,
    load_dataset_kwargs: dict,
    n_subject: int,
    target_var: str,
    cache_path: str,
    target_timepoint: float,
    test_images: bool,
    start_pc: int,
    total_pcs: int,
) -> tuple:
    """
    Compute UMAP for data reconstructed from PCs start_pc to total_pcs.

    Args:
        start_pc: Starting PC index (1-indexed)
        total_pcs: Total number of PCs (ending PC index, 1-indexed)
    """
    distances = []

    for subject in range(1, n_subject + 1):
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

        subject_data_train = _reshape_subject_data(
            data_train, subject, target_var, average=True
        ).sel(time=target_timepoint)
        subject_data_test = _reshape_subject_data(
            data_test, subject, target_var, average=True
        ).sel(time=target_timepoint)

        # Fit PCA on train data
        pmodel = PCA()
        pmodel.fit(torch.tensor(subject_data_train.values))

        # Reconstruct test data keeping only PCs from start_pc to total_pcs
        if test_images:
            reconstructed = _reconstruct_with_pcs(
                pmodel, subject_data_train, subject_data_test, start_pc, total_pcs
            )
        else:
            reconstructed = _reconstruct_with_pcs(
                pmodel, subject_data_train, subject_data_train, start_pc, total_pcs
            )

        y_data = reconstructed

        def pdistance(x, **kwargs):
            return 1 - pearson_r(x, **kwargs)

        distance = pdistance(y_data.T, return_diagonal=False)
        distance.fill_diagonal_(0)

        distances.append(distance)

    # Average distances across subjects
    distances = torch.mean(torch.stack(distances), dim=0).numpy()

    # Compute UMAP
    mapper = umap.UMAP(random_state=SEED, metric="precomputed")
    y_umap = mapper.fit_transform(distances)

    # Plot UMAP
    _plot_umap(
        y_umap,
        f"{dataset}_test" if test_images else f"{dataset}_train",
        file_path=f"{cache_path}/target_timepoint={target_timepoint}/start_pc={start_pc}.png",
        batch_size=32,
    )

    return distances, y_umap


def umap_pc_reconstruction(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    target_timepoint: float,
    test_images: bool = True,
    total_pcs: int = None,
) -> dict:
    """
    Generate UMAP plots for data reconstructed with different PC ranges.

    For each k from 1 to total_pcs, reconstruct the representation by:
    1. Fit PCA on train data
    2. Transform test data
    3. Zero out the first k-1 PCs (keep PCs k to total_pcs)
    4. Inverse transform back
    5. Compute pairwise distances and run UMAP

    Each UMAP result for each k is saved separately.

    Args:
        dataset: Dataset name
        load_dataset_kwargs: Kwargs for load_dataset
        scorer_kwargs: Scorer kwargs (for cache path consistency)
        target_timepoint: Time point to analyze
        test_images: If True, use test images; if False, use train images
        total_pcs: Total number of PCs to use (if None, determined from data)

    Returns:
        Dictionary with distances and UMAP coordinates for each k
    """
    cache_str = f"main_analyses/umap_pc_reconstruction.test_images={test_images}"
    cache_path = _cache_path(
        cache_str,
        dataset,
        load_dataset_kwargs,
        scorer_kwargs,
        include_root=True,
    )

    n_subject = load_n_subjects(dataset)
    target_var = load_target_var(f"{dataset}_train")

    # Determine total_pcs from data if not specified
    if total_pcs is None:
        data_train = load_dataset(
            f"{dataset}_train",
            subjects=1,
            **load_dataset_kwargs,
        )
        subject_data_train = _reshape_subject_data(
            data_train, 1, target_var, average=True
        ).sel(time=target_timepoint)
        total_pcs = min(len(subject_data_train.neuroid), len(subject_data_train.presentation))

    results = {}

    for start_pc in tqdm(range(1, total_pcs + 1), desc="PC range"):
        logging.info(f"Processing start_pc={start_pc} (keeping PCs {start_pc} to {total_pcs})")

        cacher = cache(
            f"{cache_path}/target_timepoint={target_timepoint}/start_pc={start_pc}.pkl",
        )

        distances, y_umap = cacher(_umap_pc_reconstruction_single_k)(
            dataset=dataset,
            load_dataset_kwargs=load_dataset_kwargs,
            n_subject=n_subject,
            target_var=target_var,
            cache_path=cache_path,
            target_timepoint=target_timepoint,
            test_images=test_images,
            start_pc=start_pc,
            total_pcs=total_pcs,
        )

        results[start_pc] = {
            "distances": distances,
            "umap": y_umap,
        }

    return results
