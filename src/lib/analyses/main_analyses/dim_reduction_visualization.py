import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch

from lib.analyses.loaders import load_weights
from lib.datasets import (
    load_dataset,
    load_target_var
)
from lib.analyses._utilities import _cache_path
from lib.utilities import SEED

from sklearn.manifold import MDS

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(
    context="notebook",
    style="ticks",
    palette="Set2",
    rc={
        "figure.dpi": 256, "savefig.dpi": 256,
        "savefig.bbox": "tight",
        "figure.labelsize": "small",
    },
)



def dim_reduction_visualization(
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    space: str,
    reduce_dims: set[str, str],
    list_dim: str,
    analysis: str = "multiclass",
    subjects: (int | list[int] | str) = "all",
    alpha: float = 0.05,
    **kwargs,
):
    reduce_id = f"space={space}.reduce_dims={reduce_dims[0]}_{reduce_dims[1]}.list_dim={list_dim}"
    
    target_var = load_target_var(dataset)
    
    match space:
        case "weights":
            X = load_weights(analysis, dataset, load_dataset_kwargs, scorer_kwargs, subjects="intersection", significant_only=True, alpha=alpha)
        case "eeg":
            X = (
                load_dataset(dataset, subjects=subjects, **load_dataset_kwargs,)
                .mean("presentation")
            )
            X = X.assign_coords({target_var: np.arange(len(X[target_var]))})
    
    cache_path = _cache_path(f"main_analyses/dim_reduction_visualization/analysis={analysis}/{reduce_id}", dataset, load_dataset_kwargs, scorer_kwargs, include_root=True)
    
    for subject in X.subject.values:
        x = (X
            .sel(subject=subject)
            .dropna("time")
            .transpose(*reduce_dims, list_dim)
        )
        for l in x[list_dim].values:
            x_l = torch.tensor(x.sel({list_dim: l}).values)
            embedding = MDS(random_state=SEED)
            x_embd = embedding.fit_transform(x_l)
            df = pd.DataFrame(x_embd, columns=["embedding 1", "embedding 2"])
            df[reduce_dims[0]] = x[reduce_dims[0]].values
            
            plt.close()
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x="embedding 1", y="embedding 2", hue=reduce_dims[0], ax=ax)
            plt.tight_layout()
            
            fig_cache_path = cache_path / f"subject={subject}/{list_dim}={l.replace("/", "_or_")}.png"
            fig_cache_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_cache_path)
    