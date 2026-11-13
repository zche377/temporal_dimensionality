import shutil
import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import torch
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(
    context="paper",
    style="ticks",
    palette="Set2",
    rc={
        "figure.dpi": 600, "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.labelsize": "small",
    },
)

def _plot_eigenspectrum(
    eigenvalue,
    cache_path: Path = None,
):
    if isinstance(eigenvalue, pd.DataFrame):
        df = eigenvalue
    else:
        df = pd.DataFrame({"pc": np.arange(len(eigenvalue))+1, "eigenvalue": eigenvalue})
    plt.close()
    fig, ax = plt.subplots()
    sns.lineplot(df, x="pc", y="eigenvalue", ax=ax)
    ax.set(xscale="log", yscale="log")
    plt.tight_layout()
    sns.despine()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cache_path)
    return fig

def _plot_gen_with_p(
    df: pd.DataFrame,
    sig_df: pd.DataFrame,
    value_dim: str, 
    center: float = 0,
    cmap: str = "vlag",
    cbar_label: str = None,
    cache_path: Path = None,
    title: str = None,
):
    plt.close()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        df, ax=ax, 
        center=center,
        cmap=cmap,
        cbar_kws={'shrink': 0.75}
    )
    ax.invert_yaxis()
    
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])

    # Set font size for the remaining tick labels
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Add markers for significant cells
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if not sig_df.iloc[i, j]:  # If this cell is not significant
                # Add an asterisk in the center of the cell
                ax.text(j + 0.5, i + 0.5, '-', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=3, color='grey',
                        fontweight='bold')

    if cbar_label:
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=12,)
        cbar.ax.tick_params(labelsize=12)
        
    if title:
        ax.set_title(title, fontsize=16,)
    
    ax.set_xlabel(ax.get_xlabel(), fontsize=16,)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16,)
        
    if cache_path is not None:
        cache_path = cache_path / f"value_dim={value_dim}.png"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cache_path) 
        
def _plot_generalization(
    df: pd.DataFrame,
    row_dim: str,
    value_dim: str, 
    generalization_dim: str,
    cache_path: Path = None,
    subject: (int | str) = "all",
    reorder_by_cluster: bool = False,
    first_row_order_only: bool = True,
    order: list[float] = None,
    order_idx: list[float] = None,
    symmetric: bool = False,
    center: float = 0,
    n_row_lim: int = 8,
    cmap: str = "vlag",
    similarity=True,
) -> None:
    plt.close()
    if row_dim is not None:
        nrows = df[row_dim].nunique()
        row_dim_list = df[row_dim].unique()
    else:
        nrows = 1
        
    if n_row_lim is not None and nrows > n_row_lim:
        logging.info(f"Limit number of rows to {n_row_lim}")
        nrows = n_row_lim
    
    if nrows == 1:
        fig, ax = plt.subplots()
        ax = [ax]
    if n_row_lim is None:
        fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize=(5, nrows*4))
    else:
        fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize=(15, nrows*12))
    
    order_indices = []
    for i in range(nrows):
        iax = ax[i] if nrows > 1 else ax
        if nrows > 1:
            idf = df[df[row_dim]==row_dim_list[i]]
        else:
            idf = df
        idf = (idf
            .pivot(
                index=f"{generalization_dim}_test", columns=f"{generalization_dim}_train", values=value_dim
        ))
        if order_idx is not None:
            idf = idf.iloc[order_idx, order_idx]
            order_indices.append(order_idx)
        elif order is not None:
            order_idx = np.argsort(order)
            idf = idf.iloc[order_idx, order_idx]
            order_indices.append(order_idx)
        elif reorder_by_cluster:
            # needs to assert that values are similarity with max == 1
            if not first_row_order_only or i == 0:
                if similarity:
                    dm = np.triu(1 - idf.values, k=1)
                else:
                    dm = np.triu(idf.values, k=1)
                dm = np.nan_to_num(dm, nan=0)
                dm = ssd.squareform(dm + dm.T)
                linked = sch.linkage(dm, optimal_ordering=True, method="average")
                order_idx = sch.dendrogram(linked, no_plot=True)['leaves']
            idf = idf.iloc[order_idx, order_idx]
            order_indices.append(order_idx)
        
        sns.heatmap(idf.isna(), ax=iax, cmap=['black'], center=center, cbar=False,)
        sns.heatmap(idf, mask=idf.isna(), ax=iax, cmap=cmap, center=center)
        if symmetric:
            iax.set(xlabel=generalization_dim, ylabel=generalization_dim)
        if row_dim is not None:
            iax.set_title(f"{row_dim}={row_dim_list[i]}")
        iax.invert_yaxis()
    plt.xticks(rotation=90)
    plt.tight_layout()
    if cache_path is not None:
        cache_path = cache_path / f"value_dim={value_dim}" / f"subject={subject}.png"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cache_path) 
    if len(order_indices) == 0:
        return fig, None
    else:
        return fig, order_indices[0]