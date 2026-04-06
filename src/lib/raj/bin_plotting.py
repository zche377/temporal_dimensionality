import typing

import numpy as np
import numpy.typing as npt
import xarray as xr
from flox.xarray import xarray_reduce

from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from bonner.plotting import apply_offset
from matplotlib.axes import Axes

T = typing.TypeVar("T", xr.DataArray, xr.Dataset)


def offset_spectra(
    spectra: pd.DataFrame,
    /,
    *,
    keys: Sequence[str],
    offset_key: str = "rank",
    offset_magnitude: float = 1.025,
) -> pd.DataFrame:
    return apply_offset(
        spectra,
        keys=keys,
        offset_key=offset_key,
        offset_magnitude=offset_magnitude,
        offset_type="multiplicative",
    ).reset_index(drop=True)


def plot_spectrum(
    ax: Axes,
    *,
    spectrum: xr.Dataset,
    metric: str = "covariance",
    errorbar: tuple[str, float] = ("fold-sd", 1),
    hide_insignificant: bool = False,
    **kwargs,
) -> None:
    means = spectrum[metric].mean(dim="fold")

    match errorbar:
        case ("fold-sd", n):
            fold_sd = spectrum[metric].std("fold", ddof=1)
            errors = n * fold_sd
        case ("fold-se", n):
            fold_se = spectrum[metric].std("fold", ddof=1) / np.sqrt(
                spectrum.sizes["fold"] - 1,
            )
            errors = n * fold_se
        case ("bootstrap", n):
            bootstrap_sd = (
                spectrum[f"{metric} (bootstrapped)"]
                .mean("fold")
                .std("bootstrap", ddof=1)
            )
            errors = n * bootstrap_sd.rename(metric)
        case _:
            raise NotImplementedError

    if hide_insignificant:
        significant = spectrum["significant"].to_numpy()
        ax.errorbar(
            means["rank"].isel(rank=significant),
            means.isel(rank=significant),
            errors.isel(rank=significant),
            **kwargs,
        )
        ax.errorbar(
            means["rank"].isel(rank=significant),
            means.isel(rank=significant),
            errors.isel(rank=significant),
            **kwargs
            | {
                "mew": 1,
                "alpha": 0.25,
                "mfc": "None",
                "label": "",
            },
        )
    else:
        ax.errorbar(means["rank"], means, errors, **kwargs)


def plot_spectra(
    spectra: xr.Dataset,
    *,
    ax: Axes,
    hue: str,
    hue_order: Sequence[int | str] | None = None,
    hue_reference: int | str | None = None,
    hue_labels: Sequence[str] | None = None,
    marker: str | None = "s",
    palette: str = "crest_r",
    metric: str = "covariance",
    errorbar: tuple[str, float] = ("fold-sd", 1),
    null_quantile: float = 0.99,
    hide_insignificant: bool = False,
) -> None:
    hues = np.unique(spectra[hue].to_numpy()) if hue_order is None else hue_order

    if isinstance(palette, str):
        color_palette = sns.color_palette(palette, n_colors=len(hues))
    else:
        color_palette = palette

    if hide_insignificant:
        significant = spectra[metric].mean("fold") > (
            spectra[f"{metric} (permuted)"]
            .mean("fold")
            .quantile(null_quantile, dim="permutation")
        )

    spectra_offset = offset_spectra(
        spectra[metric].to_dataframe().reset_index(),
        keys=[hue],
    )

    for i_hue, hue_ in enumerate(hues):
        spectrum = (
            spectra_offset.loc[spectra_offset[hue] == hue_]
            .to_xarray()
            .drop_vars("index")
            .set_index({"index": ["fold", "rank"]})
            .unstack("index")
        )
        if hide_insignificant:
            spectrum = spectrum.assign_coords({
                "significant": ("rank", significant.sel({hue: hue_}).to_numpy()),
            })

        kwargs_significant = {
            "ls": "None",
            "c": color_palette[i_hue],
            "marker": ("s" if hue_ == hue_reference else "o")
            if marker is None
            else marker,
            "zorder": 2 if hue_ == hue_reference else 1.99,
            "mew": 0,
            "alpha": 1 if (hue_ == hue_reference) or (hue_reference is None) else 0.75,
            "label": hue_labels[i_hue] if hue_labels is not None else hue_,
        }

        plot_spectrum(
            ax,
            spectrum=spectrum,
            metric=metric,
            errorbar=errorbar,
            hide_insignificant=hide_insignificant,
            **kwargs_significant,
        )

def bin_data(
    data: T,
    *,
    bin_edges: dict[str, npt.NDArray[np.number]],
    bin_centers: dict[str, npt.NDArray[np.number]],
    dim: str,
) -> T:
    dims = list(bin_edges.keys())
    binned = (
        xarray_reduce(
            data,
            *dims,
            func="nanmean",
            expected_groups=tuple(bin_edges.values()),
            isbin=True,
        )
        .assign_coords(
            {
                f"{dim_}_bins": (
                    f"{dim_}_bins",
                    bin_centers[dim_],
                )
                for dim_ in dims
            },
        )
        .stack({dim: [f"{dim_}_bins" for dim_ in dims]})
        .rename({f"{dim_}_bins": dim_ for dim_ in dims})
        .dropna(dim=dim, how="all")
    )
    if len(dims) == 1:
        binned = binned.reset_index(dim).rename({dims[0]: dim}).set_index({dim: [dim]})
    return binned


def extract_uniformly_spaced_bins(
    start: float,
    stop: float,
    *,
    n: int | None = None,
    spacing: float | None = None,
    adjust_endpoints: bool = True,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    if n is None:
        if spacing is None:
            error = "exactly one of `spacings` and `n_bins` must be provided"
            raise ValueError(error)
        n = int(np.ceil((stop - start) / spacing))

    bin_edges = np.linspace(start, stop, n + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    if adjust_endpoints:
        bin_edges = expand_endpoints(bin_edges)

    return bin_edges, bin_centers


def extract_geometrically_spaced_bins(
    start: float,
    stop: float,
    n: int | None = None,
    *,
    base: int = 10,
    density: float | None = None,
    adjust_endpoints: bool = True,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    if n is None:
        if density is None:
            error = "exactly one of `n` and `density` must be provided."
            raise ValueError(error)

        n_levels = (np.log(stop) - np.log(start)) / np.log(base)
        n = int(density * n_levels)
    else:
        n += 1

    bin_edges = np.geomspace(start, stop, num=n)
    bin_centers = np.exp(np.log(bin_edges)[:-1] + np.diff(np.log(bin_edges)) / 2)

    if adjust_endpoints:
        bin_edges = expand_endpoints(bin_edges)

    return bin_edges, bin_centers


def expand_endpoints(
    bin_edges: npt.NDArray[np.floating],
    *,
    factor: float = 0.001,
) -> npt.NDArray[np.floating]:
    bin_edges_expanded = bin_edges
    bin_edges_expanded[0] *= 1 - factor
    bin_edges_expanded[-1] *= 1 + factor
    return bin_edges_expanded


def assign_bins(
    data: npt.NDArray[np.floating],
    *,
    bin_edges: npt.NDArray[np.floating],
    bin_centers: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return bin_centers[np.digitize(data, bin_edges) - 1]