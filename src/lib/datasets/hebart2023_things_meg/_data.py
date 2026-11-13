import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
import xarray as xr

from lib.utilities import _append_path

from bonner.datasets.hebart2023_things_meg import (
    IDENTIFIER,
    N_SUBJECTS,
    load_preprocessed_data,
    StimulusSet
)
from bonner.caching import cache



def _subjects(subjects: (int | list[int] | str) = "all") -> list[int]:
    if subjects == "all":
        subjects = list(range(1, N_SUBJECTS + 1))
    if isinstance(subjects, int):
        subjects = [subjects]
    return subjects

def _reshape_data(
    data: xr.DataArray,
    var: str
) -> xr.DataArray:
    grouped = data.groupby(var)
    grouped_list = []
    for _, g in grouped:
        coords_dict = {coord: g.coords[coord][0].values.item() for coord in g.coords if g.coords[coord].dims[0]==var}
        g = g.drop_vars(list(coords_dict.keys())).rename({var: "presentation"})
        # g = g.assign_coords(presentation=np.arange(len(g.presentation)))
        v = coords_dict.pop(var)
        g = g.expand_dims(dim={var: [v]})
        g = g.assign_coords({
            coord: (var, [coord_v])
            for coord, coord_v in coords_dict.items()
        })
        grouped_list.append(g)
    return xr.concat(grouped_list, dim=var)

def _train_data(
    subject: int,
    **kwargs,
) -> xr.DataArray:
    data = load_preprocessed_data(subject, data_type="train", **kwargs,)
    return xr.concat([data], "presentation").assign_coords({"subject": subject,})

def load_train_data(
    subjects: (int | list[int] | str) = "all",
    exclude_test_concepts: int = 0,
    **load_dataset_kwargs,
) -> xr.DataArray:
    """Load the MEG train set.

    exclude_test_concepts: drop every train image whose concept (`object`) appears in the test set, so
    train and test concepts are disjoint (concept-level held-out, as in THINGS-EEG2). The data-layer cache
    is keyed without it, so the full train set's cache is reused and the images are dropped after loading.
    """
    data_list = []
    cache_path = f"data/dataset={IDENTIFIER}/type=train"
    cache_path = _append_path(cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    for subject in _subjects(subjects):
        data = cache(f"{cache_path}/subject={subject}.nc")(
            _train_data
        )(
            subject, **load_dataset_kwargs
        )
        if exclude_test_concepts:
            test_concepts = load_test_data(subjects=subject, **load_dataset_kwargs)["object"].values
            data = data.isel(object=~np.isin(data["object"].values, test_concepts))
        data_list.append(data.sortby("img_files"))
    return xr.concat(data_list, dim="subject")


# TODO: add option to only load certain channels (e.g., occipital)

def _test_data(
    subject: int,
    **kwargs,
) -> xr.DataArray:
    data = load_preprocessed_data(subject, data_type="test", **kwargs,)
    return _reshape_data(data, "object").assign_coords({"subject": subject,})

def load_test_data(
    subjects: (int | list[int] | str) = "all",
    exclude_test_concepts: int = 0,
    **load_dataset_kwargs,
) -> xr.DataArray:
    """Load the MEG test set. `exclude_test_concepts` only changes the train set; it is accepted (and
    left out of the cache key) so one load_dataset_kwargs dict serves both splits."""
    data_list = []
    cache_path = f"data/dataset={IDENTIFIER}/type=test"
    cache_path = _append_path(cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    for subject in _subjects(subjects):
        data = cache(f"{cache_path}/subject={subject}.nc")(
            _test_data
        )(
            subject, **load_dataset_kwargs
        )
        data_list.append(data)
    return xr.concat(data_list, dim="subject")

def load_stimulus_set(
    data_type: str,
    dataset: str,
):
    stimulus_set = StimulusSet(data_type)
    stimulus_set.identifier = dataset
    return stimulus_set
 