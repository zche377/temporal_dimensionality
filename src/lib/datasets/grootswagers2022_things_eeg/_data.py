import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import xarray as xr

from lib.utilities import _append_path

from bonner.datasets.grootswagers2022_things_eeg import (
    IDENTIFIER,
    N_SUBJECTS,
    EXCLUDED_SUBJECTS,
    load_preprocessed_data,
)
from bonner.caching import cache



def _subjects(subjects: (int | list[int] | str) = "all") -> list[int]:
    if subjects == "all":
        subjects = list(range(1, N_SUBJECTS + 1))
    if isinstance(subjects, int):
        subjects = [subjects]
    return list(set(subjects)-set(EXCLUDED_SUBJECTS))

def _reshape_data(
    data: xr.DataArray,
    df: pd.DataFrame,
    var: str
) -> xr.DataArray:
    data = data.assign_coords({
        var: ("presentation", df[var].values),
    })
    grouped = data.groupby(var)
    grouped_list = []
    for _, g in grouped:
        v = g[var].values[0]
        g = g.drop_vars([var, "presentation"]).assign_coords({
            var: v,
        })
        grouped_list.append(g)
    return xr.concat(grouped_list, dim=var)


def _load_main_data(
    subject: int,
    **kwargs,
) -> xr.DataArray:
    data, df = load_preprocessed_data(subject, is_validation=False, **kwargs,)
    return _reshape_data(data, df, "object").assign_coords({"subject": subject,})

def load_main_data(
    subjects: (int | list[int] | str) = "all",
    **load_dataset_kwargs,
) -> xr.DataArray:
    cache_path = f"data/dataset={IDENTIFIER}/type=main"
    cache_path = _append_path(cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    data_list = [
        cache(f"{cache_path}/subject={subject}.nc")(
             _load_main_data
        )(
            subject, **load_dataset_kwargs
        )
        for subject in _subjects(subjects)
    ]
    if len(data_list) == 0:
        return None
    return xr.concat(data_list, dim="subject")


def _load_validation_data(
    subject: int,
    **kwargs,
) -> xr.DataArray:
    data, df = load_preprocessed_data(subject, is_validation=True, **kwargs,)
    if data is None:
        return xr.DataArray(None)
    df["object"] = ['_'.join(s.split('_')[:-1]) for s in df["stimname"].values]
    return _reshape_data(data, df, "object").assign_coords({"subject": subject,})

def load_validation_data(
    subjects: (int | list[int] | str) = "all",
    **load_dataset_kwargs,
) -> xr.DataArray:
    data_list = []
    cache_path = f"data/dataset={IDENTIFIER}/type=validation"
    cache_path = _append_path(cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    for subject in _subjects(subjects):
        data = cache(f"{cache_path}/subject={subject}.nc")(
            _load_validation_data
        )(
            subject, **load_dataset_kwargs
        )
        if data.values is not None or not data.isnull():
            data_list.append(data)
    if len(data_list) == 0:
        return None
    return xr.concat(data_list, dim="subject")
 