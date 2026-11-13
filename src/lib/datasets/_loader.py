from typing import Any

import numpy as np
import torch.utils
import torch.utils.data
import xarray as xr
import torch

from bonner.computation.decomposition import PCA
from lib.utilities import SEED
from lib.datasets import StimulusSetPreprocessWrapper


        
def load_dataset(dataset: str, **kwargs: Any) -> xr.DataArray:
    match dataset:
        case "things_eeg_main":
            from lib.datasets.grootswagers2022_things_eeg import load_main_data
            X = load_main_data(**kwargs)
        case "things_eeg_validation":
            from lib.datasets.grootswagers2022_things_eeg import load_validation_data
            X = load_validation_data(**kwargs)
        case "things_eeg_2_train":
            from lib.datasets.gifford2022_things_eeg_2 import load_train_data
            X = load_train_data(**kwargs)
        case "things_eeg_2_test":
            from lib.datasets.gifford2022_things_eeg_2 import load_test_data
            X = load_test_data(**kwargs)
        case "things_meg_train":
            from lib.datasets.hebart2023_things_meg import load_train_data
            X = load_train_data(**kwargs)
        case "things_meg_test":
            from lib.datasets.hebart2023_things_meg import load_test_data
            X = load_test_data(**kwargs)
    return X

def load_stimulus_set(dataset: str, **kwargs: Any):
    match dataset:
        case "things_eeg_2_train":
            from lib.datasets.gifford2022_things_eeg_2 import load_stimulus_set
            return load_stimulus_set("train", dataset)
        case "things_eeg_2_test":
            from lib.datasets.gifford2022_things_eeg_2 import load_stimulus_set
            return load_stimulus_set("test", dataset)
        case "things_eeg_2":
            from lib.datasets.gifford2022_things_eeg_2 import load_stimulus_set
            return load_stimulus_set("all", dataset)
        case "things_meg_train":
            from lib.datasets.hebart2023_things_meg import load_stimulus_set
            return load_stimulus_set("train", dataset)
        case "things_meg_test":
            from lib.datasets.hebart2023_things_meg import load_stimulus_set
            return load_stimulus_set("test", dataset)
        case "things_meg":
            from lib.datasets.hebart2023_things_meg import load_stimulus_set
            return load_stimulus_set("all", dataset)
        
def load_metadata(dataset: str, **kwargs: Any):
    match dataset:
        case "things_eeg_2_train":
            from bonner.datasets.gifford2022_things_eeg_2 import load_metadata
            return load_metadata("train")
        case "things_eeg_2_test":
            from bonner.datasets.gifford2022_things_eeg_2 import load_metadata
            return load_metadata("test")
        case "things_eeg_2":
            from bonner.datasets.gifford2022_things_eeg_2 import load_metadata
            return load_metadata("all")
        case "things_meg_train":
            from bonner.datasets.hebart2023_things_meg import load_metadata
            return load_metadata("train")
        case "things_meg_test":
            from bonner.datasets.hebart2023_things_meg import load_metadata
            return load_metadata("test")
        case "things_meg":
            from bonner.datasets.hebart2023_things_meg import load_metadata
            return load_metadata("all")
        
def load_dataloader(dataset: str, preprocess=torch.nn.Identity(), stimulus_set_kwargs={}, dataloader_kwargs={}):
    stimulus_set =  StimulusSetPreprocessWrapper(load_stimulus_set(dataset, **stimulus_set_kwargs), preprocess=preprocess)
    return torch.utils.data.DataLoader(stimulus_set, **dataloader_kwargs)
        
def load_n_subjects(dataset: str) -> int:
    match dataset:
        case ("things_eeg_main" | "things_eeg_validation"):
            from bonner.datasets.grootswagers2022_things_eeg import N_SUBJECTS
            return N_SUBJECTS
        case ("things_eeg_2" | "things_eeg_2_train" | "things_eeg_2_test"):
            from bonner.datasets.gifford2022_things_eeg_2 import N_SUBJECTS
            return N_SUBJECTS
        case ("things_meg" | "things_meg_train" | "things_meg_test"):
            from bonner.datasets.hebart2023_things_meg import N_SUBJECTS
            return N_SUBJECTS
        
def load_target_var(dataset: str) -> str:
    match dataset:
        case _:
            return "object"


def per_image_rep_split(
    X: xr.DataArray,
    object_dim: str,
    presentation_dim: str = "presentation",
    n_splits: int = 2,
    seed: int = 0,
) -> list[xr.DataArray]:
    """Stringer-style PER-IMAGE repetition split.

    For each object, its non-NaN presentations (reps) are randomly partitioned into `n_splits`
    groups of ``floor(n_valid / n_splits)`` reps each (any remainder is dropped, matching the
    global-split path's ``len // n_splits``). Returns `n_splits` **full-shape masked** copies of X:
    in copy `s`, every rep NOT assigned to group `s` (and every NaN pad) is set to NaN, so a
    downstream ``.mean(presentation)`` (skipna) averages exactly that object's rep-half.

    Masked (not sliced) so the arrays stay rectangular and concat-able along a new "split" axis, and
    so ragged/NaN-padded MEG works (each object keeps ``floor(valid/2)`` of its OWN reps rather than a
    shared global slice). The split is identical across every other dim (subject/channel/time).
    """
    reduce_dims = [d for d in X.dims if d not in (object_dim, presentation_dim)]
    valid = X.notnull().any(dim=reduce_dims).transpose(object_dim, presentation_dim)
    valid_np = valid.values  # (n_object, n_presentation) bool
    n_obj, n_pres = valid_np.shape

    valid_counts = valid_np.sum(axis=1)
    assert (valid_counts >= n_splits).all(), (
        f"per_image_rep_split: {int((valid_counts < n_splits).sum())} object(s) have < {n_splits} "
        f"valid reps (min={int(valid_counts.min())}); a split-half would be empty/all-NaN."
    )

    # -1 = dropped (invalid pad, or the discarded remainder); 0..n_splits-1 = group assignment.
    assign = np.full((n_obj, n_pres), -1, dtype=np.int64)
    for o in range(n_obj):
        idx = np.nonzero(valid_np[o])[0]
        # per-(seed, object) independent stream -> reproducible + order-independent across objects.
        perm = np.random.default_rng([seed, o]).permutation(idx)
        npps = len(idx) // n_splits
        for s in range(n_splits):
            assign[o, perm[s * npps:(s + 1) * npps]] = s

    assign_da = xr.DataArray(
        assign,
        dims=(object_dim, presentation_dim),
        coords={
            object_dim: valid.coords[object_dim],
            presentation_dim: valid.coords[presentation_dim],
        },
    )
    return [X.where(assign_da == s) for s in range(n_splits)]


def load_presentation_reshaped_data(
    dataset: str, 
    split: bool = True,
    split_dim: str = "presentation",
    stack: bool = False,
    stack_dims: list[str] = ["target_var", "presentation"],
    n_splits: int = 2,
    n_seeds: int = 1,
    per_image: bool = False,
    average: bool = False,
    presentation_indices: list[int] = None,
    **kwargs: Any
):
    assert split_dim == "presentation"
    X = load_dataset(dataset, **kwargs,)

    if presentation_indices is not None:
        # subset the presentation axis BEFORE any split (e.g. session-subset of test data)
        X = X.isel(presentation=presentation_indices)

    if split_dim == "target_var":
        split_dim = load_target_var(dataset)
    if stack_dims[0] == "target_var":
        stack_dims[0] = load_target_var(dataset)
    if stack_dims[1] == "target_var":
        stack_dims[1] = load_target_var(dataset)

    averaged_per_seed = False
    if split:
        seed_Xs = []
        if per_image:
            # Stringer per-image half-rep split (ragged-MEG-safe). Same layout as the global path
            # (seed x split x ...), but each object is split among its OWN valid reps.
            object_dim = load_target_var(dataset)
            # Each masked copy is full-presentation, so holding all 2 * n_seeds of them until the final
            # mean scales host RAM with n_seeds. With averaging requested and no stack, reduce each copy
            # before the concat: the masks share X's coordinates, so the same values are averaged.
            averaged_per_seed = average and not stack
            for seed in range(n_seeds):
                split_Xs = per_image_rep_split(
                    X, object_dim=object_dim, presentation_dim=split_dim,
                    n_splits=n_splits, seed=seed,
                )
                if averaged_per_seed:
                    split_Xs = [split_X.mean("presentation") for split_X in split_Xs]
                seed_Xs.append(xr.concat(split_Xs, dim="split"))
                del split_Xs
        else:
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                len_split_dim = len(X[split_dim].values)
                split_index = rng.choice(np.arange(len_split_dim), size=len_split_dim, replace=False)
                npps = len_split_dim // n_splits
                split_Xs = []
                for i in range(n_splits):
                    split_Xs.append(X.sel({
                        split_dim: split_index[i*npps: (i+1)*npps]
                    }))
                seed_Xs.append(xr.concat(split_Xs, dim="split"))
        X = xr.concat(seed_Xs, dim="seed")
        del seed_Xs
    
    if stack:
        assert not average or "presentation" not in stack_dims
        X = (X
            .stack(stack=stack_dims)
            .reset_index("stack")
            .drop_vars(stack_dims[1])
            .rename({"stack": stack_dims[1]})
        )
        
    if average and not averaged_per_seed:
        assert not stack or "presentation" not in stack_dims
        X = X.mean("presentation")

    return X

            
