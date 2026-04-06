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
           
def load_presentation_reshaped_data(
    dataset: str, 
    split: bool = True,
    split_dim: str = "presentation",
    stack: bool = False,
    stack_dims: list[str] = ["target_var", "presentation"],
    n_splits: int = 2, 
    n_seeds: int = 1, 
    average: bool = False,
    **kwargs: Any
):
    assert split_dim == "presentation"
    X = load_dataset(dataset, **kwargs,)
    
    if split_dim == "target_var":
        split_dim = load_target_var(dataset)
    if stack_dims[0] == "target_var":
        stack_dims[0] = load_target_var(dataset)
    if stack_dims[1] == "target_var":
        stack_dims[1] = load_target_var(dataset)
    
    if split:
        seed_Xs = []
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
        
    if average:
        assert not stack or "presentation" not in stack_dims
        X = X.mean("presentation")
    
    return X

            
