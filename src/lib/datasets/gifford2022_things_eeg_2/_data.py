import logging
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import MapDataPipe, Dataset

from lib.utilities import _append_path

from bonner.datasets.gifford2022_things_eeg_2 import (
    IDENTIFIER,
    N_SUBJECTS,
    load_preprocessed_data,
    load_stimuli,
    StimulusSet,
    load_events_list,
    CACHE_PATH
)
from bonner.caching import cache
from bonner.computation.decomposition import PCA



def _subjects(subjects: (int | list[int] | str) = "all") -> list[int]:
    if subjects == "all":
        subjects = list(range(1, N_SUBJECTS + 1))
    if isinstance(subjects, int):
        subjects = [subjects]
    return subjects


def _load_train_data(
    subject: int,
    **kwargs,
) -> xr.DataArray:
    data = load_preprocessed_data(subject,  **kwargs)
    return data.assign_coords({"subject": subject,})

def load_train_data(
    subjects: (int | list[int] | str) = "all",
    pca: int = None,
    **load_dataset_kwargs,
) -> xr.DataArray:
    # TODO: preproceesing for train needs fixing from the original author
    # assert "from_raw" not in list(load_dataset_kwargs.keys())
    data = []
    cache_path = f"data/dataset={IDENTIFIER}/type=train"
    cache_path = _append_path(cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    for subject in _subjects(subjects):
        temp_data = cache(f"{cache_path}/subject={subject}.nc")(
            _load_train_data
        )(
            subject, **load_dataset_kwargs
        )
        if temp_data.values is not None or not temp_data.isnull():
            data.append(temp_data)
    if len(data) == 0:
        return None
    data = xr.concat(data, dim="subject")
    if pca:
        pmodel = PCA()
        pca_data = cache(f"{cache_path}/subject={pca}.nc")(
            _load_train_data
        )(
            pca, **load_dataset_kwargs
        )
        pmodel.fit(torch.tensor(pca_data.mean("presentation").sel(time=.11).values))
        data = data.copy(
            data=pmodel.transform(torch.tensor(data.transpose("subject", "object", "presentation", "time", "neuroid").values)).transpose(-1, -2)
        )
        data["neuroid"] = np.arange(len(data["neuroid"]))+1
    return data

def _load_test_data(
    subject: int,
    **kwargs,
) -> xr.DataArray:
    data = load_preprocessed_data(subject, data_type="test", **kwargs)
    return data.assign_coords({"subject": subject,})

def load_test_data(
    subjects: (int | list[int] | str) = "all",
    pca: int = None, # TEST: use single subject and single time
    **load_dataset_kwargs,
) -> xr.DataArray:
    data = []
    cache_path = f"data/dataset={IDENTIFIER}/type=test"
    cache_path = _append_path(cache_path, "load_dataset_kwargs", load_dataset_kwargs)
    for subject in _subjects(subjects):
        temp_data = cache(f"{cache_path}/subject={subject}.nc")(
            _load_test_data
        )(
            subject, **load_dataset_kwargs
        )
        if temp_data.values is not None or not temp_data.isnull():
            data.append(temp_data)
    if len(data) == 0:
        return None
    data = xr.concat(data, dim="subject")
    if pca:
        pmodel = PCA()
        pca_data = cache(f"{cache_path}/subject={pca}.nc")(
            _load_test_data
        )(
            pca, **load_dataset_kwargs
        )
        pmodel.fit(torch.tensor(pca_data.mean("presentation").sel(time=.11).values))
        data = data.copy(
            data=pmodel.transform(torch.tensor(data.transpose("subject", "object", "presentation", "time", "neuroid").values)).transpose(-1, -2)
        )
        data["neuroid"] = np.arange(len(data["neuroid"]))+1
    return data

def load_stimulus_set(
    data_type: str,
    dataset: str,
):
    stimulus_set = StimulusSet(data_type)
    stimulus_set.identifier = dataset
    return stimulus_set

def create_video_sequence(images, event_order, fps=100):
    """Create video sequence with images and blank screens."""
    img_duration_frames = int(0.1 * fps)
    soa_frames = int(0.2 * fps)
    blank_duration = soa_frames - img_duration_frames
    
    n_sequences = len(event_order) // 20
    frames_per_sequence = soa_frames * 20 + int(2.75 * fps) + int(0.75 * fps)  # stimulus + response + intra
    
    height, width = images.shape[1:3]
    video = torch.ones((n_sequences, frames_per_sequence, 3, height, width), dtype=torch.float32) * 0.625
    
    # Create fixation point masks
    center_x, center_y = width // 2, height // 2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    outer_radius = 15
    outer_mask = ((x - center_x)**2 + (y - center_y)**2 <= outer_radius**2)
    
    cross_thickness = 2
    vertical_mask = (abs(x - center_x) <= cross_thickness) & (abs(y - center_y) <= outer_radius)
    horizontal_mask = (abs(y - center_y) <= cross_thickness) & (abs(x - center_x) <= outer_radius)
    cross_mask = vertical_mask | horizontal_mask
    
    center_radius = 2
    center_mask = ((x - center_x)**2 + (y - center_y)**2 <= center_radius**2)
    
    # Apply fixation to all frames
    video[:, :, 0, outer_mask] = 1.0  # Red outer circle
    video[:, :, 1:, outer_mask] = 0.0
    video[:, :, :, cross_mask] = 0.0  # Black cross
    video[:, :, 0, center_mask] = 1.0  # Red center dot
    video[:, :, 1:, center_mask] = 0.0
    
    for seq in range(n_sequences):
        for img_idx in range(20):
            image_num = event_order[seq * 20 + img_idx]
            if image_num != 99999:
                img = torch.tensor(images[image_num if image_num >= 0 else -1].transpose(2, 0, 1))
            else:
                img = cv2.imread(CACHE_PATH / "images" / "target_from_web.jpg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            
            frame_start = img_idx * soa_frames
            frame_end = frame_start + img_duration_frames
            video[seq, frame_start:frame_end] = img
            
            # Reapply fixation over image
            video[seq, frame_start:frame_end, 0, outer_mask] = 1.0
            video[seq, frame_start:frame_end, 1:, outer_mask] = 0.0
            video[seq, frame_start:frame_end, :, cross_mask] = 0.0
            video[seq, frame_start:frame_end, 0, center_mask] = 1.0
            video[seq, frame_start:frame_end, 1:, center_mask] = 0.0
    
    return video

class StimuliSequence(MapDataPipe):
    def __init__(
        self,
        data_type: str,
        subject: int,
        session: int,
        fps: int = 30,
        transforms = torch.nn.Identity(),
        transformers_image_processor = None
    ) -> None:
        self.data_type = data_type
        self.subject = subject
        self.session = session
        self.fps = fps
        self.transforms = transforms
        self.transformers_image_processor = transformers_image_processor
        self.stimuli = load_stimuli(data_type)
        self.events_list = load_events_list(subject, data_type, exclude_target=False)[session]
        self.n_sequence_per_session = len(self.events_list) // 20
    
    def __len__(self) -> int:
        return self.n_sequence_per_session
    
    def __getitem__(self, index) -> torch.Tensor:
        x = self.transforms(create_video_sequence(
            self.stimuli.values,
            self.events_list[index*20:(index+1)*20],
            fps=self.fps,
        )[0])
        if self.transformers_image_processor is not None:
            x = self.transformers_image_processor(list(x), return_tensors="pt")
        return x
 