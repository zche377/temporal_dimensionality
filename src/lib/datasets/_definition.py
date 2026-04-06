import torch
from torch.utils.data import MapDataPipe

class StimulusSetPreprocessWrapper(MapDataPipe):
    def __init__(
        self, 
        stimulus_set: MapDataPipe,
        preprocess = torch.nn.Identity(),
    ) -> None:
        self.stimulus_set = stimulus_set
        self.preprocess = preprocess
        
    def __getitem__(self, idx: int):
        return self.preprocess(self.stimulus_set.__getitem__(idx))

    def __len__(self) -> int:
        return self.stimulus_set.__len__()