import torch.nn as nn
import xarray as xr

from deepjuice import *
from tqdm import tqdm
from bonner.models.utilities import BONNER_MODELS_HOME
from bonner.caching import cache

from lib.utilities import hash_string
from lib.datasets._loader import load_dataloader, load_metadata
from deepjuice.systemops.devices import cuda_device_report
from deepjuice.reduction import get_jl_lemma, make_srp_matrix, compute_srp


class DJModel:
    def __init__(
        self,
        model_uid: str,
        hook: str = "srp",
        memory_limit: str = "auto",
        nodes: str = "Identity",
        node_subset: bool = True,
        seed: int = 0,
        n_components: int = "auto",
    ):
        self.model_uid = model_uid
        self.model, self.preprocess = get_deepjuice_model(model_uid)
        self.hook = hook
        self.memory_limit = memory_limit
        # if self.memory_limit == 'auto' and self.hook == "srp":
        #     # Calculate the memory limit and generate the feature_extractor
        #     total_memory_string = cuda_device_report(to_pandas=True)[0]['Total Memory']
        #     total_memory = int(float(total_memory_string.split()[0]))
        #     memory_limit = int(total_memory * 0.75)
        #     self.memory_limit = f'{memory_limit}GB'
        self.nodes = nodes
        self.node_subset = node_subset
        self.seed = seed
        self.n_components = n_components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def __call__(
        self,
        dataset: str,
        stimulus_set_kwargs: dict = {},
        dataloader_kwargs: dict = {},
        return_extractor: bool = False,
    ) -> dict[str, xr.DataArray]:
        dataloader = load_dataloader(dataset, self.preprocess, stimulus_set_kwargs, dataloader_kwargs)
        feature_extractor = FeatureExtractor(self.model, dataloader, flatten=self.hook is None or self.hook!="gm", initial_report=False, memory_limit=self.memory_limit, exclude_oversize=True,)
        if self.node_subset:
            feature_extractor.update_keep([
                feature_extractor.search_uids(self.nodes)[0], 
                feature_extractor.search_uids(self.nodes)[-1]
            ])
        else:
            feature_extractor.update_keep(self.nodes)
        if return_extractor:
            return feature_extractor

        hash_str = hash_string('.'.join(feature_extractor.get_keep()))

        metadata = load_metadata(dataset)
        
        def _get_feature_maps_subset(batch):
            hook, hook_kwargs = self._get_hook()
            feature_maps_subset = feature_extractor.get(batch)
            fm_dict = {}
            for node, feature_map in feature_maps_subset.items():
                feature_map = hook(feature_map, **hook_kwargs)
                fm_dict[node] = xr.DataArray(
                    feature_map.cpu().numpy(),
                    dims=("presentation", "neuroid"),
                    coords={c: ("presentation", metadata[c].values) for c in metadata.columns}
                )
                del feature_map
            return fm_dict
        
        feature_maps = {}
        for i, batch in enumerate(feature_extractor.get_batches()):
            cache_path = f"{BONNER_MODELS_HOME}/dataset={dataset}/model={self.model_uid}.nodes={self.nodes}.hook={self.hook}.hash={hash_str}.batch={i}"
            if self.hook=="srp":
                cache_path += f".seed={self.seed}.n_components={self.n_components}"
            cache_path += ".pkl"
            feature_maps.update(cache(cache_path)(_get_feature_maps_subset)(batch))
            
        return feature_maps
                
    def get_metadata(self):
        dataloader = get_data_loader([f"{os.getenv("PROJECT_HOME")}/src/lib/models/example_stimuli/bnv.jpg" for i in range(3)], self.preprocess)
        feature_extractor = FeatureExtractor(self.model, dataloader, flatten=self.hook!="gm", initial_report=False, memory_limit=self.memory_limit, exclude_oversize=True)
        feature_extractor.update_keep(self.nodes)
        metadata = get_feature_map_metadata(self.model, dataloader, input_dim=0)
        return metadata[metadata.output_uid.isin(feature_extractor.get_batch_metadata().uid)]
            
    def _get_hook(self):
        if self.hook is None:
            return nn.Identity(), {}
        match self.hook:
            case "srp":
                # TEMP
                hook_kwargs = {"seed": self.seed, "use_sparse_dot": False, "device": "cpu"}
                if isinstance(self.n_components, int):
                    hook_kwargs["n_components"] = self.n_components
                return sparse_random_projection, hook_kwargs
            case "gm":
                return _global_maxpool, {}

# TODO: now naively assume max over the last two dims
def _global_maxpool(x):
    assert x.ndim == 4
    return x.amax(dim=[-2, -1])
