"""Per-image features from Doerig et al. (2025) LLM-trained BLT-vNet RCNN.

The "language-aligned representation" of Doerig et al. (2025, Nat. Mach. Intell.) is the
LLM-trained RCNN of their Fig 4 — an *image-computable* network trained to predict MPNet caption
embeddings from images — NOT the MPNet text embeddings of Fig 1-3. We pass each THINGS stimulus
image through the RCNN and take the **pre-readout final conv layer, last timestep, spatial-averaged
-> 512-d** (exactly the representation Doerig used for all brain-alignment results, Fig 4e).

Feature extraction runs offline in TensorFlow (`scripts/extract_llm_rcnn_features.py`), which writes
a per-dataset ``.npz``; this loader wraps it into the per-image DataArray the analyses consume.
"""

import numpy as np
import xarray as xr

from bonner.caching import BONNER_CACHING_HOME

# fixed feature identity: Doerig BLT-vNet, mpnet-trained, seed1, pre-readout last conv layer / last
# timestep. Encoded in the cache filename so a variant swap changes the file (manual invalidation).
FEATURE_TAG = "blt_vNet_mpnet_seed1_lastlayer"

# short dataset name -> the dataset id used in the `data/dataset=...` cache dirs
_DATASET_ID = {
    "things_eeg_2": "gifford2022.things_eeg_2",
    "things_meg": "hebart2023.things_meg",
}


def llm_rcnn_feature_path(dataset: str):
    """On-disk path of the offline-extracted RCNN feature ``.npz`` for a dataset."""
    dataset_id = _DATASET_ID[dataset]
    return (
        BONNER_CACHING_HOME
        / "data"
        / f"dataset={dataset_id}"
        / f"llm_rcnn_{FEATURE_TAG}.npz"
    )


def load_llm_rcnn_embeddings(dataset: str) -> xr.DataArray:
    """RCNN per-image features for a dataset, joinable to neural data by ``img_files``.

    Returns a DataArray with dims ``(presentation, neuroid)`` and an ``img_files`` coord (image
    basename, e.g. ``aircraft_carrier_06s.jpg``) over ``presentation`` — the same layout the model
    SRP features use, so the analyses select with
    ``.sel(presentation=embd.img_files.isin(subject_data.img_files)).sortby("img_files")``.
    """
    path = llm_rcnn_feature_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"RCNN feature cache missing: {path}\n"
            f"Run scripts/extract_llm_rcnn_features.py (TensorFlow env) for dataset={dataset} first."
        )
    npz = np.load(path, allow_pickle=False)
    features = npz["features"].astype("float32")  # (n_images, 512)
    img_files = np.asarray(npz["img_files"]).astype(str)  # (n_images,)
    return xr.DataArray(
        features,
        dims=("presentation", "neuroid"),
        coords={"img_files": ("presentation", img_files)},
    )
