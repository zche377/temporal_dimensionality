"""Offline extraction of Doerig BLT-vNet RCNN features for THINGS stimulus images.

Runs in a *dedicated TensorFlow env* (this repository's environment is PyTorch-only and has no TF).
For each THINGS image it extracts the **pre-readout final conv layer, last timestep,
spatial-averaged -> 512-d** activation — exactly the representation Doerig et al. (2025) used for
all their brain-alignment results (Fig 4e, "512 features"). Output is a per-dataset `.npz`
consumed by `lib.datasets.doerig_rcnn.load_llm_rcnn_embeddings`.

Setup (run once, outside this repo's env):
  1. Create a TF env (TF 2.x compatible with the Dec-2023 Keras HDF5 weights), e.g.
       conda create -n blt_tf python=3.10 && conda activate blt_tf
       pip install "tensorflow==2.15.*" pillow numpy h5py
  2. git clone https://github.com/adriendoerig/blt_mpnet
  3. Download the seed1 weights folder `blt_vNet_half_channels_mpnet_Dec23_seed1` from
       https://natural-scenes-dataset.s3.amazonaws.com/index.html#other/blt_mpnet_weights/
     (the folder ships hparams.pickle + _code_used_for_training/ + training_checkpoints/).

Run:
  # cheap alignment pre-check FIRST (seconds): extract a handful, print their img_files basenames,
  # then compare to the neural pipeline's img_files (in the main env) before the full run:
  python extract_llm_rcnn_features.py --dataset things_eeg_2 --limit 8 \
      --model-path .../blt_vNet_half_channels_mpnet_Dec23_seed1 --blt-mpnet-dir .../blt_mpnet

  # full extraction (prefer a GPU; ~16.7k eeg2 / ~26k meg images):
  python extract_llm_rcnn_features.py --dataset things_eeg_2 \
      --model-path .../blt_vNet_half_channels_mpnet_Dec23_seed1 --blt-mpnet-dir .../blt_mpnet
  python extract_llm_rcnn_features.py --dataset things_meg  ... (same flags)

Alignment check in the MAIN env after extraction (should print True):
  from lib.datasets.doerig_rcnn import load_llm_rcnn_embeddings
  from lib.datasets._loader import load_dataset
  f = set(load_llm_rcnn_embeddings("things_eeg_2").img_files.values)
  n = set(load_dataset("things_eeg_2_train", subjects=1).img_files.values)
  print(n.issubset(f))   # every neural img must have an extracted feature
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

# THINGS stimulus image roots, relative to BONNER_DATASETS_HOME. Feature is keyed by the image
# BASENAME (e.g. "aircraft_carrier_06s.jpg"), which is exactly the neural pipeline's img_files coord.
_IMAGE_DIRS = {
    "things_eeg_2": [
        "gifford2022.things_eeg_2/images/training_images",
        "gifford2022.things_eeg_2/images/test_images",
    ],
    "things_meg": [
        "hebart2023.things_meg/images/object_images",
    ],
}

# short dataset name -> dataset id used in the `data/dataset=...` cache dirs (must match the loader)
_DATASET_ID = {
    "things_eeg_2": "gifford2022.things_eeg_2",
    "things_meg": "hebart2023.things_meg",
}

FEATURE_TAG = "blt_vNet_mpnet_seed1_lastlayer"  # must match lib.datasets.doerig_rcnn.FEATURE_TAG


def enumerate_images(dataset, datasets_home):
    """Return (paths, basenames) for every .jpg under the dataset's image roots, basename-sorted."""
    paths = []
    for root in _IMAGE_DIRS[dataset]:
        paths.extend(glob.glob(os.path.join(datasets_home, root, "**", "*.jpg"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .jpg images for {dataset} under {datasets_home}")
    by_base = {}
    for p in paths:
        base = os.path.basename(p)
        if base in by_base and by_base[base] != p:
            raise RuntimeError(f"Duplicate image basename {base!r}: {by_base[base]} vs {p}")
        by_base[base] = p
    basenames = sorted(by_base)
    return [by_base[b] for b in basenames], basenames


def load_batch(paths, image_size):
    """Load + preprocess a list of image paths exactly as blt_mpnet/example.py: RGB, resize, [-1,1]."""
    imgs = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize((image_size, image_size))
        img = np.asarray(img).astype(np.float32)
        img = (img / 127.5) - 1.0
        imgs.append(img)
    return np.stack(imgs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(_IMAGE_DIRS))
    ap.add_argument("--model-path", required=True, help="path to blt_vNet_half_channels_mpnet_Dec23_seed1")
    ap.add_argument("--blt-mpnet-dir", required=True, help="path to the cloned blt_mpnet repo")
    ap.add_argument("--datasets-home", default=os.environ.get("BONNER_DATASETS_HOME"),
                    help="root holding the THINGS image folders (default: $BONNER_DATASETS_HOME)")
    ap.add_argument("--caching-home", default=os.environ.get("BONNER_CACHING_HOME"),
                    help="must equal BONNER_CACHING_HOME used by the analyses (default: $BONNER_CACHING_HOME)")
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-layers", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="extract only the first N images (pre-check)")
    args = ap.parse_args()
    if args.datasets_home is None or args.caching_home is None:
        ap.error("set BONNER_DATASETS_HOME and BONNER_CACHING_HOME, or pass --datasets-home and --caching-home")

    sys.path.insert(0, args.blt_mpnet_dir)
    import tensorflow as tf  # noqa: E402  (import after sys.path/env setup)
    from helpers import load_hparams, load_model_from_path, get_activities_model  # noqa: E402
    from extract_activities import _maybe_average_spatial  # noqa: E402

    hparams = load_hparams(args.model_path)
    image_size = hparams["image_size"]
    hparams["batch_size"] = args.batch_size

    model, hparams = load_model_from_path(
        model_path=args.model_path, epoch=args.epoch, hparams=hparams, n_classes=768,
    )
    # build once: nested [layer][timestep] activity model; we read only the last layer/last timestep
    activities_model, layer_names, _ = get_activities_model(
        model, n_layers=args.n_layers, hparams=hparams, pre_post_norm="post", include_readout=False,
    )
    print(f"last layer/timestep read out: {layer_names[-1][-1]}")

    paths, basenames = enumerate_images(args.dataset, args.datasets_home)
    if args.limit is not None:
        paths, basenames = paths[: args.limit], basenames[: args.limit]
    print(f"[{args.dataset}] {len(paths)} images; image_size={image_size}")

    feats = []
    for i in range(0, len(paths), args.batch_size):
        batch = load_batch(paths[i : i + args.batch_size], image_size)
        raw = activities_model(batch)          # nested [layer][timestep] tensors
        act = raw[-1][-1]                       # last layer, last timestep -> (N, H, W, C)
        feats.append(_maybe_average_spatial(act, average_spatial=True).astype(np.float32))
        if (i // args.batch_size) % 20 == 0:
            print(f"  {i + len(batch)}/{len(paths)}")
    features = np.concatenate(feats, axis=0)    # (n_images, 512)
    assert features.shape[0] == len(basenames)
    if features.shape[1] != 512:
        print(f"WARNING: expected 512 channels for the pre-readout last conv layer, got "
              f"{features.shape[1]} — confirm this is the intended layer.")

    if args.limit is not None:
        print("PRE-CHECK basenames:", basenames)
        print("PRE-CHECK feature shape:", features.shape, "— compare basenames to neural img_files.")
        return

    out_dir = os.path.join(args.caching_home, "data", f"dataset={_DATASET_ID[args.dataset]}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"llm_rcnn_{FEATURE_TAG}.npz")
    np.savez(out_path, features=features, img_files=np.asarray(basenames))
    print(f"Saved {features.shape} to {out_path}")


if __name__ == "__main__":
    main()
