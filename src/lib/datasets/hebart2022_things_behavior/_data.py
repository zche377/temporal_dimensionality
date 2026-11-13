from bonner.datasets.hebart2022_things_behavior import load_embeddings, load_triplet_results, load_object_labels

import itertools
from tqdm.auto import tqdm
import numpy as np
import xarray as xr
from bonner.caching import cache



def sort_embeddings(sortby: str = "std", ntop: int = 8):
    embd = load_embeddings()
    match sortby:
        case "std":
            b = embd.behavior.sortby(embd.std("object"), ascending=False).values
    if ntop is not None:
        return embd.sel(behavior=b[:ntop])
    else:
        return embd.sel(behavior=b)

@cache("data/dataset=hebart2022.things.behavior/llm_mpnet_embedding.nc")
def load_llm_embeddings():
    """LLM (all-mpnet-base-v2) embedding of each THINGS concept word.

    No per-image captions exist for THINGS, so we embed the bare concept word (the
    ``unique_id`` with underscores replaced by spaces, e.g. ``air_conditioner`` ->
    "air conditioner"); the same vector applies to every image in a concept. Layout
    mirrors ``load_embeddings()`` exactly -- dims ``(object, behavior)`` indexed by the
    same ``unique_id`` ordering -- so it drops into the behavior/everything decoding
    paths unchanged.
    """
    from sentence_transformers import SentenceTransformer

    object = load_object_labels()
    text = [str(label).replace("_", " ") for label in object]
    model = SentenceTransformer("all-mpnet-base-v2")
    embd = model.encode(text, convert_to_numpy=True, show_progress_bar=True)

    return xr.DataArray(
        embd,
        dims=("object", "behavior"),
        coords={"object": object, "behavior": np.arange(embd.shape[1])},
    )


@cache("data/dataset=hebart2022.things.behavior/triplet_rdm.nc")
def load_triplet_rdm():
    df = load_triplet_results()
    n_image = df.image1.nunique()
    rdm = np.zeros((n_image, n_image), dtype=np.float32)
    count = np.zeros((n_image, n_image), dtype=np.int32)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        images = [row.image1, row.image2, row.image3]
        cells = list(itertools.permutations(images, 2))
        image_choice = images[row.choice-1]
        for cell in cells:
            i1, i2 = cell
            count[i1-1, i2-1] += 1
            if i1 == image_choice or i2 == image_choice:
                rdm[i1-1, i2-1] += 1
    rdm = rdm / count
    object = load_object_labels()
    return xr.DataArray(
        np.nan_to_num(rdm, nan=0.0),
        dims=("object0", "object1"),
        coords={"object0": object, "object1": object},
    )