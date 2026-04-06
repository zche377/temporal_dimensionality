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