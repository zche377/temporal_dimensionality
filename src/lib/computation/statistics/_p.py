import numpy as np
import torch
import logging
logging.basicConfig(level=logging.INFO)



def compute_p(true, null, H0):
    isnumpy = isinstance(true, np.ndarray)
    if isnumpy:
        true = torch.tensor(true).to("cuda")
        null = torch.tensor(null).to("cuda")
    assert null.shape[1:] == true.shape
    n_perm = null.shape[0]
    denominator = n_perm + 1
    match H0:
        case 'two_tailed':
            p = (torch.sum(torch.abs(null) >= torch.abs(true), dim=0) + 1) / denominator
        case 'greater':
            p = 1 - ((torch.sum(true > null, dim=0) + 1) / denominator)
        case 'less':
            p = 1 - ((torch.sum(true < null, dim=0) + 1) / denominator)
    if isnumpy:
        p = p.cpu().numpy()
    return p