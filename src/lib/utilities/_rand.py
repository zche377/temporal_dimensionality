import torch



def _rand_orthonormal(shape0, shape1):
    return torch.linalg.qr(torch.randn(shape0, shape1))[0]