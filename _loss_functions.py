from torch import Tensor

def gaussian_fidelity(noisy_img: Tensor, net_img: Tensor, args: dict = {}) -> dict:
    return { 'loss': (noisy_img - net_img).square().mean() }

def rician_fidelity(noisy_img: Tensor, net_img: Tensor, args: dict) -> dict:
    from torch import log
    from torch.special import i0
    std = args['std']
    return { 'loss': ((net_img.square() / (2 * std ** 2)) \
        - log(i0((noisy_img.mul(net_img)) / std ** 2))).mean() }