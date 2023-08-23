import torch
import torch.nn.functional as F


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def loss(h1: torch.Tensor, h2: torch.Tensor, decoded: torch.Tensor, x: torch.Tensor, mean: bool = True):
    l1 = semi_loss(h1, h2)  # (13752，)
    l2 = semi_loss(h2, h1)  # (13752，)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    mse_fun = torch.nn.MSELoss()
    mse_loss = mse_fun(decoded, x)

    return ret + mse_loss
