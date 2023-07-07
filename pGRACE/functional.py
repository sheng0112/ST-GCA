import torch
from torch_geometric.utils import degree, to_undirected


def degree_drop_weights(edge_index):  # 计算边权重
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1].long())  # (13752,)
    deg_col = deg[edge_index[1].long()].to(torch.float32)  # (491722,)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def feature_drop_weights(x, node_c):  # 计算节点特征权重
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):  # 筛选重要的边
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    # 实现了一个蒙特卡罗抽样，根据每条边的权重，返回一个相等概率的布尔掩码，即有一定概率选中该边
    return edge_index[:, sel_mask]


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x
