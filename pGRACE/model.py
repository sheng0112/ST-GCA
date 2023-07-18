import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

from torchvision import models
from torch.nn.modules.module import Module
from torch.nn.modules.activation import PReLU
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, to_undirected

from pGRACE.functional import drop_edge_weighted, drop_feature_weighted_2, degree_drop_weights, feature_drop_weights
from pGRACE.utils import get_edges, edge_c


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=PReLU, base_model=GCNConv, k: int = 2,
                 skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]  # (767,256) ,jittable()，加速张量计算
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))  # (256,128)
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class simCLR_model(Module):
    def __init__(self, feature_dim=64):
        super(simCLR_model, self).__init__()

        self.f = []

        # load resnet50 structure
        for name, module in models.resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 64, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)

        return F.normalize(out, dim=-1)


class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, simCLR: simCLR_model, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.simCLR: simCLR_model = simCLR
        self.tau: float = tau  # 0.2

        self.projector = nn.Sequential(
            nn.Linear(num_proj_hidden, num_proj_hidden),
            nn.ReLU(),
            nn.Linear(num_proj_hidden, num_proj_hidden),
        )

        self.num_hidden = num_hidden  # 128

    def forward_gene(self, x, edges):
        return self.encoder(x, edges)

    def forward_image(self, x):
        return self.simCLR(x)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)  # (13752,128)
        h2 = self.projection(z2)  # (13752,128)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)  # (13752，)
            l2 = self.semi_loss(h2, h1)  # (13752，)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class train_model:
    def __init__(self, args, network, optimizer, device):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.device = device

    def train(self, trainloader, epoch):
        # with tqdm(total=len(trainloader)) as t:
        self.network.train()
        image_loss = 0
        gene_loss = 0
        total_loss = 0
        train_cnt = 0

        # train_bar = tqdm(trainloader)

        # for i, batch in enumerate(trainloader):
        for features, image_1, image_2, spatial, idx in trainloader:
            # t.set_description(f'Epoch {epoch} train')

            # features, image_1, image_2, spatial, _ = batch

            edges = get_edges(trainloader.dataset.graph, idx, flag=1).t()

            # edges = get_edges(trainloader.dataset.graph).t()
            self.drop_weights = degree_drop_weights(edges).to(self.device)

            # edge_index_ = to_undirected(edges)
            # node_deg = degree(edge_index_[1].long())
            # self.feature_weights = feature_drop_weights(features, node_c=node_deg).to(self.device)

            edge_index_1 = drop_edge_weighted(edges, self.drop_weights, self.args.drop_edge_rate_1,
                                              threshold=0.7)
            edge_index_2 = drop_edge_weighted(edges, self.drop_weights, self.args.drop_edge_rate_2,
                                              threshold=0.7)
            edge_index_1 = edge_c(edge_index_1)
            edge_index_2 = edge_c(edge_index_2)

            # x_1 = drop_feature_weighted_2(features, self.feature_weights, self.args.drop_feature_rate_1)
            # x_2 = drop_feature_weighted_2(features, self.feature_weights, self.args.drop_feature_rate_2)

            z1 = self.network.encoder(features, edge_index_1.to(torch.int64))
            z2 = self.network.encoder(features, edge_index_2.to(torch.int64))

            loss_g = self.network.loss(z1, z2)

            i_1 = self.network.simCLR(image_1)
            i_2 = self.network.simCLR(image_2)

            o1 = self.network.projection(i_1)
            o2 = self.network.projection(i_2)

            out = torch.cat([o1, o2], dim=0)
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size_I,
                                                            device=sim_matrix.device)).bool()
            sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size_I, -1)

            pos_sim = torch.exp(torch.sum(o1 * o2, dim=-1) / self.args.temperature)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

            loss_i = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            loss = loss_i + loss_g

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            image_loss += loss_i.item()
            gene_loss += loss_g.item()
            total_loss += loss
            train_cnt += 1
            #print("=======total_loss========")
            #print(total_loss)

        return total_loss / train_cnt

    def fit(self, trainloader):
        self.network = self.network.to(self.device)

        for epoch in range(self.args.num_epochs):
            loss = self.train(trainloader, epoch + 1)
            print("====epoch loss====")
            print(loss)
