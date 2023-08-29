import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import DeepGraphInfomax

from pGRACE.layers import GraphConvolution
from pGRACE.utils import corruption


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 linear_encoder_hidden,
                 linear_decoder_hidden,
                 activate="relu",
                 p_drop=0.01):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop

        current_encoder_dim = self.input_dim
        self.encoder = nn.Sequential()
        for le in range(len(self.linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}',
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate,
                                                 self.p_drop))
            current_encoder_dim = self.linear_encoder_hidden[le]

        current_decoder_dim = linear_decoder_hidden[0]
        self.decoder = nn.Sequential()

        for ld in range(1, len(self.linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate,
                                                 self.p_drop))
            current_decoder_dim = self.linear_decoder_hidden[ld]

        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}',
                                buildNetwork(self.linear_decoder_hidden[-1],
                                             self.input_dim, "sigmoid", self.p_drop))

    def forward(self, x):
        feat = self.encoder(x)
        return feat

    def forward_d(self, x):
        feat = self.decoder(x)
        return feat


class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class GRACE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_en,
                 num_de,
                 num_hidden,
                 ):
        super(GRACE, self).__init__()
        self.encoder = Encoder(input_dim, num_en, num_de)
        self.DGI_model = DeepGraphInfomax(
            hidden_channels=num_hidden, encoder=GCN(num_en[-1], num_hidden),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption)
        self.attention = Attention(num_hidden)

    def forward(self, feat, fadj, sadj):
        z = self.encoder(feat)
        pos_1, neg_1, summary_1 = self.DGI_model(z, fadj)
        pos_2, neg_2, summary_2 = self.DGI_model(z, sadj)
        emb1 = self.attention(torch.stack([pos_1, neg_1], dim=1))
        emb2 = self.attention(torch.stack([pos_2, neg_2], dim=1))
        emb = (emb1 + emb2) / 2
        de_z = self.encoder.forward_d(emb)
        return pos_1, neg_1, summary_1, pos_2, neg_2, summary_2, emb, de_z


def buildNetwork(
        in_features,
        out_features,
        activate="relu",
        p_drop=0.0
):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001))
    if activate == "relu":
        net.append(nn.ELU())
    elif activate == "sigmoid":
        net.append(nn.Sigmoid())
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net)
