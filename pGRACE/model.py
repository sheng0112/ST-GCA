import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from pGRACE.utils import get_activation


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
        """
        for ld in range(1, len(self.linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate,
                                                 self.p_drop))
            current_decoder_dim = self.linear_decoder_hidden[ld]
        """
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}',
                                buildNetwork(self.linear_decoder_hidden[-1],
                                             self.input_dim, "sigmoid", self.p_drop))

    def forward(self, x):
        feat = self.encoder(x)
        return feat

    def forward_d(self, x):
        feat = self.decoder(x)
        return feat


class GCA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_hidden: int,
                 num_proj_hidden: int,
                 activation=get_activation('prelu'),
                 base_model=GCNConv,
                 k: int = 2,
                 skip=False):
        super(GCA, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, num_hidden).jittable()]  # (767,256) ,jittable()，加速张量计算
            for _ in range(1, k - 1):
                self.conv.append(base_model(num_hidden, num_hidden))
            self.conv.append(base_model(num_hidden, num_proj_hidden))  # (256,128)
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, num_hidden)
            self.conv = [base_model(in_channels, num_hidden)]
            for _ in range(1, k):
                self.conv.append(base_model(num_hidden, num_proj_hidden))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

        self.fc1 = torch.nn.Linear(num_proj_hidden,
                                   num_proj_hidden)  # Linear(in_features = 128, out_features = 128, bias = True)
        self.fc2 = torch.nn.Linear(num_proj_hidden,
                                   num_proj_hidden)  # Linear(in_features = 128, out_features = 128, bias = True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index.to(torch.int64)))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):  # in_size=32
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):  # z:(3639,2,32)
        w = self.project(z)
        beta = torch.softmax(w, dim=1)  # (3639,2,1)
        return (beta * z).sum(1), beta


# feat.shape[1], args.num_en, args.num_de, args.num_hidden, args.num_proj, feat.shape[1]
class GRACE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_en,
                 num_de,
                 num_hidden,
                 num_proj,
                 ):
        super(GRACE, self).__init__()
        self.encoder = Encoder(input_dim, num_en, num_de)
        self.gca = GCA(num_en[-1], num_hidden, num_proj)
        self.attention = Attention(num_proj)

    def forward(self, x: torch.Tensor, edge_index_1: torch.Tensor, edge_index_2: torch.Tensor):
        z = self.encoder(x)
        emb1 = self.gca(z, edge_index_1)
        emb2 = self.gca(z, edge_index_2)
        h1 = self.gca.projection(emb1)
        h2 = self.gca.projection(emb2)
        emb = torch.stack([emb1, emb2], dim=1)
        emb, _ = self.attention(emb)
        de_z = self.encoder.forward_d(emb)
        return h1, h2, emb, de_z


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
