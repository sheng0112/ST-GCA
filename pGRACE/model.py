import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from pGRACE.utils import permutation


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


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class GCL(nn.Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(GCL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        # self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        # torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)

        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return emb, emb_a, ret, ret_a


class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class GRACE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_en,
                 num_de,
                 num_hidden,
                 graph_neigh
                 ):
        super(GRACE, self).__init__()
        self.encoder = Encoder(input_dim, num_en, num_de)
        self.gcl = GCL(num_en[-1], num_hidden, graph_neigh)
        self.attention = Attention(num_hidden)

    def forward(self, feat, adj):
        z = self.encoder(feat)
        z_a = permutation(z)
        emb, emb_a, ret, ret_a = self.gcl(z, z_a, adj)
        emb = torch.stack([emb, emb_a], dim=1)
        emb, _ = self.attention(emb)
        de_z = self.encoder.forward_d(emb)
        return ret, ret_a, emb, de_z


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
