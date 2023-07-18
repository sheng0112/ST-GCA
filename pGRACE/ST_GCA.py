import torch
import numpy as np

from pGRACE.functional import degree_drop_weights, drop_edge_weighted
from pGRACE.model import Encoder, GRACE, simCLR_model, train_model
from pGRACE.utils import fix_seed, get_activation, get_base_model, get_edges, edge_c


class ST_GCA:
    def __init__(self,
                 args,
                 data,
                 device=torch.device('cuda'),
                 ):
        self.args = args
        self.data = data
        self.device = device

        fix_seed(self.args.seed)

    def train(self):  # args, data:trainset, device
        self.encoder = Encoder(self.args.n_gene, self.args.num_hidden, get_activation(self.args.activation),
                               base_model=get_base_model(self.args.base_model)).to(self.device)
        self.simCLR = simCLR_model().to(self.device)
        self.model = GRACE(self.encoder, self.simCLR, self.args.num_hidden, self.args.num_proj_hidden, tau=self.args.tau).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        trainer = train_model(self.args, self.model, self.optimizer, self.device)
        trainer.fit(self.data)

    def valid(self, testloader):
        Xg = []
        Xi = []
        self.model.eval()
        valid_loss = 0
        valid_cnt = 0

        for features, image, spatial, idx in testloader:
            features = features.to(self.device)
            image = image.to(self.device)
            spatial = spatial.to(self.device)

            edges = get_edges(testloader.dataset.graph, idx, flag=1).t()
            self.drop_weights = degree_drop_weights(edges).to(self.device)
            edge_index = drop_edge_weighted(edges, self.drop_weights, self.args.drop_edge_rate_1,
                                            threshold=0.7)
            edge_index = edge_c(edge_index)

            xg = self.model.encoder(features, edge_index.to(torch.int64))
            xi = self.model.simCLR(image)

            Xg.append(xg.detach().cpu().numpy())
            Xi.append(xi.detach().cpu().numpy())

        Xg = np.vstack(Xg)
        Xi = np.vstack(Xi)

        return Xg, Xi
