import torch
import numpy as np
from scipy.sparse import csr_matrix

from pGRACE.get_graph import graph
from pGRACE.functional import degree_drop_weights, feature_drop_weights, drop_edge_weighted, drop_feature_weighted_2
from pGRACE.model import Encoder, GRACE
from pGRACE.utils import preprocess, get_feature, fix_seed, get_activation, get_base_model
from torch_geometric.utils import degree, to_undirected


class ST_GCA():
    def __init__(self,
                 adata,
                 device=torch.device('cuda'),
                 seed=3,
                 learning_rate=0.01,
                 num_hidden=128,
                 num_proj_hidden=32,
                 activation='prelu',
                 base_model='GCNConv',
                 num_epochs=800,
                 drop_edge_rate_1=0.2,
                 drop_edge_rate_2=0.6,
                 drop_feature_rate_1=0.3,
                 drop_feature_rate_2=0.4,
                 tau=0.1,
                 weight_decay=1e-5
                 ):
        self.adata = adata.copy()
        self.device = device
        self.seed = seed
        self.learning_rate = learning_rate
        self.num_hidden = num_hidden
        self.num_proj_hidden = num_proj_hidden
        self.activation = activation
        self.base_model = base_model
        self.num_epochs = num_epochs
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.tau = tau
        self.weight_decay = weight_decay

        fix_seed(self.seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        if 'adj' not in adata.obsm.keys():
            self.adj, self.edges = graph(adata, distType="KDTree", k=30, rad_cutoff=150).main()

        #self.features = torch.FloatTensor(csr_matrix(self.adata.obsm['feat'].copy())).to(self.device)
        self.features = torch.from_numpy(csr_matrix(self.adata.obsm['feat'].copy()).todense()).to(self.device)
        self.adj = torch.FloatTensor(self.adj + np.eye(self.adj.shape[0]))
        #self.edges = torch.FloatTensor(np.array(list(self.edges))).t()
        self.edges = torch.from_numpy(np.array(list(self.edges))).t()
        self.dim_input = self.features.shape[1]

    def train(self):
        self.encoder = Encoder(self.dim_input, self.num_hidden, get_activation(self.activation),
                      base_model=get_base_model(self.base_model)).to(self.device)
        self.model = GRACE(self.encoder, self.num_hidden, self.num_proj_hidden, self.tau).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.drop_weights = degree_drop_weights(self.edges).to(self.device)

        edge_index_ = to_undirected(self.edges)
        node_deg = degree(edge_index_[1].long())
        self.feature_weights = feature_drop_weights(self.features, node_c=node_deg).to(self.device)

        print('Begin to train ST data...')
        self.model.train()

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            def drop_edge(edge_index, drop_weights, p):
                return drop_edge_weighted(edge_index, drop_weights, p, threshold=0.7)

            self.edge_index_1 = drop_edge(self.edges, self.drop_weights, self.drop_edge_rate_1)
            self.edge_index_2 = drop_edge(self.edges, self.drop_weights, self.drop_edge_rate_2)

            self.x_1 = drop_feature_weighted_2(self.features, self.feature_weights, self.drop_feature_rate_1)
            self.x_2 = drop_feature_weighted_2(self.features, self.feature_weights, self.drop_feature_rate_2)

            self.z1 = self.model(self.x_1, self.edge_index_1)
            self.z2 = self.model(self.x_2, self.edge_index_2)

            loss = self.model.loss(self.z1, self.z2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        with torch.no_grad():
            self.model.eval()

            self.emb_rec = self.model(self.x_1, self.edge_index_1).detach().cpu().numpy()

            self.adata.obsm['emb'] = self.emb_rec

            return self.adata