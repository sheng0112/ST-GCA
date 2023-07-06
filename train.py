import argparse
import random
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics

import torch
import os

from pGRACE.get_graph import graph
from pGRACE.functional import degree_drop_weights, feature_drop_weights, drop_edge_weighted, drop_feature_weighted_2
from pGRACE.model import Encoder, GRACE
from pGRACE.utils import get_base_model, get_activation, clustering, preprocess, get_feature
from torch_geometric.utils import degree, to_undirected

import matplotlib as plt

plt.use('TkAgg')


def train(x, edge_index, drop_weights, feature_weights, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1,
          drop_feature_rate_2):
    model.train()
    optimizer.zero_grad()

    def drop_edge(p, drop_weights):
        return drop_edge_weighted(edge_index, drop_weights, p, threshold=0.7)

    # 对比学习
    edge_index_1 = drop_edge(drop_edge_rate_1, drop_weights)  # (2,211503)，涉及一个随机抽样，所以不同
    edge_index_2 = drop_edge(drop_edge_rate_2, drop_weights)  # (2,344077)
    # x_1 = drop_feature(data.x, param['drop_feature_rate_1']) #(13752,767), 随机掩码
    # x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    x_1 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_1)  # 上面那个相当于没用
    x_2 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_2)  # (13752,767),上同

    z1 = model(x_1, edge_index_1)  # (13752,128), x_1，剩下的节点特征，edge_index_1，剩下的边
    z2 = model(x_2, edge_index_2)  # (13752,128)

    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()

    return loss.item(), z1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--dataset', type=str, default='Amazon-Computers')
    # parser.add_argument('--param', type=str, default='local:amazon_computers.json')
    parser.add_argument('--seed', type=int, default=39788)
    # parser.add_argument('--verbose', type=str, default='train,eval,final')
    # parser.add_argument('--save_split', type=str, nargs='?')
    # parser.add_argument('--load_split', type=str, nargs='?')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 128,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.2,
        'drop_edge_rate_2': 0.6,
        'drop_feature_rate_1': 0.3,
        'drop_feature_rate_2': 0.4,
        'tau': 0.1,
        'num_epochs': 1000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # the location of R (used for the mclust clustering)
    os.environ['R_HOME'] = 'C:\Program Files\R\R-4.2.2'
    os.environ['R_USER'] = 'D:\py\\anaconda\envs\\pytorch\Lib\site-packages\\rpy2'

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), default=default_param[key])
    args = parser.parse_args()

    # parse param
    # sp = SimpleParam(default=default_param)
    # param = sp(source=args.param, preprocess='nni') #nni，自动化超参数训练

    # merge cli arguments and parsed param

    # use_nni = args.param == 'nni'
    # if use_nni and args.device != 'cpu':
    # args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    device = torch.device(args.device)

    # path = osp.expanduser('~/datasets')
    # path = osp.join(path, args.dataset)
    # dataset = get_dataset(path, args.dataset)

    # data = dataset[0]
    # data = data.to(device) #x(13752,767), edge_index(2,491722), y(13752)

    # generate split
    # split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    adata = sc.read_visium('D:\PycharmProjects\DeepST-main\data\DLPFC\\151673',
                           count_file='filtered_feature_bc_matrix.h5')

    adata = preprocess(adata)
    adata = get_feature(adata)
    adj, edges = graph(adata.obsm["spatial"], distType="KDTree", k=30, rad_cutoff=150).main()
    #data = torch.from_numpy(adata.X.todense())
    data = csr_matrix(adata.obsm['feat'])
    data = torch.from_numpy(data.todense())
    edges = torch.from_numpy(np.array(list(edges))).t()

    # adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    encoder = Encoder(data[1].numel(), args.num_hidden, get_activation(args.activation),
                      base_model=get_base_model(args.base_model), k=args.num_layers).to(device)
    model = GRACE(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if args.drop_scheme == 'degree':
        drop_weights = degree_drop_weights(edges).to(device)
    else:
        drop_weights = None

    if args.drop_scheme == 'degree':
        edge_index_ = to_undirected(edges)
        node_deg = degree(edge_index_[1])  # 节点的度(13752,)
        feature_weights = feature_drop_weights(data, node_c=node_deg).to(device)
    else:
        feature_weights = torch.ones((data.size(1),)).to(device)

    # log = args.verbose.split(',')  # ['train','eval','final']

    for epoch in range(1, args.num_epochs + 1):
        loss, z1 = train(data, edges, drop_weights, feature_weights, args.drop_edge_rate_1, args.drop_edge_rate_2,
                         args.drop_feature_rate_1, args.drop_feature_rate_2)
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch == args.num_epochs:
            adata.obsm['emb'] = z1.detach().cpu().numpy()
            clustering(adata, method='leiden')
            df_meta = pd.read_csv('D:\PycharmProjects\DeepST-main\data\DLPFC\\151673' + '\metadata.tsv', sep='\t')
            df_meta_layer = df_meta['layer_guess']
            adata.obs['ground_truth'] = df_meta_layer.values
            adata = adata[~pd.isnull(adata.obs['ground_truth'])]
            ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
            print('==============================')
            print(ARI)
            sc.pl.spatial(adata,
                          img_key="hires",
                          color=["ground_truth", "domain"],
                          title=["Ground truth", "ARI=%.4f" % ARI],
                          show=True)

        # if epoch % 100 == 0:
        # acc = test()

        # if 'eval' in log:
        # print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

    # acc = test(final=True)

    # if 'final' in log:
    # print(f'{acc}')
