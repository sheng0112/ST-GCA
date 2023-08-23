import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import metrics
from torch_geometric.utils import degree, to_undirected

import pGRACE
from pGRACE.functional import degree_drop_weights, feature_drop_weights, drop_edge_weighted
from pGRACE.get_graph import graph
from pGRACE.loss import loss
from pGRACE.model import GRACE
from pGRACE.utils import clustering, preprocess


def train(args, x, edges, drop_weights):
    model.train()
    for epoch in range(1, args.num_epochs + 1):
        optimizer.zero_grad()

        x = x.to(args.device)

        edge_index_1 = drop_edge_weighted(edges, drop_weights, args.drop_edge_rate_1,
                                          threshold=0.7).to(args.device)
        edge_index_2 = drop_edge_weighted(edges, drop_weights, args.drop_edge_rate_2,
                                          threshold=0.7).to(args.device)

        # x_1 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_1).to(args.device)
        # x_2 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_2).to(args.device)

        h1, h2, emb, de_z = model(x, edge_index_1, edge_index_2)

        loss = pGRACE.loss.loss(h1, h2, de_z, x)

        loss.backward()
        optimizer.step()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

    return emb


def init_args():
    parser = argparse.ArgumentParser(description='Graph Contrastive Learning with Adaptive Augmentation')

    # Experiment Settings
    parser.add_argument('--seed', type=int, default=3407)  ###
    parser.add_argument('--learning_rate', type=float, default=0.01)  ###

    # Model Design
    parser.add_argument('--num_en', type=int, default=[64, 32])  ###
    parser.add_argument('--num_de', type=int, default=[32])  ###
    parser.add_argument('--num_hidden', type=int, default=32)  ###
    parser.add_argument('--num_proj', type=int, default=32)  ###
    parser.add_argument('--activation', type=str, default='prelu')  ###
    parser.add_argument('--base_model', type=str, default='GCNConv')  ###
    parser.add_argument('--num_layers', type=int, default=2)

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=800)  ###
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)  ###
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.7)  ###
    # parser.add_argument('--drop_feature_rate_1', type=float, default=0.4)
    # parser.add_argument('--drop_feature_rate_2', type=float, default=0.3)

    # Model Hyperparameters
    parser.add_argument('--tau', type=float, default=0.1)  ###
    parser.add_argument('--weight_decay', type=float, default=1e-5)  ###

    parser.add_argument('--no-cuda', action='store_true', default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    return args


if __name__ == '__main__':
    args = init_args()
    device = torch.device(args.device)

    adata = sc.read_visium('/opt/data/private/151673',
                           count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    adata, feat = preprocess(adata)
    # adata = get_feature(adata)

    # feat = torch.FloatTensor(adata.obsm['feat']).to(device)
    feat = feat.to(device)
    edges = graph(adata.obsm["spatial"], distType="kneighbors_graph", k=30, rad_cutoff=150).main()
    # edges = load_graph(adata)
    edges = torch.FloatTensor(np.array(list(edges))).t().to(device)

    #
    model = GRACE(feat.shape[1], args.num_en, args.num_de, args.num_hidden, args.num_proj).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    drop_weights = degree_drop_weights(edges.to(torch.int64)).to(device)

    edge_index_ = to_undirected(edges)
    node_deg = degree(edge_index_[1].to(torch.int64))  # 节点的度(13752,)
    feature_weights = feature_drop_weights(feat, node_c=node_deg).to(device)

    emb = train(args, feat, edges, drop_weights)

    adata.obsm['emb'] = emb.detach().cpu().numpy()
    clustering(adata, method='leiden')
    df_meta = pd.read_csv('/opt/data/private/151673' + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    print('==============================')
    print(ARI)
    """
    sc.pl.spatial(adata,
                  color=["ground_truth", "domain"],
                  title=["Ground truth", "ARI=%.4f" % ARI],
                  show=True)
    """
