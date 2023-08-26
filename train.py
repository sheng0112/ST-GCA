import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import metrics
from torch import nn
import torch.nn.functional as F

from pGRACE.get_graph import construct_interaction
from pGRACE.model import GRACE
from pGRACE.utils import clustering, preprocess, add_contrastive_label, preprocess_adj


def train(args, x, adj, label_CSL):
    loss_CSL = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        ret, ret_a, emb, de_z = model(x, adj)

        loss_sl_1 = loss_CSL(ret, label_CSL)
        loss_sl_2 = loss_CSL(ret_a, label_CSL)
        loss_feat = F.mse_loss(x, de_z)

        loss = loss_feat + (loss_sl_1 + loss_sl_2)

        loss.backward()
        optimizer.step()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

    return emb


def init_args():
    parser = argparse.ArgumentParser(description='Graph Contrastive Learning with Adaptive Augmentation')

    # Experiment Settings
    parser.add_argument('--seed', type=int, default=3407)  ###
    parser.add_argument('--learning_rate', type=float, default=0.001)  ###

    # Model Design
    parser.add_argument('--num_en', type=int, default=[128, 64])  ###
    parser.add_argument('--num_de', type=int, default=[32, 128])  ###
    parser.add_argument('--num_hidden', type=int, default=32)  ###

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1300)  ###

    # Model Hyperparameters
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

    adata = sc.read_visium('/opt/data/private/DLPFC/151673', count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    preprocess(adata)
    construct_interaction(adata)
    add_contrastive_label(adata)

    feat = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)  # (3639, 1000)

    adj = adata.obsm['adj']
    graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(adj.shape[0])).to(device)

    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)

    adj = preprocess_adj(adj)
    adj = torch.FloatTensor(adj).to(device)

    model = GRACE(feat.shape[1], args.num_en, args.num_de, args.num_hidden, graph_neigh).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    emb = train(args, feat, adj, label_CSL)

    adata.obsm['emb'] = emb.detach().cpu().numpy()
    clustering(adata, method='leiden')
    df_meta = pd.read_csv('/opt/data/private/DLPFC/151673' + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    print('==============================')
    print(ARI)
    sc.pl.spatial(adata,
                  color=["ground_truth", "domain"],
                  title=["Ground truth", "ARI=%.4f" % ARI],
                  show=True)
