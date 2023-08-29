import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from sklearn import metrics
import torch.nn.functional as F
from sklearn.cluster import KMeans

from pGRACE.get_graph import spatial_construct_graph, features_construct_graph
from pGRACE.model import GRACE
from pGRACE.utils import clustering, preprocess


import matplotlib as plt
plt.use('TkAgg')


def train(args, x, fadj, sadj):

    model.train()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        emb_1, neg_1, summary_1, emb_2, neg_2, summary_2, emb, de_z = model(x, fadj, sadj)


        loss_feat = F.mse_loss(x, de_z)
        loss_1 = model.DGI_model.loss(emb_1, neg_1, summary_1)
        loss_2 = model.DGI_model.loss(emb_2, neg_2, summary_2)

        loss = loss_feat + (loss_1 + loss_2)

        loss.backward()
        optimizer.step()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

    return emb


def init_args():
    parser = argparse.ArgumentParser(description='Graph Contrastive Learning with Adaptive Augmentation')

    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--num_en', type=int, default=[128, 64])
    parser.add_argument('--num_de', type=int, default=[32, 128])
    parser.add_argument('--num_hidden', type=int, default=32)

    parser.add_argument('--num_epochs', type=int, default=700)
    #1000, 0.5039, 0.6032
    #800, 0.5797, 0.6516
    parser.add_argument('--n_top_genes', type=int, default=1000)

    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--radius', type=int, default=300)

    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--no-cuda', action='store_true', default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args.cuda = torch.cuda.is_available()

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

    os.environ['R_HOME'] = 'C:\Program Files\R\R-4.2.2'
    #adata = sc.read_visium('/opt/data/private/DLPFC/151674', count_file='filtered_feature_bc_matrix.h5')
    adata = sc.read_visium('D:\PycharmProjects\DeepST-main\data\DLPFC\\151673', count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    preprocess(args, adata)

    feat = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)  # (3639, 1000)

    fadj = features_construct_graph(feat, args.k).to(device)
    sadj = spatial_construct_graph(adata, args.radius).to(device)

    model = GRACE(feat.shape[1], args.num_en, args.num_de, args.num_hidden).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    emb = train(args, feat, fadj, sadj)
    emb = emb.detach().cpu().numpy()


    adata.obsm['emb'] = emb
    clustering(adata, method='mclust')

    df_meta = pd.read_csv('D:\PycharmProjects\DeepST-main\data\DLPFC\\151673' + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
    print('==============================')
    print(ARI)
    print(NMI)

    sc.pl.spatial(adata,
                  color=["ground_truth", "domain"],
                  title=["Ground truth", "ARI=%.4f" % ARI],
                  show=True)
