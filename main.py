import torch
import scanpy as sc
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader

from pGRACE import ST_GCA
from pGRACE.args import init_args
from pGRACE.dataset import Dataset
from pGRACE.utils import clustering

import matplotlib as plt

# plt.use('TkAgg')w

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = init_args()
trainset = Dataset(path=args.path, name=args.name, gene_preprocess=args.gene_preprocess, n_genes=args.n_gene,
                   img_size=args.img_size, train=True)
trainloader = DataLoader(trainset, batch_size=128, num_workers=1, pin_memory=True)  # 注意这里，CPU和GPU的区别
testset = Dataset(path=args.path, name=args.name, gene_preprocess=args.gene_preprocess, n_genes=args.n_gene,
                  img_size=args.img_size, train=False)
testloader = DataLoader(testset, batch_size=128, num_workers=1, pin_memory=True)

model = ST_GCA.ST_GCA(args, trainloader, device=device)

model.train()

xg, xi = model.valid(testloader)
emb = xg + xi * 0.1

adata = testset.adata
adata.obsm['emb'] = emb

clustering(adata, method='leiden')

# df_meta = pd.read_csv('D:\PycharmProjects\DeepST-main\data\DLPFC\\151673\metadata.tsv', sep='\t')
# df_meta_layer = df_meta['layer_guess']
adata.obs['ground_truth'] = testset.label
adata = adata[~pd.isnull(adata.obs['ground_truth'])]
ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
print('==============================')
print(ARI)
"""
sc.pl.spatial(adata,
              img_key="hires",
              color=["ground_truth", "domain"],
              title=["Ground truth", "ARI=%.4f" % ARI],
              show=True)
"""
