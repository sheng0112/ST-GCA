import torch
import scanpy as sc
import pandas as pd
from sklearn import metrics

from pGRACE import ST_GCA
from pGRACE.utils import clustering

import matplotlib as plt

plt.use('TkAgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_clusters = 7

dataset = '151673'
file_fold = 'D:\PycharmProjects\DeepST-main\data\DLPFC\\' + str(dataset)
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

model = ST_GCA.ST_GCA(adata, device=device)

adata = model.train()

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
