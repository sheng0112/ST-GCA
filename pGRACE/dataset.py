import os
import cv2
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from torch.utils import data
from torchvision import transforms
from torchtoolbox.transform import Cutout

from pGRACE.get_graph import graph
from pGRACE.utils import adata_preprocess_pca, adata_preprocess_hvg


class Dataset(data.Dataset):
    def __init__(self, path, name, gene_preprocess='pca', n_genes=3000, img_size=16, train=True):  # no.1
        super(Dataset, self).__init__()

        adata = sc.read_visium(os.path.join(path, name), count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()

        if train == False:
            self.adata = adata

        df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
        self.label = pd.Categorical(df_meta['layer_guess']).codes
        # image
        full_image = cv2.imread(os.path.join(path, name, f'spatial/full_image.tif'))
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        patches = []
        for x, y in adata.obsm['spatial']:
            patches.append(full_image[y - img_size: y + img_size, x - img_size:x + img_size])  # (32,32,3)
        patches = np.array(patches)
        self.image = patches

        self.n_clusters = self.label.max() + 1
        self.spatial = adata.obsm['spatial']
        self.n_pos = self.spatial.max() + 1

        # preprocess
        if gene_preprocess == 'pca':
            self.gene = adata_preprocess_pca(adata, pca_n_comps=n_genes).astype(np.float32)
        elif gene_preprocess == 'hvg':
            # self.gene = np.array(adata_preprocess_hvg(adata, n_top_genes=n_genes)).astype(np.float32)
            self.gene = adata_preprocess_hvg(adata, n_top_genes=n_genes).todense()

        self.graph = graph(self.spatial, distType="KDTree", k=30, rad_cutoff=150).main()

        self.train = train
        self.img_train_transform = transforms.Compose([
            Cutout(0.5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # self.gene = self.gene[self.label != -1]
        # self.image = self.image[self.label != -1]
        # self.label = self.label[self.label != -1]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        spatial = torch.from_numpy(self.spatial[idx])
        y = self.label[idx]

        if self.train:
            xg = self.gene[idx]

            xg = torch.from_numpy(xg)

            xi_u = self.img_train_transform(self.image[idx])
            xi_v = self.img_train_transform(self.image[idx])

            return xg, xi_u, xi_v, spatial, idx

        else:
            xg = self.gene[idx]
            xg = torch.from_numpy(xg)
            xi = self.img_test_transform(self.image[idx])

            return xg, xi, spatial, idx
