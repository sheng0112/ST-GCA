# -*- coding: utf-8 -*-
import os
import torch
import random
import scipy.sparse as sp

from matplotlib import pyplot as plt
from torch.backends import cudnn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GraphConv
from sklearn.decomposition import PCA
import numpy as np
import scanpy as sc
import pandas as pd
import networkx as nx


def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def adata_preprocess_hvg(adata, n_top_genes):  # 可考虑 no.2
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return get_feature(adata)


def adata_preprocess_pca(i_adata, min_cells=3, pca_n_comps=300):  # 可考虑
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)

    return adata_X


def get_edges(graph, nodes = None, flag = 0):
    if flag == 0:
        edges = np.array(list(graph.edges()))
        return torch.from_numpy(edges)
    else:
        nodes = nodes.flatten()
        nodes = torch.unique(nodes)
        nodes = nodes.tolist()
        subgraph = graph.subgraph(nodes)
        edges = np.array(list(subgraph.edges()))
        return torch.from_numpy(edges)


def edge_c(edge_index):
    edge = edge_index
    edge_index = edge_index.flatten()
    edge_index = torch.unique(edge_index)
    edge_index = edge_index.tolist()
    dic = {}
    for i in np.arange(len(edge_index)):
        x = edge_index[i]
        dic[x] = i
    edge = edge.numpy()
    for i in np.arange(2):
        for j in np.arange(edge[0].shape[0]):
            edge[i][j] = dic[edge[i][j]]
    return torch.from_numpy(edge)


def get_feature(adata):  # 未使用
    adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

    return feat


def get_adj(x, edges):  # 未使用
    zip_list = zip(edges[0].numpy(), edges[1].numpy())
    edges = list(map(lambda item: (item[0], item[1]), zip_list))
    G = nx.Graph()
    G.add_edges_from(edges)
    adj = nx.adjacency_matrix(G).toarray()
    adj = preprocess_adj(adj)
    if x.shape[0] == adj.shape[0]:
        pass
    else:
        adj = np.pad(adj, ((0, 1), (0, 1)), mode='constant', constant_values=0.0)
    return torch.FloatTensor(adj)


def normalize_adj(adj):  # 未使用
    """Symmetrically normalize adjacency matrix."""
    adj = np.where(adj > 1, 1, adj)
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):  # 未使用
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return adj_normalized


def get_base_model(name: str):
    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GraphConv': GraphConv,
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):  # 待使用，看效果
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, method='mclust', start=0.1, end=3.0, increment=0.01, ):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1.
    end : float
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """

    # pca = PCA(n_components=35, random_state=42)
    # embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = adata.obsm['emb'].copy()

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']
    else:
        print('error')


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


"""
def get_predicted_results(name, path, z):
    adata = load_ST_file(os.path.join(path, name))
    df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
    label = pd.Categorical(df_meta['layer_guess']).codes
    n_clusters = label.max() + 1
    adata = adata[label != -1]

    adj_2d = calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].tolist(),
                                  histology=False)

    raw_preds = eval_mclust_ari(label[label != -1], z, n_clusters)

    if len(adata.obs) > 1000:
        num_nbs = 24
    else:
        num_nbs = 4

    refined_preds = refine(sample_id=adata.obs.index.tolist(), pred=raw_preds, dis=adj_2d, num_nbs=num_nbs)
    ari = adjusted_rand_score(label[label != -1], refined_preds)

    return ari, refined_preds
"""

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
