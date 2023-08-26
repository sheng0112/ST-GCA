import networkx as nx
import numba
import ot
import numpy as np
from scipy import stats
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph


class graph():
    def __init__(self,
                 data,
                 rad_cutoff,
                 k,
                 distType='euclidean', ):
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]

    def graph_computing(self):

        dist_list = ["euclidean", "braycurtis", "canberra", "mahalanobis", "chebyshev", "cosine",
                     "jensenshannon", "mahalanobis", "minkowski", "seuclidean", "sqeuclidean", "hamming",
                     "jaccard", "jensenshannon", "kulsinski", "mahalanobis", "matching", "minkowski",
                     "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                     "sqeuclidean", "wminkowski", "yule"]

        if self.distType == 'spearmanr':
            SpearA, _ = stats.spearmanr(self.data, axis=1)
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = SpearA[node_idx, :].reshape(1, -1)
                res = tmp.argsort()[0][-(self.k + 1):]
                for j in np.arange(0, self.k):
                    graphList.append((node_idx, res[j]))

        elif self.distType == "BallTree":
            from sklearn.neighbors import BallTree
            tree = BallTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)  # 距离和下标
            indices = ind[:, 1:]
            graphList = []
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = []
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList = []
            for node_idx in range(self.data.shape[0]):
                indices = np.where(A[node_idx] == 1)[0]
                for j in np.arange(0, len(indices)):
                    graphList.append((node_idx, indices[j]))

        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = []
            for node_idx in range(indices.shape[0]):
                for j in range(indices[node_idx].shape[0]):
                    if distances[node_idx][j] > 0:
                        graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' % (len(graphList) / self.data.shape[0]))

        elif self.distType in dist_list:
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = self.data[node_idx, :].reshape(1, -1)
                distMat = distance.cdist(tmp, self.data, self.distType)
                res = distMat.argsort()[:self.k + 1]
                tmpdist = distMat[0, res[0][1:self.k + 1]]
                boundary = np.mean(tmpdist) + np.std(tmpdist)
                for j in np.arange(1, self.k + 1):
                    if distMat[0, res[0][j]] <= boundary:
                        graphList.append((node_idx, res[0][j]))
                    else:
                        pass

        else:
            raise ValueError(
                f"""\
                {self.distType!r} does not support. Disttype must in {dist_list} """)

        return graphList

    def List2Dict(self, graphList):
        graphdict = {}
        tdict = {}
        for graph in graphList:
            end1 = graph[0]
            end2 = graph[1]
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist

        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []

        return graphdict

    def main(self):
        adj_mtx = self.graph_computing()  # 找到每个点的最近邻
        graphdict = self.List2Dict(adj_mtx)  # 将其存放在字典中
        graph = nx.from_dict_of_lists(graphdict)
        edges = np.array(list(graph.edges()))

        return edges


def construct_interaction(adata, n_neighbors=15):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']  # (3639, 2)

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')  # (3639, 3639)
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj
