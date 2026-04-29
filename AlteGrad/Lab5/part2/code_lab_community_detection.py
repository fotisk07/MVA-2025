"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
from collections import defaultdict


############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    A = nx.adjacency_matrix(G).astype(float)

    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = diags(1.0 / degrees)
    L_rw = eye(G.number_of_nodes()) - D_inv @ A

    eig_val, eig_vec = eigs(L_rw, k=k, which="SR")  # SR = smallest real parts
    eig_vec = eig_vec.real  # ensure real values
    eig_val = eig_val.real

    idx = eig_val.argsort()
    U = eig_vec[:, idx]  # sorted eigenvectors
    U = U[:, :k]  # keep only k small eigenvectors

    # 4. K-means on the rows of U
    kmeans = KMeans(n_clusters=k, n_init=20)
    clusters = kmeans.fit_predict(U)

    return {node: cid for node, cid in zip(G.nodes(), clusters)}


############## Task 4
path = "datasets/CA-HepTh.txt"
G = nx.read_edgelist(path=path, delimiter="\t")
GCC = G.subgraph(max(nx.connected_components(G), key=len))
clustering = spectral_clustering(GCC, 50)

##################
# your code here #
##################


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    ##################
    # your code here #
    ##################
    comms = defaultdict(list)
    for node, cid in clustering.items():
        comms[cid].append(node)

    communities = list(comms.values())
    return nx.community.modularity(G, communities)


############## Task 6
clustering_rand = {node: randint(0, 49) for node in GCC.nodes()}

print("Modularity of 50 clustesr : ", modularity(GCC, clustering))
print("Modularity of random partition : ", modularity(GCC, clustering_rand))


##################
# your code here #
##################
