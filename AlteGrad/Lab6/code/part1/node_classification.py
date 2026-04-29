"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
# your code here #
##################

nx.draw_networkx(G, node_color=list(idx_to_class_label.values())) #because nodes are ordered in this graph
plt.savefig("Network_classes_visuals.png")
plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, num_walks=n_walks, walk_length=walk_length, n_dim=n_dim) # your code here


embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

##################
# your code here #
##################
print(X_train.shape, y_train.shape)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test accuracy (Deep Walk embeddings):", acc)

############## Task 8
# Generates spectral embeddings

##################
# your code here #
##################

A = nx.to_numpy_array(G)
degrees = A.sum(axis=1)
D_inv = np.diag(1.0 / degrees)

L_rw = np.eye(n) - D_inv @ A
vals, vecs = eigs(L_rw, k=2, which='SM')
spectral_embeddings = np.real(vecs)
X_train = spectral_embeddings[idx_train]
X_test  = spectral_embeddings[idx_test]

clf = LogisticRegression().fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Test accuracy (Spectral embeddings):", acc)
