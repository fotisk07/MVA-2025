import re
import warnings

import numpy as np
from nltk.stem.porter import PorterStemmer
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore")


def load_file(filename):
    labels = []
    docs = []

    with open(filename, encoding="utf8", errors="ignore") as f:
        for line in f:
            content = line.split(":")
            labels.append(content[0])
            docs.append(content[1][:-1])

    return docs, labels


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs):
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])

    return preprocessed_docs


def get_vocab(train_docs, test_docs):
    vocab = dict()

    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


path_to_train_set = "datasets/train_5500_coarse.label"
path_to_test_set = "datasets/TREC_10_coarse.label"

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


# Task 11
def create_graphs_of_words(docs, vocab, window_size):
    graphs = []

    for doc in docs:
        if len(doc) == 0:
            G = nx.Graph()
            G.add_node(0, label="__EMPTY__")
            graphs.append(G)
            continue

        G = nx.Graph()
        local_id = {}
        next_id = 0

        # assign local node ids
        for w in doc:
            if w not in local_id:
                local_id[w] = next_id
                G.add_node(next_id, label=w)
                next_id += 1

        n = len(doc)
        for i in range(n):
            wi = doc[i]
            idx_i = local_id[wi]
            for j in range(i + 1, min(i + 1 + window_size, n)):
                wj = doc[j]
                idx_j = local_id[wj]
                if idx_i != idx_j:
                    if G.has_edge(idx_i, idx_j):
                        G[idx_i][idx_j]["weight"] += 1
                    else:
                        G.add_edge(idx_i, idx_j, weight=1)

        graphs.append(G)

    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3)
G_test_nx = create_graphs_of_words(test_data, vocab, 3)
# nx.draw_networkx(G_train_nx[3], with_labels=True)


# Task 12
# Convert NetworkX graphs to GraKeL graphs
G_train = list(graph_from_networkx(G_train_nx, node_labels_tag="label"))
G_test = list(graph_from_networkx(G_test_nx, node_labels_tag="label"))

gk = WeisfeilerLehman(
    n_iter=1,
    base_graph_kernel=VertexHistogram,  # instantiate
    normalize=False,
)

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# Task 13

# Train an SVM classifier and make predictions

##################
# your code here #
##################
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)

y_pred = clf.predict(K_test)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))


# Task 14

from grakel.kernels import ShortestPath, RandomWalk, PyramidMatch

base_kernels = {
    "VertexHistogram": VertexHistogram,
    "None": None,
    # "PyramidMatch": PyramidMatch,
}

results = {}

for name, bk in base_kernels.items():
    print("\n=== WL +", name, "===")

    gk = WeisfeilerLehman(
        n_iter=1,
        base_graph_kernel=bk,
        normalize=False,
    )

    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)

    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    results[name] = acc

print("\nSummary:", results)

##################
# your code here #
##################
print("""
=== Task 14: Kernel Feasibility Analysis on Graph-of-Words ===

We experimented with several graph kernels on the dataset
Graph-of-words graphs are small but have extremely high-cardinality node labels and are moderately
dense due to the sliding window construction.
NON-SCALABLE KERNELS
   - ShortestPath
   - WL + ShortestPath
   - PyramidMatch
   - RandomWalk
   - Graphlet kernels

   These kernels rely on counting or enumerating path types, distances, or
   substructures. With high-cardinality labels, the feature space explodes.


Conclusion:
Graph-of-words is inherently incompatible with most classical kernels due to
its large label alphabet and dense local structure. Only WL-based and histogram
kernels are suitable for this representation. Therefore, for Task 14, we report
results only for the kernels that run on the full dataset, and we document the
scalability issues encountered for the others.
""")
