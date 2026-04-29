"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx


############## Task 1 ##############
path = "datasets/CA-HepTh.txt"
print("TASK 1")
G = nx.read_edgelist(path=path, delimiter="\t")
print("Number of Nodes", G.number_of_nodes())
print("Number of Edges", G.number_of_edges())


############## Task 2 ##############
print("TASK 2")
number_of_cc = sum([1 for _ in nx.connected_components(G)])
print("Number of connected components", number_of_cc)
largest_component = G.subgraph(max(nx.connected_components(G), key=len))
print(
    "Ratio of Edges {:.2f}".format(
        largest_component.number_of_edges() / G.number_of_edges()
    )
)
print(
    "Ratio of Nodes {:.2f}".format(
        largest_component.number_of_nodes() / G.number_of_nodes()
    )
)
