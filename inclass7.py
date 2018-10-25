import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from networkx.algorithms.community.centrality import girvan_newman

B = pd.read_csv('dm-graphtask-adj.txt', header=None)

G = nx.from_numpy_matrix(B.values)

comp = girvan_newman(G, most_valuable_edge=None)

k = 4
counter = 0
clusters = []

for communities in itertools.islice(comp, k):
    if counter == 3:
        for c in communities:
            nodelist = list(c)
            clusters.append(nodelist)
            print (nodelist)

    counter += 1

pos = nx.spring_layout(G)

# nodes
colors = ['r', 'g', 'b', 'y', 'm']

for i in range(5):
    nx.draw_networkx_nodes(G,pos,
                           nodelist=clusters[i],
                           node_color=colors[i],
                           node_size=500,
                           alpha=0.8)

# edges
nx.draw_networkx_edges(G,pos)

plt.show()