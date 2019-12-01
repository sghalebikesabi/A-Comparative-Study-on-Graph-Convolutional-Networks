"""Generates random graphs as described in the report in subsection Data."""

import numpy as np
import pickle
import random
import scipy.sparse as sp

from sparse_help_funcs import sparse2tuple
from preprocess_model_inputs import norm_adj


random.seed(449)

def generate_graph(n_nodes, node_degree):
    """Saves a randomly generated graph as dictionary in directory graph_data.
    Args:
        n_nodes - number of nodes
        node_degree - average node degree
    Returns:
        saves dictionary with keys features,labels and adjacency as lilmatrix
        in pickle file
    """
    max_edges = n_nodes*(n_nodes-1)/2
    n_edges = node_degree*n_nodes
    random_graphs = {}
    random_graphs["features"] = sp.identity(n_nodes).tolil()
    random_graphs["adjacency"] = sp.identity(n_nodes).tolil()
    random_graphs["labels"] = sp.lil_matrix(np.ones(n_nodes))

    for i in range(n_nodes):
      for j in range(i):
        if random.uniform(0,max_edges) < n_edges:
          random_graphs["adjacency"][i,j] += 1
          random_graphs["adjacency"][j,i] += 1

    random_graphs['adjacency'] = sparse2tuple(norm_adj(random_graphs['adjacency']))
    random_graphs['features'] = sparse2tuple(random_graphs['features'])

    with open('graph_data/random_graph_{0}_{1}.pkl'.format(n_nodes,node_degree),
              'wb') as output:
      pickle.dump(random_graphs, output, pickle.HIGHEST_PROTOCOL)


# create three random graphs with varying number of nodes
for iter in [3,4,5]:
    n_nodes = 10**iter
    node_degree = 2
    generate_graph(n_nodes, node_degree)
    print('n_nodes=',n_nodes, 'node_degree=',node_degree,'saved')


# create three random graphs with varying node degrees
for iter in [2,10,50]:
    n_nodes = 10**4
    node_degree = iter
    generate_graph(n_nodes, node_degree)
    print('n_nodes=',n_nodes, 'node_degree=',node_degree,'saved')
