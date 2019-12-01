"""Reads in reddit dataset as downloaded from http://snap.stanford.edu/graphsage/#datasets
    and dumps dictionary with keys adjacency, labels and features as pickle file.
    Note: WORKS ONLY IF NETWORKX VERSION == 1.11"""

import json
from networkx.readwrite import json_graph
import numpy as np
import os
import pickle
import scipy.sparse as sp
import zipfile

from sparse_help_funcs import sparse2tuple
from preprocess_model_inputs import norm_adj


## extract reddit files

os.chdir('/Users/sahra/Desktop/Implementation/original_data')
zip_ref = zipfile.ZipFile("reddit.zip", 'r')
zip_ref.extractall()
zip_ref.close()


## read in data

G = json_graph.node_link_graph(json.load(open("reddit/reddit-G.json")))
# networkx graph

id_map = json.load(open("reddit/reddit-id_map.json"))
# dictionary mapping the graph node ids to consecutive integers

class_map = json.load(open("reddit/reddit-class_map.json"))
# dictionary mapping the graph node ids to classes

feats = np.load("reddit/reddit-feats.npy")
# array of node features ordered by id_map


## transform data into graph structure

# parameters
n = len(list(id_map.items())) #  n is number of nodes
e = len(set(np.array(list(class_map.items()))[:,1])) # e is number of classes

# one-hot encoded labels
reorder = [id_map.get(j, j) for j in np.array(list(class_map.items()))[:,0]]
classes = np.array(list(class_map.items()))[:,1][reorder]
labels = sp.lil_matrix(np.zeros((n,e)))
for i in range(n):
    labels[i,int(classes[i])] = 1

# adjacency matrix
adjacency = sp.identity(n).tolil()
for edge in G.edges():
    adjacency[id_map[edge[0]], id_map[edge[1]]] = 1
adjacency = sparse2tuple(norm_adj(adjacency))

# log transform num comments and score acc to Hamilton and Ying (2017)
feats[:, 0] = np.log(feats[:, 0] + 1.0)
feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
feats_ = sparse2tuple(sp.lil_matrix(feats))


## dump dictionary as pickle file

os.chdir('/Users/sahra/Desktop/Implementation')
reddit = {'adjacency':adjacency,'labels':labels,'features':feats_}
with open('graph_data/reddit.pkl','wb') as output:
  pickle.dump(reddit, output, pickle.HIGHEST_PROTOCOL)
