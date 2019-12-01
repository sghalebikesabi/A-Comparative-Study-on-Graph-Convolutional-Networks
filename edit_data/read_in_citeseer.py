"""Reads in citeseer dataset as downloaded from https://linqs.soe.ucsc.edu/data
    and dumps dictionary with keys adjacency, labels and features as pickle file."""


import numpy as np
import pickle
import scipy.sparse as sp
import tarfile

from sparse_help_funcs import sparse2tuple
from preprocess_model_inputs import norm_adj


## extract citeseer files

tar = tarfile.open("original_data/citeseer.tgz")
tar.extractall()
tar.close()


## read in data into lists

with open('citeseer/citeseer.cites') as f:
  cites = list(f)
cites = [cites[j].split("\t") for j in range(len(cites))]
# cites: specifies in each item in entry 0 the paper ID of a paper that cites
#        a paper with the ID in entry 1

with open('citeseer/citeseer.content') as f:
  content = list(f)
content = [content[j].split("\t") for j in range(len(content))]
# content: specifies in each item in entry 0 the paper ID of a paper with all
#          its feature values in the following entries. Last entry is research
#          area of paper as string ('Agents', 'AI', 'DB', 'IR', 'ML', 'HCI')

# delete "\n" in last entry of each item
for i in range(len(content)):
  content[i][-1] = content[i][-1].replace('\n','')
for i in range(len(cites)):
  cites[i][-1] = cites[i][-1].replace('\n','')

# save unique paper IDs in 'IDs'
IDs = []
for i in range(len(content)):
  IDs.append(content[i][0])

# delete all links referencing papers not in IDs
i = 0
while i < len(cites):
  if not cites[i][0] in IDs or not cites[i][1] in IDs:
    cites.pop(i)
  else:
    i+=1


## transform data into graph structure

# parameters
n = len(content) # n is number of nodes
d = len(content[0])-2 # d is number of features per node
classes = ['Agents','AI','DB','IR','ML','HCI']
e = len(classes) # e is the number of classes

# create 'features' and 'labels' array
features = np.zeros((n,d))
labels = np.zeros((n,e))
for i in range(n):
  features[i,:] = content[i][1:(d+1)]
  labels[i,:] = [content[i][-1] == classes[j] for j in range(e)]
features = sparse2tuple(sp.lil_matrix(features))

if np.sum([np.sum(labels[j,:])>1 for j in range(n)])>0:
  raise Exception('At least one node has more than one class!')

# create symmetric 'adjacency' matrix
adjacency = sp.identity(n).tolil()
ID_dict = {k: v for k, v in zip(IDs, range(n))}
heads = [ID_dict.get(j, j) for j in np.array(cites)[:,0]]
tails = [ID_dict.get(j, j) for j in np.array(cites)[:,1]]
for i in range(len(cites)):
  adjacency[heads[i],tails[i]] = 1
  adjacency[tails[i],heads[i]] = 1
adjacency = sparse2tuple(norm_adj(adjacency))

## dump dictionary as pickle file

citeseer = {'adjacency':adjacency,'labels':sp.lil_matrix(labels),
            'features':features}
with open('graph_data/citeseer.pkl','wb') as output:
  pickle.dump(citeseer, output, pickle.HIGHEST_PROTOCOL)
