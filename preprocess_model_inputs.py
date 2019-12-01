"""Help functions for preprocessing graph data as input in our models."""

import numpy as np
import random
import scipy.sparse as sp


def data_split(labels, num_train, num_val, num_test, train_per_class=False, seed=None):
    """Loads dataset_str from working directory 'graph_data'.
    Args:
        labels - one-hot encoded label data
    Returns:
        tuple containing...
        train_labels - two dimensional numpy array of size n with entry i being the
            label of node i if node is included in training dataset, 0 otherwise
        val_labels - two dimensional numpy array of size n with entry i being the
            label of node i if node is included in validation dataset, 0 otherwise
        test_labels - two dimensional numpy array of size n with entry i being the
            label of node i if node is included in test dataset, 0 otherwise
        train_mask - one dimensional numpy array of size n with entry i being True
            if node is included in training dataset
        val_mask - one dimensional numpy array of size n with entry i being True if
            node is included in validation dataset
        test_mask - one dimensional numpy array of size n with entry i being True if
            node is included in test dataset
        where n denotes the number of nodes and e is the number of classes
    """
    if seed is not None:
        random.seed(seed)

    n,e = labels.shape

    # create training, test and validation dataset indices
    if train_per_class==True:
        train_idx = np.reshape(np.array([random.sample(set(np.where(labels[:,i]
                    == 1)[0]),  num_train) for i in range(e)]), (num_train*e,))
    else:
        train_idx = random.sample(range(n),num_train)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = 1
    not_train_idx = list(set(range(n)) - set(train_idx))
    val_mask = np.zeros(n, dtype=bool)
    val_idx = random.sample(not_train_idx, num_val)
    val_mask[val_idx] = 1
    test_mask = np.zeros(n, dtype=bool)
    test_idx = random.sample(list(set(not_train_idx) - set(val_idx)), num_test)
    test_mask[test_idx] = 1

    # create label vectors of length n which only specify labels for train/test/val
    train_labels = np.zeros((n,e))
    test_labels = np.zeros((n,e))
    val_labels = np.zeros((n,e))
    train_labels[train_idx,] = labels[train_idx,]
    test_labels[test_idx,] = labels[test_idx,]
    val_labels[val_idx,] = labels[val_idx,]

    ## return tuple
    return(train_idx,train_labels,val_labels,test_labels,train_mask,val_mask,test_mask)


def preprocess_adj_bias(adj1):
    """Preprocess adjacency matrix as needed for GAT. Inspired by Velickovic et al. (2018)"""
    adj = adj1.tolil()
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return indices, adj.data, adj.shape


def norm_adj(adj):
    """Normalize adjacency matrix according to Kipf and Welling (2016)."""
    adj = adj.tocsr()
    rowsum = np.array(adj.sum(1))
    d_mat_inv_sqrt = sp.diags((rowsum**(-0.5))[:,0])
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()


def column_prop(adj):
    """Determine importance distribution up to proportionality according to Chen et al (2018)"""
    column_norm = sp.linalg.norm(adj, axis=0)
    norm_sum = np.sum(column_norm)
    return column_norm/norm_sum


def get_second_nb(adjacency, idx):
    """Get indices of the second order neighborhood of nodes with index idx."""
    n1_idx = set(idx)
    for i in idx:
        n1_idx.update(np.where(adjacency[i,:])[0])
    n2_idx = set(n1_idx)
    for i in list(n1_idx):
        n2_idx.update(np.where(adjacency[i,:])[0])
    return(list(n2_idx))
