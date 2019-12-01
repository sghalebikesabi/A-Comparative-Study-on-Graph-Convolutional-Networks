"""Help functions for working with sparse matrices"""

import numpy as np
import scipy.sparse as sp


def matrix2tuple(matrix, tol =1e-4):
  """Turns a matrix into its three-tuple representation as needed for
     'tf.sparse.SparseTensor'.
  Args:
    matrix - matrix in form of numpy array
  Returns:
    indices, values, denseshape as specified in 'tf.sparse.SparseTensor'
  """
  length = np.sum(matrix!=0)
  indices = np.zeros((length,2))
  values = np.zeros((length))
  k = 0
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      if abs(matrix[i,j]) > tol:
        indices[k,:] = [i,j]
        values[k] = matrix[i,j]
        k+= 1
  return np.array(indices), np.array(values), matrix.shape


def tuple2dense(tuple_):
  """Turns tuple representation of matrix into a dense matrix."""
  return tuple2scipy_sparse(tuple_).toarray()


def tuple2scipy_sparse(tuple_):
  """Turns tuple representation of a matrix into a scipy_sparse matrix."""
  row = np.array([tuple_[0][i,0] for i in range(len(tuple_[1]))])
  column = np.array([tuple_[0][i,1] for i in range(len(tuple_[1]))])
  return sp.coo_matrix((tuple_[1],(row, column)), tuple_[2])


def sparse2tuple(sparse):
    """Turns sparse matrix into tuple representation."""
    r_indices, c_indices, values = sp.find(sparse)
    return np.vstack((r_indices, c_indices)).transpose(), values, sparse.get_shape()
