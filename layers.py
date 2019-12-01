import numpy as np
import tensorflow as tf


## Layer classes

class Layer():
  """Base layer class. Defines basic API for all layer classes.
  Implementation inspired by keras (http://keras.io) acc to Kipf and Welling (2017).
  Args:
      name - Defines the variable scope of the layer
  Methods:
      _call(inputs) - Defines computation graph of layer
      __call__(inputs) - Wrapper for _call()
  """

  def __init__(self, **kwargs):
      name = kwargs.get('name')
      if not name:
          layer = self.__class__.__name__.lower()
          name = layer + '_' + str(get_layer_number())
      self.name = name
      self.vars = {}
      self.sparse_inputs = False


  def _call(self, inputs):
      return inputs


  def __call__(self, inputs):
      with tf.name_scope(self.name):
          outputs = self._call(inputs)
          return outputs
         
            

class GraphConvolution(Layer):
  """Graph convolutional layer class. Implementation inspired by Kipf and
      Welling (2017) and keras (http://keras.io).
    Args:
      input_dim - Input dimension
      output_dim - Output dimension
      placeholders - Contains inputs for layer
      sparse_inputs - True if inputs are sparse
      act - Activation function
    Methods:
      _call(inputs) - defines computation graph of layer 
  """

  def __init__(self, input_dim, output_dim, placeholders,
               sparse_inputs=False, act=tf.nn.relu, **kwargs):
    super(GraphConvolution, self).__init__(**kwargs)
    
    self.sparse_inputs = sparse_inputs
    self.act = act
    self.support = placeholders['support']
    self.dropout = placeholders['dropout']
    self.num_features_nonzero = placeholders['num_features_nonzero']
    
    with tf.variable_scope(self.name + '_vars'):
      self.vars['weights'] = glorot([input_dim, output_dim])
      self.vars['bias'] = tf.zeros(tf.Variable([output_dim], name='bias'))
      

  def _call(self, inputs):
    # dropout
    if self.dropout is not None:
      if self.sparse_inputs:
        output = tf.sparse.retain(inputs, tf.random_uniform(
            self.num_features_nonzero) > self.dropout) * (1/(1-self.dropout)) 
      else:
        output = tf.nn.dropout(inputs, rate=self.dropout)
    else:
      output = inputs
      
    # convolution
    weighted = dot(output, self.vars['weights'], sparse=self.sparse_inputs)
    output = dot(self.support, weighted, sparse=True)      

    # bias
    output += self.vars['bias']
      
    return self.act(output)
  
    
    
class attn_mech(Layer):
  """Graph attention layer class inspired by Velickovic et al (2018).
    Args:
      n_heads - Number  of attention heads
      out_sz - Ourput Dimension
      bias_mat - Bias matrix that ensures that attention is only computed for 
          neighbor nodes
      activation - Activation function
      nb_nodes - Number of nodes
      dropout - Dropout rate for layer
      coef_drop - Dropout rate for attention mechanism
      is_out - True if last layer 
      name - Unique layer ID  
    Methods:
      _call(inputs) - defines computation graph of layer, concatenates attention 
          heads
      _attn_head - One attention head acc to Velickovic et al (2018).
  """

  def __init__(self, n_heads, out_sz, bias_mat, activation, nb_nodes, 
               dropout=None, coef_drop=0.0, is_out=False, name=None, **kwargs):
    super(attn_mech, self).__init__(**kwargs)
    
    self.vars = {}  
    self.out_dim = out_sz
    self.bias_mat = bias_mat
    self.act = activation
    self.dropout = dropout
    self.coef_drop = coef_drop
    self.n_head = n_heads
    self.is_out = is_out
    self.nb_nodes = nb_nodes


  def _call(self, inputs):
    attns = []
    for _ in range(self.n_head):
        attns.append(self._attn_head(inputs))
    if self.is_out == False:
      h = tf.concat(attns, axis=-1)
    else:
      h = tf.add_n(attns) / self.n_head
    return(h)
    
    
  def _attn_head(self, seq):
    
    if self.dropout is not None:
        seq = tf.nn.dropout(seq, rate = self.dropout)
    
    seq_fts = tf.layers.conv1d(seq, self.out_dim, 1, use_bias=False)

    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
    
    f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
    f_2 = tf.reshape(f_2, (self.nb_nodes, 1))

    f_1 = self.bias_mat * f_1
    f_2 = self.bias_mat * tf.transpose(f_2, [1,0])

    logits = tf.sparse_add(f_1, f_2)
    lrelu = tf.SparseTensor(indices=logits.indices, 
            values=tf.nn.leaky_relu(logits.values), 
            dense_shape=logits.dense_shape)
    coefs = tf.sparse_softmax(lrelu)

    if self.coef_drop != 0.0:
        coefs = tf.SparseTensor(indices=coefs.indices,
                values=tf.nn.dropout(coefs.values, rate=self.coef_drop),
                dense_shape=coefs.dense_shape)
    if self.dropout != 0.0:
        seq_fts = tf.nn.dropout(seq_fts, rate = self.dropout)

    coefs = tf.sparse_reshape(coefs, [self.nb_nodes, self.nb_nodes])
    seq_fts = tf.reshape(seq_fts,[tf.shape(seq_fts)[1],tf.shape(seq_fts)[2]])
    vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
    vals = tf.expand_dims(vals, axis=0)
    vals.set_shape([1, self.nb_nodes, self.out_dim])
    ret = tf.contrib.layers.bias_add(vals)

    return self.act(ret) 



class Dense(Layer):
    """Dense layer inspired by Kipf and Welling (2016)
    Args:
      input_dim - Input dimension
      output_dim - Output dimension
      placeholders - Inputs to layer 
      sparse_inputs - True if inputs are sparse,
      act - activation function  
    Methods:
      _call(inputs) - defines computation graph of layer, concatenates attention 
          heads
      _attn_head - One attention head acc to Velickovic et al (2018).
    """

    def __init__(self, input_dim, output_dim, placeholders, sparse_inputs=False,
                 act=tf.nn.relu, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = placeholders['dropout']
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim])
            self.vars['bias'] = tf.Variable(tf.zeros([output_dim], dtype=tf.float32), name='bias')


    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = tf.sparse.retain(inputs, tf.random_uniform(
            self.num_features_nonzero) > self.dropout) * (1/(1-self.dropout)) 
        else:
            x = tf.nn.dropout(x,rate=self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        output += self.vars['bias']

        return self.act(output)



## some help functions

# global variable counting layers
nr_layers = 0

def get_layer_number():
  """Helper function for getting unique layer IDs."""
  global nr_layers
  nr_layers += 1
  return nr_layers


def glorot(shape):
    """Weight initilization of Glorot and Bengio (2010)."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, 
                                dtype=tf.float32)
    return tf.Variable(initial, name='weights')


def dot(x, y, sparse=False):
  """Wrapper for tf.matmul (sparse vs dense) acc. to Kipf and Welling (2016)."""
  if sparse:
      res = tf.sparse_tensor_dense_matmul(x, y)
  else:
      res = tf.matmul(x, y)
  return res

