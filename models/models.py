import tensorflow as tf

from layers import GraphConvolution, attn_mech, Dense
from metrics import masked_accuracy, masked_softmax_cross_entropy


class Model(object):
  """Base model class. Defines basic API for all model classes. Implementation 
  inspired Kipf and Welling (2017).
  Args:
      name - Defines the variable scope of the layer
  Methods:
      build - Wrapper for _build(), appends layers, calculates loss, and set minimizer
  """

  
  def __init__(self):
          
      self.name = self.__class__.__name__

      self.vars = {}
      self.placeholders = {}

      self.layers = []
      self.activations = []

      self.inputs = None
      self.outputs = None

      self.loss = 0
      self.accuracy = 0
      self.topk_accuracy = 0
      self.f1 = 0

      self.optimizer = None
      self.opt_op = None
      
  def build(self):
      with tf.variable_scope(self.name):
          self._build()

      # Build sequential layer model
      self.activations.append(self.inputs)
      for layer in self.layers:
          hidden_l = layer(self.activations[-1])
          self.activations.append(hidden_l)
      self.outputs = self.activations[-1]

      # Store model variables for easy access
      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.vars = {var.name: var for var in variables}

      # Build metrics
      self._loss()
      self._accuracy()
      self._topk_accuracy()

      self.opt_op = self.optimizer.minimize(self.loss)



class GCN(Model):
  """Graph convolutional network. Implementation inspired by Kipf and Welling 
     (2017) and keras (http://keras.io).
    Args:
      placeholders - Inputs for the model
      input_dim - Input dimension
      hidden - Number of hidden nodes for each hidden layer
    Methods:
      _loss - Calculates loss
      _accuracy - Calcultes accuracy
      _topk_accuracy  - Calculates top k accuracy
      _build - Builds the model
  """

  def __init__(self, placeholders, input_dim, hidden, **kwargs):
    super(GCN, self).__init__(**kwargs)

    self.inputs = placeholders['features']
    self.input_dim = input_dim
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders
    self.k = placeholders['k']
    self.learn_rate = placeholders['learn_rate']
    self.weight_decay = placeholders['weight_decay']
    self.n_hidden = hidden

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

    self.build()


  def _loss(self):
    # Weight decay loss
    for var in self.layers[0].vars.values():
        self.loss += self.weight_decay * tf.nn.l2_loss(var)
    # Cross entropy error
    self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])


  def _accuracy(self):
    self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                  self.placeholders['labels_mask'])


  def _topk_accuracy(self):
    self.topk_accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                  self.placeholders['labels_mask'],self.k)


  def _build(self):
    # length of n_hidden specifies number of hidden layers and its elements 
    # the number of nodes
    for i in range(len(self.n_hidden)):
        
      is_sparse = False
      if i ==0:
        input_dim_ = self.input_dim
        is_sparse = True
      else:
        input_dim_ = self.n_hidden[i-1]
      output_dim_ = self.n_hidden[i]

      self.layers.append(GraphConvolution(input_dim=input_dim_,
                                          output_dim=output_dim_,
                                          placeholders=self.placeholders,
                                          act=tf.nn.relu,
                                          sparse_inputs=is_sparse))


    self.layers.append(GraphConvolution(input_dim= self.n_hidden[-1],
                                        output_dim=self.output_dim,
                                        placeholders=self.placeholders,
                                        act=lambda x: x))
    
    

class GAT(Model):
  """Graph attention model. Implementation inspired by Velickovic et al (2018) 
      and keras (http://keras.io).
    Args:
      placeholders - Inputs for the model
      input_dim - Input dimension
      nb_nodes - Number of nodes
      nb_classes - Number of classes
      hidden - Number of hidden nodes for each hidden layer
      activation - Activation function
    Methods:
      _loss - Calculates loss
      _accuracy - Calcultes accuracy
      _topk_accuracy  - Calculates top k accuracy
      _build - Builds the model
  """
  
  def __init__(self, placeholders, input_dim, nb_nodes, nb_classes, hidden, 
               activation=tf.nn.elu, **kwargs):
    super(GAT, self).__init__(**kwargs)

    self.inputs = placeholders['features'] 
    self.attn_drop = placeholders['attn_drop']
    self.ffd_drop = placeholders['dropout']
    self.bias_mat = placeholders['support']
    self.activation = activation
    self.weight_decay = placeholders['weight_decay']
    self.learn_rate = placeholders['learn_rate']
    self.nb_classes = nb_classes
    self.nb_nodes = nb_nodes
    self.num_features_nonzero = placeholders['num_features_nonzero']
    self.k = placeholders['k']
    self.hid_units = hidden
    
    self.lbl_in = placeholders['labels']
    self.msk_in = placeholders['labels_mask']

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
    
    self.build()
    
    
  def _loss(self):
    # Weight decay loss
    vars = tf.trainable_variables()
    for var in vars: 
      if var.name not in ['bias', 'gamma', 'b', 'g', 'beta']:
        self.loss += self.weight_decay * tf.nn.l2_loss(tf.cast(var,tf.float32))
    # Cross entropy error
    self.log_resh = tf.reshape(self.outputs, [-1, self.nb_classes])
    self.lab_resh = tf.reshape(self.lbl_in, [-1, self.nb_classes])
    self.msk_resh = tf.reshape(self.msk_in, [-1])
    self.loss += masked_softmax_cross_entropy(self.log_resh, self.lab_resh, self.msk_resh) 

    
  def _accuracy(self):
    self.accuracy = masked_accuracy(self.log_resh, self.lab_resh, self.msk_resh)


  def _topk_accuracy(self):
    self.topk_accuracy = masked_accuracy(self.log_resh, self.lab_resh, self.msk_resh, self.k)


  def _build(self):
      n_heads = [8, 1]
      self.layers.append(attn_mech(n_heads=n_heads[0], out_sz=self.hid_units[0], 
                 bias_mat=self.bias_mat, activation=self.activation, dropout=self.ffd_drop, 
                 coef_drop=self.attn_drop, nb_nodes = self.nb_nodes))
      
      for i in range(1, len(self.hid_units)):
          self.layers.append(attn_mech(n_heads=n_heads[i], out_sz=self.hid_units[i], 
                 bias_mat=self.bias_mat, activation=self.activation,dropout=self.ffd_drop, 
                 coef_drop=self.attn_drop, nb_nodes = self.nb_nodes))

      self.layers.append(attn_mech(n_heads=n_heads[-1], out_sz=self.nb_classes, 
                         bias_mat=self.bias_mat, activation=lambda x: x,
                         dropout=self.ffd_drop, coef_drop=self.attn_drop, 
                         is_out=True, nb_nodes = self.nb_nodes))



class FastGCN(Model):
    """Graph convolutional network inspired by Chen et al (2018). 
    Args:
      placeholders - Inputs for the model
      input_dim - Input dimension
      hidden - Number of hidden nodes for each hidden layer
    Methods:
      _loss - Calculates loss
      _accuracy - Calcultes accuracy
      _topk_accuracy  - Calculates top k accuracy
      _build - Builds the model
    """

    
    def __init__(self, placeholders, input_dim, hidden, **kwargs):
        super(FastGCN, self).__init__(**kwargs)
        self.inputs = placeholders['features'] # A*X for the bottom layer, not original feature X
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.support = placeholders['support']
        self.learning_rate =  placeholders['learn_rate']
        self.n_hidden = hidden
        self.weight_decay = placeholders['weight_decay']
        self.k = placeholders['k']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
        
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], None, 1)
        
    def _topk_accuracy(self):
        self.topk_accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], None, self.k)

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=self.n_hidden[0],
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False))

        self.layers.append(GraphConvolution(input_dim=self.n_hidden[0],
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.support,
                                            act=lambda x: x,
                                            dropout=True))
