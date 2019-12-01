"""Trains GCN models and tests them"""

import numpy as np
import os
import pickle
import random
from sklearn.preprocessing import normalize
import tensorflow as tf
import time

os.chdir('/Users/sahra/Desktop/Implementation')

from models import GAT
from sparse_help_funcs import sparse2tuple, tuple2scipy_sparse, tuple2dense
from preprocess_model_inputs import data_split, preprocess_adj_bias, column_prop
from metrics import masked_f1


## Set random seeds

seed = 449
np.random.seed(seed)
tf.set_random_seed(seed)


## uncomment following lines if run for the first time: generate and edit data

# import edit_data.generate_random_graphs
# import edit_data.read_in_citeseer
# import edit_data.read_in_stackoverflow
# import edit_data.read_in_reddit


## read  in data

os.chdir('/Users/sahra/Desktop/Implementation/graph_data')
graph_files = [f for f in os.listdir('.')]

#for file_iter in graph_files:
file_nr = 0
file_iter =  graph_files[file_nr]
if file_iter != '.DS_Store':
    with open(file_iter, 'rb') as f:
        x = pickle.load(f)
    exec("%s = x" % (file_iter[:-4]))


## run code for one dataset

# read in the necessary variables
exec("data = %s" % (graph_files[file_nr][:-4]))

features = data['features']
features = normalize(tuple2scipy_sparse(features), norm='l2', axis=1).toarray()
adjacency = data['adjacency']
labels = data['labels'].toarray()
n, e = labels.shape
if n<e:
   labels = np.transpose(labels)
   n, e = labels.shape
adjacency = tuple2scipy_sparse(adjacency)
features= features[np.newaxis]
adjacency = preprocess_adj_bias(adjacency)


# set hyperparameters 
train_size, val_size, test_size = np.array((0.1*n,0.3*n,0.6*n), dtype=int)
train_per_class = False
num_runs = 1
num_epoch = 10
patience = 30
batch_size = train_size
num_batches = int(np.ceil(train_size/batch_size))
k = 3
weight_decay =  5e-4
learn_rate = 0.005
hidden = [16] 
dropout = 0.5
model_func = GAT
n_heads = [8, 1]
rank = 100


# define placeholders
placeholders = {
    'support' : tf.sparse_placeholder(dtype=tf.float32),
    'features': tf.placeholder(tf.float32, shape=(1, n, features.shape[2])),
    'labels': tf.placeholder(tf.float32, shape=(1, n, e)),
    'labels_mask': tf.placeholder(tf.int32, shape = (1, n)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
    'k': tf.placeholder(tf.int64),
    'learn_rate': tf.placeholder(tf.float32),
    'weight_decay': tf.placeholder(tf.float32),
    'attn_drop' : tf.placeholder(dtype=tf.float32, shape=()),
    'training' : tf.placeholder(dtype=tf.bool, shape=())
}
    

def construct_feed_dict(features, support, labels, labels_mask, k, learn_rate,
                        weight_decay,  placeholders, dropout, model_func, 
                        attn_drop=0.0, is_train=False):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: 0})
    feed_dict.update({placeholders['k']: k})
    feed_dict.update({placeholders['learn_rate']: learn_rate})
    feed_dict.update({placeholders['weight_decay']: weight_decay})
    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['attn_drop']: attn_drop})
    feed_dict.update({placeholders['training']: is_train})
    feed_dict.update({placeholders['support']: support})
    return feed_dict


# define variables
train_f1 = []
val_f1 = []
test_f1 = []
train_acc = []
val_acc = []
test_acc = []
per_batch_training_time = []
per_epoch_training_time = []
per_run_training_time = []


def evaluate(features, support, labels, mask=0, k=1, learn_rate=0., weight_decay=0.,
             placeholders=0, dropout=0, attn_drop = 0.0, is_train = False):
    """Evaluates trained model."""
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, k,
                                        learn_rate, weight_decay, placeholders, 
                                        dropout, attn_drop, is_train)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


## train and test models 100 times

for run_iter in range(num_runs):
    
    # train/val/test data splits
    train_idx,train_labels,val_labels,test_labels,train_mask,val_mask,test_mask= data_split(
            labels, train_size, val_size, test_size, train_per_class)
    random_batching = random.sample(train_idx, train_size)
    val_labels = val_labels[np.newaxis]
    test_labels = test_labels[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]
    
    # Create model
    model = model_func(placeholders, input_dim = features.shape[-1], nb_nodes = n, 
                       nb_classes =  e, hidden = hidden)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    t_run = time.time()
    cost_val = []
    # run epochs
    for epoch in range(num_epoch):

        t_epoch = time.time()

        for batch in range(num_batches):
          t_batch = time.time()
          
          # batch data
          batch_indices = random_batching[batch*batch_size:min(n,(batch+1)*batch_size)]
          batch_mask = np.zeros((n), dtype = bool)
          batch_labels = np.zeros((n,e))          
          batch_labels[batch_indices,:] = train_labels[batch_indices,:]
          batch_mask[batch_indices] = train_mask[batch_indices]
          batch_mask=batch_mask[np.newaxis]
          batch_labels=batch_labels[np.newaxis]
          
          # Construct feed dictionary
          feed_dict = construct_feed_dict(features, adjacency, batch_labels, batch_mask,
                                          k, learn_rate, weight_decay, placeholders,
                                          0.5, 0.6, True)

          # Training step
          outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs],
                          feed_dict=feed_dict)
          train_f1.append(masked_f1(outs[3],batch_labels,batch_mask))
          train_acc.append(outs[2])

          # Validation
          cost, acc, preds, duration = evaluate(features, adjacency, val_labels, 
                                        val_mask, k, learn_rate, weight_decay, 
                                        placeholders,0)
          cost_val.append(cost)
          val_f1.append(masked_f1(preds, val_labels, val_mask))
          val_acc.append(acc)

          per_batch_training_time.append(time.time()-t_batch)

        per_epoch_training_time.append(time.time()-t_epoch)

        # early stopping
        if epoch > patience and cost_val[-1] > np.mean(cost_val[-(10+1):-1]):
              break

    per_run_training_time.append(time.time()-t_run)

        
    # Testing
    test_cost, test_acc1, test_preds, test_duration = evaluate(features, adjacency,
             test_labels, test_mask, k, learn_rate, weight_decay, placeholders,0)
    test_f1.append(masked_f1(test_preds,test_labels,test_mask))
    test_acc.append(test_acc1)
    print("Test set results: run=",'%03d' % (run_iter + 1),"loss=", "{:.5f}".format(test_cost),
        ", accuracy=", "{:.5f}".format(test_acc1), ", time=", "{:.5f}".format(test_duration))


##  save  results
os.chdir('/Users/sahra/Desktop/Implementation/results')


variables = ('train_f1','val_f1', 'test_f1','train_acc','val_acc','test_acc',
             'per_batch_training_time','per_epoch_training_time', 'per_run_training_time',
             'train_size','val_size','test_size','train_per_class','num_runs',
             'num_epoch','patience','batch_size','num_batches','weight_decay',
             'learn_rate','hidden','dropout','model_func')

save_dict = {}
save_dict.update({k:v for k,v in locals().copy().items() if k[:2] != '__' and 
                  k != 'save_dict' and k in variables})

with open(graph_files[file_nr][:-4] + '_' + str(model_func)[15:-2] + '.pkl', 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
