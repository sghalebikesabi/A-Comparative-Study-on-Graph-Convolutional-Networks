"""Trains GCN models and tests them"""

import numpy as np
import os
import pickle
import random
from sklearn.preprocessing import normalize
import tensorflow as tf
import time

os.chdir('/Users/sahra/Desktop/Implementation')

from models import GCN
from sparse_help_funcs import sparse2tuple, tuple2scipy_sparse, tuple2dense, matrix2tuple
from preprocess_model_inputs import data_split, get_second_nb
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
file_nr = 2
file_iter =  graph_files[file_nr]
if file_iter != '.DS_Store':
    with open(file_iter, 'rb') as f:
        x = pickle.load(f)
    exec("%s = x" % (file_iter[:-4]))


## run code for one dataset

# read in the necessary variables
exec("data = %s" % (graph_files[file_nr][:-4]))

features = data['features']
features = sparse2tuple(normalize(tuple2scipy_sparse(features), norm='l2', axis=1))
adjacency  = data['adjacency']
dense_adjacency = tuple2dense(data['adjacency'])
dense_features = tuple2dense(features)
labels = data['labels'].toarray()
n, e = labels.shape
if n<e:
   labels = np.transpose(labels)
   n, e = labels.shape


# set hyperparameters 
train_size, val_size, test_size = np.array((0.1*n,0.3*n,0.6*n), dtype=int)
train_per_class = False
num_runs = 100
num_epoch = 300
patience = 10
batch_size = 128
num_batches = int(np.ceil(train_size/batch_size))
k = 3
weight_decay =  5e-4
learn_rate = 0.005
hidden = [16] 
dropout = 0.5
model_func = GCN
n_heads = [8, 1]
rank = 100


# define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=features[2]),
    'labels': tf.placeholder(tf.float32, shape=(None, e)),
    'labels_mask': tf.placeholder(tf.int32, shape = [None]),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
    'k': tf.placeholder(tf.int64),
    'learn_rate': tf.placeholder(tf.float32),
    'weight_decay': tf.placeholder(tf.float32),
}


def construct_feed_dict(features, support, labels, labels_mask, k, learn_rate,
                        weight_decay,  placeholders, dropout):
    """Constructs feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['k']: k})
    feed_dict.update({placeholders['learn_rate']: learn_rate})
    feed_dict.update({placeholders['weight_decay']: weight_decay})
    feed_dict.update({placeholders['dropout']: dropout})
    return feed_dict


def evaluate(features, support, labels, mask, k=1, learn_rate=0., weight_decay=0.,
             placeholders=0, dropout=0):
    """Evaluates trained model."""
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, k,
                                        learn_rate, weight_decay, placeholders, 
                                        dropout)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


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


## train and test models 100 times

for run_iter in range(num_runs):
    
    # train/val/test data splits
    train_idx,train_labels,val_labels,test_labels,train_mask,val_mask,test_mask= data_split(
            labels, train_size, val_size, test_size, train_per_class)
    random_batching = random.sample(train_idx, train_size)
    
    input_dim  =  features[2][1]
    
    # Create model
    model = model_func(placeholders, input_dim, hidden = hidden)

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
          
          # only feed nodes that are needed (in the 2nd order neighborhood)
          needed = get_second_nb(dense_adjacency,batch_indices)
          needed_features = matrix2tuple(dense_features[needed,:])
          needed_adjacency = matrix2tuple(dense_adjacency[needed,:][:,needed])
          needed_batch_labels = batch_labels[needed]
          needed_batch_mask = batch_mask[needed]
          
          feed_dict = construct_feed_dict(needed_features, needed_adjacency, 
                                          needed_batch_labels, needed_batch_mask,
                                          k, learn_rate, weight_decay, placeholders,
                                          0.5)

          # Training step
          outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs],
                          feed_dict=feed_dict)
          train_f1.append(masked_f1(outs[3],needed_batch_labels,needed_batch_mask))
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
                  k != 'save_dict'and k in variables})

with open(graph_files[file_nr][:-4] + '_' + str(model_func)[15:-2] + '.pkl', 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
