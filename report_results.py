"""Reads in pickle files with results and evaluates them"""

import matplotlib as plt
import numpy as np
import os
import pickle


## read in results

os.chdir('/Users/sahra/Desktop/results/')
graph_files = [f for f in os.listdir('.')]

for file_iter in graph_files:
    
    file_nr = 4
    file_iter = graph_files[file_nr]
    if file_iter != '.DS_Store':
        with open(file_iter, 'rb') as f:
            x = pickle.load(f)
        exec("%s = x" % (file_iter[:-4]))        
        
        ## print results
        
        # variables = ('train_f1','val_f1', 'test_f1','train_acc','val_acc','test_acc',
            #'per_batch_training_time','per_epoch_training_time', 'per_run_training_time')
            
        variables = ('per_epoch_training_time')
        
        print(file_iter,'\n')
        for v in variables:
            print(v,':')
            print('mean:', np.mean(data[v]))
            print('std:', np.std(data[v]))
            #print('max:', np.max(data[v]))
            #print('min:', np.min(data[v]))
        print('\n\n') 
