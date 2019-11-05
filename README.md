# A Comparative Study on Graph Convolutional Networks

## Abstract
Since Kipf and Welling (2016) introduced graph convolutional networks in the semi-supervised node classification setting, several extensions have been proposed. For assessing the performance of these methods, the majority of the literature relies on an experimental setup introduced by Yang et al. (2016) which consists mainly of small, sparse graphs. The purpose of this study is to analyze the performance of different GCN methods based on a set of data which is complex with respect to size and sparsity. In particular, we focus on three GCN architectures: basic GCNs, FastGCNs and graph attention networks. Micro F1 scores and training time per epoch are reported for three real-world datasets and five simulated graphs. We find that all methods are comparable in prediction performance. While GAT achieves accuracies up to three percent points faster than the basic GCN framework, it is up to 20 times slower. FastGCN is the fastest method for large and dense datasets if training time per batch is considered. 

## Repository Structure
Please refer to the following for documentation on the files and directories in this repository.

### 'report.pdf'
This file contains the report in pdf format.

### Folder 'Implementation'
This folder contains all the code and data used in our study.

#### Folder 'edit_data'
This folder contains the code for generating random graphs as specified in the report and transforming the original data into a dictionary with keys 'adjacency' (value: normalised adjacency matrix as sparse matrix), 'labels' (value: one-hot encoded labels as sparse matrix) and 'features' (value: feature matrix as sparse matrix)
* 'extract_stackoverflow_data.sql' (SQL Query for downloading data)
* 'read_in_stackoverflow.py' (transforms stackoverflow data as described above)
* 'read_in_citeseer.py' (transforms citesser data as described above)
* 'read_in_reddit.py' (transforms reddit data as described above)
* 'generate_random_graphs.py' (generates 5 random graphs with different size and node degree)

#### '__init__.py'
Empty file which allows imports from outside this directory.

#### 'layers.py'
This file defines layer classes for the models.

#### 'metrics.py'
This file defines metric functions (loss, accuracy, etc).

#### 'models.py'
This file defines the model classes (GCN, FastGCN, GAT).

#### 'preprocess_model_inputs.py'
This file contains functions for transforming the graph data into appropriate
model input.

#### 'report_results.py'
This file reads in the pickle files that contain the results and prints out relevant measures.

#### 'run_FastGCN.py'
This file trains the FastGCN on the datasets and tests it.

#### 'run_GCN.py'
This file trains the GCN on the datasets and tests it.

#### 'run_GAT.py'
This file trains the GAT on the datasets and tests it.

#### 'sparse_help_funcs.py'
This file contains some functions for handling the transformations between sparse
and dense matrices.
