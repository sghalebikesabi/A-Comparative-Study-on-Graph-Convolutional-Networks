"""Reads in stackoverflow data as downloaded from https://data.stackexchange.com/stackoverflow/query/new
   Please refer to 'extract_stackoverflow_data.sql' file for query informaiton."""

from itertools import compress
import tensorflow as tf
import numpy as np
import operator
import os
import pandas as pd
import pickle
import re
import scipy.sparse as sp
import zipfile

from sparse_help_funcs import sparse2tuple
from preprocess_model_inputs import norm_adj


## extract zipfiles

os.chdir('/Users/sahra/Desktop/Implementation/original_data')
zip_ref = zipfile.ZipFile("stackoverflow.zip", 'r')
zip_ref.extractall()
zip_ref.close()


## read in data

questions = pd.read_csv('questions.csv',sep=',',header=0)
answers = pd.read_csv('answers.csv',sep=',',header=0)
comments = pd.read_csv('comments.csv',sep=',',header=0)
votes = pd.read_csv('votes.csv',sep=',',header=0)


## extract most common tags

# save postIDs and tags
n_old = questions.shape[0]
postIds_old = questions['Id']
tags_old = []
for i in range(n_old):
    tags_with_empty = re.split('<|>| ',questions['Tags'][i])
    keep = [tags_with_empty[j]!="" for j in range(len(tags_with_empty))]
    splitted_tags = list(compress(tags_with_empty, keep))
    tags_old.append(splitted_tags)

# save all possible tags
all_tags = set()
for i in range(n_old):
    all_tags = all_tags|set(tags_old[i])

# count occurrences
all_tags_count = dict.fromkeys(all_tags, 0)
for i in range(n_old):
    new_words = tags_old[i]
    for j in new_words:
        all_tags_count[j] += 1

# only keep most occurrent tags
nr_tags = 30
top_tags = sorted(all_tags_count.items(), key=operator.itemgetter(1),
                  reverse=True)[:nr_tags]
top_tags = set([top_tags[i][0] for i in range(len(top_tags))])

postIds_list = []
tags_list = []
kept = []
for i in range(n_old):
    current_tags_in_top = list(set(tags_old[i]) & top_tags)
    if len(current_tags_in_top)==1:
        postIds_list.append(postIds_old[i])
        tags_list.append(current_tags_in_top)
        kept.append(i)


## construct the graph

n = len(postIds_list)
users_list = []

# UserIds per post
for i in range(n):
    users_ = [questions['LastEditorUserId'][i],
              answers['OwnerUserId'][answers['ParentId'] == postIds_list[i]],
              comments['UserId'][comments['PostId'] == postIds_list[i]],
              votes['UserId'][votes['PostId'] == postIds_list[i]]]
    users_list.append([questions['OwnerUserId'][i]])
    for j in users_:
        if isinstance(j, pd.Series):
            if j.empty == False:
                for k in j.values:
                    users_list[i].append(k)
        elif str(j)!='nan':
            users_list[i].append(j)

# all UserIds
all_users = set()
for i in range(n):
    all_users = all_users|set(users_list[i])
all_users = {x for x in all_users if x==x}

# for each user save corresponding postIds
users_posts = dict.fromkeys(all_users, {})
for i in range(n):
    new_users = users_list[i]
    for j in new_users:
        if str(j)!='nan':
            users_posts[j] = set(users_posts[j])|set([postIds_list[i]])

# adjacency matrix
adjacency = sp.identity(n).tolil()
postIds_dict = {k: v for k, v in zip(postIds_list, range(n))}
for key,values in users_posts.items():
    for postId1 in values:
        for postId2 in values:
            adjacency[postIds_dict[postId1],postIds_dict[postId2]] = 1
n_edges = sp.csr_matrix.count_nonzero(adjacency.tocsr()) - n
adjacency = sparse2tuple(norm_adj(adjacency))

# features - bag of words model
post_bodies = questions['Body'][kept]
vocab_size = 1000
tokenize = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(post_bodies)
features = tokenize.texts_to_matrix(post_bodies)
features = sparse2tuple(sp.lil_matrix(features))


# labels
labels = sp.lil_matrix(np.zeros((n,nr_tags)))
for i in range(n):
    for j in tags_list[i]:
        labels[i,list(top_tags).index(j)] = 1

## dump dictionary as pickle file

os.chdir('/Users/sahra/Desktop/Implementation')
stackoverflow = {'adjacency':adjacency,'labels':labels,'features':features}
with open('graph_data/stackoverflow.pkl','wb') as output:
  pickle.dump(stackoverflow, output, pickle.HIGHEST_PROTOCOL)
