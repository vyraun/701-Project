import torch
import numpy
import time
from pymagnitude import *

# This is the code from ALlen NLP, used in function encode_allenai_elmo

#from allennlp.modules.elmo import Elmo, batch_to_ids
#options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#elmo = Elmo(options_file, weight_file, 2, dropout=0)

# THis is the code to use bert
#from bert_serving.client import BertClient
#bc = BertClient()

# This is elmo from pymagnitue
elmo_vecs = Magnitude('elmo_2x1024_128_2048cnn_1xhighway_weights_GoogleNews_vocab.magnitude')

import csv
import sys
import pickle
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_hub as hub

q1 = []
q2 = []

# filename = '../glove.6B/glove.6B.50d.txt' #sys.argv[1]

# Glove = {}
# f = open(filename)

# print("Loading Glove vectors.")
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     Glove[word] = coefs
# f.close()

# print("Done.")
# X_train = []
# X_train_names = []
# for x in Glove:
#         X_train.append(Glove[x])
#         X_train_names.append(x)

# X_train = np.asarray(X_train)
# X_train_names = np.asarray(X_train_names)

def return_single_elmo(l):
    sentence  = elmo_vecs.query([l])
    return sentence[0]
    #return np.mean(sentence)

def batcher(l1, l2, batch=50):
    pass

def encode_bert(l):
    return bc.encode(l) #, bc.encode(l2)[0]

def encode_allenai_elmo(l):
    sentences = [l]
    print(l)
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    return embeddings

def encode_elmo(l1, l2, type="default"):

    print("Elmo Encoding Started")

    with tf.Graph().as_default():
      elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
      embeddings_elmo1 = elmo(l1, signature="default", as_dict=True)[type] # elmo
      embeddings_elmo2 = elmo(l2, signature="default", as_dict=True)[type]

      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.tables_initializer())

          h1 = sess.run(embeddings_elmo1)
          h2 = sess.run(embeddings_elmo2)

      h1 = np.asarray(h1)
      h2 = np.asarray(h2)
    
    print("Elmo Encoding Started")

    return h1, h2


def encode_sent(l1, l2):

    print("SEntence Encoding Started")

    with tf.Graph().as_default():

      embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
      embeddings_sentenc1 = embed(l1)
      embeddings_sentenc2 = embed(l2)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        h1 = sess.run(embeddings_sentenc1)
        h2 = sess.run(embeddings_sentenc2)

    h1 = np.asarray(h1)
    h2 = np.asarray(h2)
    
    print("Sentence Encoding Done")

    return h1, h2


def encode_bow(l1, l2, i):
    
    q1 = [0]*50
    q2 = [0]*50

    for x in l1:
        if x in Glove:
            q1 += Glove[x]
    
    for x in l2:
        if x in Glove:
            q2 += Glove[x]

    return q1, q2

def encode_subspace(l1, l2, i):

    q1 =  np.column_stack(tuple([Glove[x] if x in Glove else np.zeros(50) for i, x in enumerate(l1.split())]))
    q2 = np.column_stack(tuple([Glove[x] if x in Glove else np.zeros(50) for i, x in enumerate(l2.split())]))

    return q1, q2
    

def load_quora(filename):
    i = 0

    values = []
    skip = False

    t1, t2 = [], []

    question_1 = []
    question_2 = []
    
    cur_batch_q1, cur_batch_q2 = [], []
    
    print("Data Reading Started")

    with open(filename) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"): 
            if skip:
                #if len(cur_batch_q1) >= 50:
                #   start_time = time.time()
                #   vec1, vec2 = encode_bert(cur_batch_q1), encode_bert(cur_batch_q2)
                #   for vec in vec1:
                #       question_1.append(vec)
                #   for vec in vec2:
                #       question_2.append(vec)
                #   del cur_batch_q1[:]
                #   del cur_batch_q2[:]
                #   print("--- %s seconds for this batch---" % (time.time() - start_time))
                 
                    #print("Line {0}".format(i))
                    if line[1]!='' and line[2]!='':                        
                        q1, q2 = line[1], line[2]
                        #cur_batch_q1.append(q1)
                        #cur_batch_q2.append(q2)
                        #print(q1, q2)
                        question_1.append(return_single_elmo(q1)) # encode_allenai_elmo(q1)
                        question_2.append(return_single_elmo(q2)) # encode_allenai_elmo(q2)
                        values.append(float(line[0]))
                        i = i + 1
                        print(i)
                        #if i == 10:
                        #    break
            else:
                skip = True
    #if len(cur_batch_q1) > 0:
    #    vec1, vec2 = encode_bert(cur_batch_q1), encode_bert(cur_batch_q2)
    #    for vec in vec1:
    #        question_1.append(vec)
    #    for vec in vec2:
    #        question_2.append(vec)
    #    del cur_batch_q1[:]
    #    del cur_batch_q2[:]
        
    print("Data Read")
    labels_2 = [ 1 - int(x) for x in  values]
    #question_1, question_2 =  encode_bert(t) # encode_allenai_elmo(t1), encode_allenai_elmo(t2) 
    return question_1, question_2, labels_2


def return_features(question_1, question_2):
    scores = []
    for i in range(len(question_1)):
        scores.append(np.dot(question_1[i], question_2[i])) # /(LA.norm(question_1[i])*LA.norm(question_2[i])))
        #print(question_1[i].reshape(-1, 1).shape)
        #print(question_1[i].reshape(-1, 1).shape)   # np.concatenate((a, b), axis=0)
        #scores.append(np.concatenate((question_1[i], question_2[i]), axis=0))

    return scores

#### Load th data and get the features

##### Train
question_1_train, question_2_train, labels_train = load_quora("train.tsv")

with open('question_1_train.pkl', 'wb') as f:
     pickle.dump(question_1_train, f)

with open('question_2_train.pkl', 'wb') as f:
     pickle.dump(question_2_train, f)     
     
features_train = return_features(question_1_train, question_2_train)
X_train, y_train = np.asarray(features_train).reshape(-1, 1), labels_train

##### Test
question_1_test, question_2_test, labels_test = load_quora("test.tsv")

with open('question_1_test.pkl', 'wb') as f:
     pickle.dump(question_1_test, f)

with open('question_2_test.pkl', 'wb') as f:
     pickle.dump(question_2_test, f)   

features_test = return_features(question_1_test, question_2_test)
X_test, y_test = np.asarray(features_test).reshape(-1, 1), labels_test

##### Learn the classifier

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced', multi_class='multinomial').fit(X_train, y_train) #class_weight : 'balanced'
#clf = tree.DecisionTreeClassifier().fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)

y_pred_test = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy (train) = {}".format(accuracy_train * 100))
print("Accuracy (test) = {}".format(accuracy_test * 100))


classes = ['Duplicate', 'Not-Duplicate']

# import matplotlib.pyplot as plt 

# plt.plot(list(range(len(y))), y)
# plt.plot(list(range(len(y))), y_pred)
# plt.show()

# from sklearn.linear_model import LogisticRegression
# from yellowbrick.classifier import DiscriminationThreshold
# from yellowbrick.classifier import ClassBalance

# # Instantiate the classification model and visualizer
# logistic = LogisticRegression()
# visualizer = DiscriminationThreshold(logistic)

# # visualizer = ClassBalance(forest, classes=classes)
# visualizer.fit(X, y)  # Fit the training data to the visualizer
# visualizer.poof()     # Draw/show/poof the data
