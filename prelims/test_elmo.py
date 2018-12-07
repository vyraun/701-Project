import csv
import sys
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_hub as hub

q1 = []
q2 = []

filename = '../glove.6B/glove.6B.50d.txt' #sys.argv[1]

Glove = {}
f = open(filename)

print("Loading Glove vectors.")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    Glove[word] = coefs
f.close()

print("Done.")
X_train = []
X_train_names = []
for x in Glove:
        X_train.append(Glove[x])
        X_train_names.append(x)

X_train = np.asarray(X_train)
X_train_names = np.asarray(X_train_names)

def encode_sentenc(l1, l2, i):

    with tf.Graph().as_default():
      elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
      embeddings_elmo1 = elmo([l1], signature="default", as_dict=True)["default"]
      embeddings_elmo2 = elmo([l2], signature="default", as_dict=True)["default"]

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        h1 = sess.run(embeddings_elmo1)
        h2 = sess.run(embeddings_elmo2)

    h1 = np.asarray(h1)
    h2 = np.asarray(h2)

    return h1, h2


def encode_elmo(l1, l2, i):

    with tf.Graph().as_default():

      embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
      embeddings_sentenc1 = embed([l1])
      embeddings_sentenc2 = embed([l2])

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        h1 = sess.run(embeddings_sentenc1)
        h2 = sess.run(embeddings_sentenc2)

    h1 = np.asarray(h1)
    h2 = np.asarray(h2)

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

    question_1 = []
    question_2 = []

    with open(filename) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"): 
            if skip:
                #print("Line {0}".format(i))
                if line[1]!='' and line[1]!='':
                    q1, q2 = encode_elmo(line[1], line[2], i)
                    question_1.append(q1)
                    question_2.append(q2)
                    values.append(float(line[0]))
                    i = i + 1
                    #if i == 100:
                    #    break
            else:
                skip = True


    labels_2 = [ 1 - int(x) for x in  values]

    return question_1, question_2, labels_2


def return_features(question_1, question_2):
    scores = []
    for i in range(len(question_1)):
        scores.append(np.dot(question_1[i], question_2[i])) # /(LA.norm(question_1[i])*LA.norm(question_2[i])))

    return scores

#### Load th data and get the features

##### Train
question_1_train, question_2_train, labels_train = load_quora("train.tsv")
features_train = return_features(question_1_train, question_2_train)
X_train, y_train = np.asarray(features_train).reshape(-1, 1), labels_train

##### Test
question_1_test, question_2_test, labels_test = load_quora("test.tsv")
features_test = return_features(question_1_test, question_2_test)
X_test, y_test = np.asarray(features_test).reshape(-1, 1), labels_test

##### Learn the classifier

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
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
