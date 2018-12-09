
# coding: utf-8

# In[23]:


import io
import numpy as np
import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))

# Load Vectors

# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data

# # Load German Vectors

# German = load_vectors('/home/vraunak/Desktop/wembeddings/wiki.multi.en.vec.txt')

German = {}
f = open('glove.6B.300d.txt')
print("Loading Glove vectors.")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    German[word] = coefs
f.close()

X_train_German = []
X_train_German_names = []
for x in German.keys():
        X_train_German.append(list(German[x]))
        X_train_German_names.append(x)
        
print("Loaded Vectors and Words into Separate Lists with the same index for German.")        

X_train_German = np.asarray(X_train_German, dtype=np.float32)
print(X_train_German.shape)

X_train_German_names = np.asarray(X_train_German_names)
print(X_train_German_names.shape)

vocab_size = len(X_train_German_names)
print("Vocab Size = ", vocab_size)

# Construct word to index and index to word dictionary

word_to_index_german = {}
index_to_word_german = {}

for i, x in enumerate(X_train_German_names):
    word_to_index_german[x] = i
    index_to_word_german[i] = x
    
print("Index to Word and Word to Index Mapping Created for German.")

vocab_size = len(X_train_German_names)
print("Vocab Size = ", vocab_size)


# In[83]:


import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


# In[84]:


# Construct bow vector for a sentence

def get_bow(text):
    #tokenized_text = nltk.word_tokenize(sentence)
    #filtered_text = [w for w in tokenized_text if not w in stop_words] # remove stop words
    #filtered_text = [word.lower() for word in tokenized_text if word.isalpha()] # remove puntuation
    bow_sentence = text.split()
    bow_vector = np.zeros(50)
    for x in bow_sentence:
        if x in word_to_index_german:
            bow_vector += German[x]

    #vector = np.dot(X_train_German.transpose(), bow_vector)
    return bow_vector


# In[118]:

# In[24]:


print("Size = ", i)

print("Creating Index for Fast Nearest Neighbour Search in the Common Space")

d = 300                         # dimension
nb = len(X_train_German_names)                     # database size
np.random.seed(1234)            # make reproducible

import faiss                   # make faiss available
index = faiss.IndexFlatIP(d)  # INner Prouct

index.add(X_train_German)       # add vectors to the index

print("Index Built = {0}, Total Vectors = {1}".format(index.is_trained, index.ntotal))


# In[25]:


def return_knn(words):

    X_search = np.asarray([German[word] for word in words], dtype=np.float32).reshape((len(words), 300))
    k = 2 
    D, I = index.search(X_search, k)
    knn_dict = {}
    nns = X_train_German_names[I]

    for i, word in enumerate(words):
        knn_dict[word] = nns[i][1]

    return knn_dict


# In[26]:


def knn_quora(filename):
    i = 0
    knn_dict = {}

    values = []
    skip = False

    question_1 = []
    question_2 = []
    j = 0

    with open(filename) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"): 
            j = j + 1
            
            print("Line ", j)
            
            if skip:
                if j == 20:
                    break
                if line[3]!='' and line[4]!='' and line[5]!='':
                    q1, q2 = nltk.word_tokenize(line[3]), nltk.word_tokenize(line[4])
                    q1 = [word.lower() for word in q1 if word.isalpha()]
                    q2 = [word.lower() for word in q2 if word.isalpha()]
                    for word in q1:
                        if word in German:
                            if word in knn_dict:
                                continue
                            else:
                                knn_dict[word] = 1
                    for word in q2:
                        if word in German:
                            if word in knn_dict:
                                continue
                            else:
                                knn_dict[word] = 1
            else:
                skip = True
                
    return knn_dict.keys()


# In[ ]:


import csv
quora_file = 'quora_duplicate_questions.tsv'
print("Quora Run STarted")

knn_dict = return_knn(knn_quora(quora_file))

#knn_dict = {}

#for i, word in enumerate(words):
#    knn_dict[word] = nns[i][0][0]

# In[ ]:

print(knn_dict)

import pickle

with open('1final_knn_quora_py.pickle', 'wb') as handle:
    pickle.dump(knn_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
