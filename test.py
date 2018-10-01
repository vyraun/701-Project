import csv
import sys
import numpy as np

filename = sys.argv[1]

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

question_1 = []
question_2 = []

def encode_bow(l1, l2, i):
    
    question_1.append([0]*50)
    question_2.append([0]*50)

    for x in l1:
        if x in Glove:
            question_1[i] += Glove[x]
    
    for x in l2:
        if x in Glove:
            question_2[i] += Glove[x]        
i = 0

values = []
skip = False

with open("quora_duplicate_questions.tsv") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"): #You can also use delimiter="\t" rather than giving a dialect.
        if skip:
            print("Line {0}".format(i))
            encode_bow(line[3], line[4], i)
            values.append(float(line[5]))
            i = i + 1
        else:
            skip = True

print(len(question_1), len(question_2), len(question_1[0]), len(question_2[0]))      

question_1 = np.asarray(question_1)
question_2 = np.asarray(question_2)
values = [ float(x) for x in  values]

scores = []

for i in range(len(question_1)):
    scores.append(np.dot(question_1[i], question_2[i]))

scores = np.asarray(scores)
predictions = [round(x) for x in scores]

from sklearn.metrics import classification_report

target_names = ['Different', 'Same']
print(classification_report(values, scores, target_names=target_names))



from numpy import linalg as LA

print(LA.norm(scores-values))