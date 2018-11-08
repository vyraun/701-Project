import csv
import sys
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA

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

def encode_subspace(l1, l2, i):

    question_1.append( np.column_stack(tuple([Glove[x] if x in Glove else np.zeros(50) for i, x in enumerate(l1.split())])) )
    question_2.append( np.column_stack(tuple([Glove[x] if x in Glove else np.zeros(50) for i, x in enumerate(l2.split())])) )
    

i = 0

values = []
skip = False

with open("quora_duplicate_questions.tsv") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"): #You can also use delimiter="\t" rather than giving a dialect.
        if skip:
            print("Line {0}".format(i))
            #encode_bow(line[3], line[4], i)
            if line[3]!='' and line[4]!='':
                encode_subspace(line[3], line[4], i)
                values.append(float(line[5]))
                i = i + 1
                if i == 100:
                    break
        else:
            skip = True

print(len(question_1), len(question_2), len(question_1[0]), len(question_2[0]))      

#question_1 = np.asarray(question_1)
#question_2 = np.asarray(question_2)
labels = [ 1.0 - float(x) for x in  values]
labels_2 = [ 1 - int(x) for x in  values]

scores = []

for i in range(len(question_1)):
    #print(question_1[i], question_2[i], LA.norm(question_1[i]))
    #scores.append(np.dot(question_1[i], question_2[i])/(LA.norm(question_1[i])*LA.norm(question_2[i])))

    q1, q2 = question_1[i], question_2[1]

    p1 = PCA(n_components=4)
    p1.fit(q1.T)
    cp1 = np.column_stack(tuple([x for i, x in enumerate(p1.components_)]))
    basis1, r1 = LA.qr(cp1)

    p2 = PCA(n_components=4)
    p2.fit(q2.T)
    cp2 = np.column_stack(tuple([x for i, x in enumerate(p2.components_)]))
    basis2, r2 = LA.qr(cp2)
    
    u, s, vh = LA.svd(np.dot(basis1.T, basis2), full_matrices=True)

    scores.append(np.sqrt(sum([t*t for t in s])))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = np.asarray(scores).reshape(-1, 1), labels_2
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)
clf.fit(X, y)
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)

print("Accuracy (train) = {}".format(accuracy * 100))

scores = np.asarray(scores, dtype='float32')
print(scores)
predictions = [round(x) for x in scores]

c = 0

for x, y in zip(predictions, labels):
    if x==y:
        c += 1
    print(x, y)

print("Accuracy = {}".format(c/len(scores)))