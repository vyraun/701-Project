import csv
import sys
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA


i = 0
zero, one = 0, 0

lens = 0
ne = []

import spacy

nlp = spacy.load('en_core_web_sm')

skip = False

with open("questions.csv") as tsv:
    for line in csv.reader(tsv): #You can also use delimiter="\t" rather than giving a dialect.
        if skip:
            if line[3]!='' and line[4]!='' and line[0]!='':
                i = i + 1
                if line[5]=='0':
                    zero = zero + 1
                else:
                    one = one + 1
                
                #print(line[3])
                ne.append(len(nlp(line[3]).ents))
                ne.append(len(nlp(line[4]).ents))

                #ne +=  
                #ne += len(nlp(line[4]))

                #ne.append(len(line[3].split())) 
                #ne.append(len(line[4].split()))

        else:
            skip = True

print(i, zero, one)

from scipy import stats
print(stats.describe(ne))
