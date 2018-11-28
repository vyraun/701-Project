
import csv
import sys
import numpy as np

import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))

def read_tokenized(filename):
    f = open(filename)
    for line in f:
        q1, q2, label = line.split("\t")
        print(q1, q2, label) 

def tokenize_quora(filename):
    i = 0

    values = []
    skip = False

    question_1 = []
    question_2 = []

    f = open("quora_tokenized.txt", "w")

    with open(filename) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"): 
            if skip:
                if line[3]!='' and line[4]!='' and line[5]!='':
                    q1, q2 = nltk.word_tokenize(line[3]), nltk.word_tokenize(line[4])
                    q1 = [w for w in q1 if not w in stop_words] # remove stop words
                    q1 = [word.lower() for word in q1 if word.isalpha()] # remove puntuation
                    q2 = [w for w in q2 if not w in stop_words] # remove stop words
                    q2 = [word.lower() for word in q2 if word.isalpha()] # remove puntuation
                    q1_tags, q2_tags = nltk.pos_tag(q1), nltk.pos_tag(q2)
                    q1_tags = [t[1] for t in q1_tags]
                    q2_tags = [t[1] for t in q2_tags]
                    f.write(' '.join([t for t in q1 + q1_tags]) + "\t" + ' '.join([t for t in q2 + q2_tags]) + "\t" + str(line[5]) + "\n")
                    i = i + 1
                    #if i == 100:
                    #    break
            else:
                skip = True

if __name__=="__main__":

    tokenize_quora(sys.argv[1])
    read_tokenized("quora_tokenized.txt")
