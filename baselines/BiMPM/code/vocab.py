#coding=utf-8
"""
Building Vocabulary for BiMPM to train on

Usage:
    vocab.py --train-src=<file> --type=<str> --data-path=<file>
"""

from docopt import docopt
from pdb import set_trace as bp
from spacy.lang.en import English
import spacy
from docopt import docopt
from tqdm import tqdm
import pickle
import csv
import os

class Vocab(object):
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.char2index = {}
        self.index2char = {}

    def build_vocab(self, sents):
        self.counter = 0
        # 0 set for padding of characters
        self.char_counter = 2
        for sent in tqdm(sents):
            for word in sent:
                lowered_word = word.text.lower()
                if lowered_word not in self.word2index:
                    self.word2index[lowered_word] = self.counter
                    self.index2word[self.counter] = lowered_word
                    self.counter += 1
                for chars in word.text:
                    char = chars.lower()
                    if char not in self.char2index:
                        self.char2index[char] = self.char_counter
                        self.index2char[self.char_counter] = char
                        self.char_counter += 1

    def word2id(self, word):
        if word.lower() in self.word2index:
            return self.word2index[word.lower()]
        else:
            return None

    def id2word(self, idx):
        if idx in self.index2word:
            return self.index2word[idx]
        else:
            return None

    def char2id(self, char):
        if char.lower() in self.char2index:
            return self.char2index[char.lower()]
        else:
            return 1

    def id2char(self, idx):
        if idx in self.index2char:
            return self.index2char[idx]
        else:
            return None

def build_quora_vocab(args):
    train_path = args['--train-src']
    sents = []
    nlp = spacy.load('en')
    reader = csv.reader(open(train_path), delimiter='\t')
    print("Tokenizing Sentences")
    for line in tqdm(reader):
        p1 = [each for each in nlp.tokenizer(line[1])]
        p2 = [each for each in nlp.tokenizer(line[2])]
        sents.append(p1)
        sents.append(p2)

    print("Creating Vocabularies both Word and Char")
    vocab_model = Vocab()
    vocab_model.build_vocab(sents)
    file_path = os.path.join(args['--data-path'], 'vocab.pkl')
    print("Word and Char Vocabs Created .. Dumping Vocab Model")
    pickle.dump(vocab_model, open(file_path, 'wb'))

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['--type'] == 'quora':
        build_quora_vocab(args)
