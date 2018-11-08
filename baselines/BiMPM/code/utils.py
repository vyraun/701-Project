import math
import csv
import spacy
import pickle
from spacy.lang.en import English
from tqdm import tqdm
from pdb import set_trace as bp
from vocab import Vocab 

def read_data(file_path, data_type):
    if data_type == "quora":
        fp = open(file_path)
        reader = csv.reader(fp, delimiter='\t')
        nlp = spacy.load('en')
        corpus = []
        for line in tqdm(reader):
            data = []
            p1_tokens = [each.text.lower() for each in nlp.tokenizer(line[1])]
            p2_tokens = [each.text.lower() for each in nlp.tokenizer(line[2])]
            data.append(line[0])
            data.append(p1_tokens)
            data.append(p2_tokens)
            data.append(line[3])
            corpus.append(data)

        return corpus
    else:
        return "Not Implemented the rest"

def load_vocab(file_path):
    return pickle.load(open(file_path, 'rb'))

def batch_iter(data, batch_size, shuffle=False):

    batch_num = math.ceil((len(data) / batch_size))
    index_array = list(range(len(data)))
    if shuffle:
        np.shuffle.random(index_array)
    for i in range(batch_num):
        indices = index_array[i * (batch_size): (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[1] + e[2]), reverse=True)
        labels = [e[0] for e in examples]
        p1 = [e[1] for e in examples]
        p2 = [e[2] for e in examples]
        idx = [e[3] for e in examples]
        yield labels, p1, p2, idx
