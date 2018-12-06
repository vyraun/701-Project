import codecs
import math
import numpy as np
from tqdm import tqdm
from pdb import set_trace as bp

def read_data(file_path, data_type):
    if data_type == "quora":
        data = codecs.open(file_path, 'r', 'utf-8').readlines()
        corpus = []
        print("Reading the data")
        for line in tqdm(data):
            example = []
            line = line.strip().split('\t')
            p1_tokens = line[0].split()
            p2_tokens = line[2].split()
            example.append(int(line[-1]))
            example.append(p1_tokens)
            example.append(p2_tokens)
            corpus.append(example)
        return corpus
    else:
        return "Not Implemented the rest"

def batch_iter(model_type, data, batch_size, shuffle=True):

    if model_type == 1 or model_type == 2:
        batch_num = math.ceil((len(data) / batch_size))
        index_array = list(range(len(data)))
        if shuffle:
            np.random.shuffle(index_array)
        for i in range(batch_num):
            indices = index_array[i * (batch_size): (i + 1) * batch_size]
            examples = [data[idx] for idx in indices]
            examples = sorted(examples, key=lambda e: len(e[1] + e[2]), reverse=True)
            labels = [e[0] for e in examples]
            p1 = [e[1] for e in examples]
            p2 = [e[2] for e in examples]
            #idx = [e[3] for e in examples]
            yield labels, p1, p2
    elif model_type == 3:
        batch_num = math.ceil((len(data[0]) / batch_size))
        index_array = list(range(len(data[0])))
        if shuffle:
            np.shuffle.random(index_array)
        for i in range(batch_num):
            indices = index_array[i * (batch_size): (i + 1) * batch_size]
            examples = [data[0][idx] for idx in indices]
            labels = [e[0] for e in examples]
            p1 = [e[1] for e in examples]
            p2 = [e[2] for e in examples]
            examples2 =  [data[1][idx] for idx in indices]
            p1_aux = [e[1] for e in examples2]
            p2_aux = [e[2] for e in examples2]
            #idx = [e[3] for e in examples]
            yield labels, (p1, p1_aux), (p2, p2_aux)

