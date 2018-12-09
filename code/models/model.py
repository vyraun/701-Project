import numpy as np
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from pdb import set_trace as bp
from models.mixmatch import Matcher


class Model(object):
    def __init__(self, args, class_size):
        self.data = args['--data']
        self.wembed_size = int(args['--embed-size'])
        self.cuda = args['--cuda']
        self.clip_length = int(args['--len-clip'])
        self.glove_path = args['--glove-path']
        self.model_type = int(args['--model-type'])
        self.vocab_data = pickle.load(open(args['--vocab-data-pkl'], 'rb'))
        print("Reading Glove File")
        self.words = []
        self.word2idx = {}
        self.id2word = {}
        idx = 1
        vectors = []
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        vectors.append(np.zeros((300, )))
        vectors.append(np.random.rand(300))
        with open(self.glove_path) as f:
            for l in f:
                word, vec = l.split(' ', 1)
                if word in self.vocab_data:
                    idx += 1
                    self.word2idx[word] = idx
                    self.id2word[idx] = word
                    vect = np.fromstring(vec, sep=' ')
                    vectors.append(vect)
                    #if idx == 100:
                    #    break
        print("Read the Glove file")
        embeddings = np.zeros((idx, self.wembed_size))
        for i in range(idx):
            embeddings[i] = vectors[i]

        embeddings = np.asarray(embeddings)
        self.model = Matcher(args, class_size, embeddings, idx)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.BCELoss()
        if self.cuda == str(1):
            self.criterion = self.criterion.cuda()
            self.creterion2 = self.criterion2.cuda()

    def get_masked_matrix(self, input_lengths):
        matrix = []
        for i in range(len(input_lengths)):
            arr = [1 for j in range(input_lengths[i])]
            for j in range(len(arr), self.clip_length):
                arr.append(0)
            matrix.append(arr)
        return torch.FloatTensor(matrix)

    def forward(self, label, p1, p2, method='train'):
        if self.model_type == 1 or self.model_type == 2:
            word_tensor1, word_tensor2, label_tensor,\
                    p1_orig_len, p2_orig_len = self.format_data(label, p1, p2)
            label_tensor = label_tensor
            p1_orig_len = self.get_masked_matrix(p1_orig_len)
            p2_orig_len = self.get_masked_matrix(p2_orig_len)
            if self.cuda == str(1):
                word_tensor1 = word_tensor1.cuda()
                word_tensor2 = word_tensor2.cuda()
                p1_orig_len = p1_orig_len.cuda()
                p2_orig_len = p2_orig_len.cuda()
                label_tensor = label_tensor.cuda()

            predicted = 0
            loss = 0
            predicted = self.model(word_tensor1, word_tensor2, p1_orig_len, p2_orig_len)
            loss = self.criterion(predicted, label_tensor)
            return predicted, loss
        elif self.model_type == 3:
            word_tensor1, word_tensor2, word_aux1, word_aux2, label_tensor,\
                    p1_orig_len, p2_orig_len = self.format_data(label, p1, p2)
            p1_orig_len, p1_orig_len_aux = p1_orig_len
            p2_orig_len, p2_orig_len_aux = p2_orig_len
            p1_orig_len = self.get_masked_matrix(p1_orig_len)
            p2_orig_len = self.get_masked_matrix(p2_orig_len)
            p1_orig_len_aux = self.get_masked_matrix(p1_orig_len_aux)
            p2_orig_len_aux = self.get_masked_matrix(p2_orig_len_aux)
            label_tensor = label_tensor
            if self.cuda == str(1):
                word_tensor1 = word_tensor1.cuda()
                word_tensor2 = word_tensor2.cuda()
                word_aux1 = word_aux1.cuda()
                word_aux2 = word_aux2.cuda()
                p1_orig_len = p1_orig_len.cuda()
                p2_orig_len = p2_orig_len.cuda()
                p1_orig_len_aux = p1_orig_len_aux.cuda()
                p2_orig_len_aux = p2_orig_len_aux.cuda()
                label_tensor = label_tensor.cuda()

            p1_orig_len = (p1_orig_len, p1_orig_len_aux)
            p2_orig_len = (p2_orig_len, p2_orig_len_aux)
            predicted = 0
            loss = 0
            predicted = self.model((word_tensor1, word_aux1), (word_tensor2, word_aux2), \
                                   p1_orig_len, p2_orig_len)
            loss = self.criterion(predicted, label_tensor)
            return predicted, loss
  
    def evaluate(self, label, p1, p2):
        return None

    def format_data(self, label, p1, p2):

        #labeller = []
        #for each in label:
        #    val = int(each)
        #    if val == 0:
        #        labeller.append([1, 0])
        #    else:
        #        labeller.append([0, 1])
        label = torch.LongTensor(label)

        if self.model_type == 1 or self.model_type == 2:
            word_p1_inp = []
            maxp1 = 0
            p1_orig_length = []
            # Construct GloVe for each word, and find max word length (for one sentence)
            for each in p1:
                p1_orig_length.append(min(len(each), self.clip_length))
                aux = []
                for i in range(min(len(each), self.clip_length)):
                    if each[i] in self.word2idx:
                        aux.append(self.word2idx[each[i]])
                    else:
                        aux.append(1)
                    maxp1 = max(maxp1, len(each[i]))
                if i < self.clip_length:
                    while i < self.clip_length - 1:
                        aux.append(0)
                        i += 1
                word_p1_inp.append(torch.LongTensor(aux))

            word_p2_inp = []
            maxp2 = 0
            p2_orig_length = []
            # Construct GloVe for each word, and find max word length (for other sentence)
            for each in p2:
                p2_orig_length.append(min(len(each), self.clip_length))
                aux = []
                for i in range(min(len(each), self.clip_length)):
                    if each[i] in self.word2idx:
                        aux.append(self.word2idx[each[i]])
                    else:
                        aux.append(1)
                    maxp1 = max(maxp1, len(each[i]))
                if i < self.clip_length:
                    while i < self.clip_length - 1:
                        aux.append(0)
                        i += 1
                word_p2_inp.append(torch.LongTensor(aux))
            word_p1_inp = pad_sequence(word_p1_inp, batch_first=True)
            word_p2_inp = pad_sequence(word_p2_inp, batch_first=True)

            # Initiliase label tensor
            #label = torch.LongTensor([int(each) for each in label])

            return (word_p1_inp, word_p2_inp, label,\
                    p1_orig_length, p2_orig_length)

        if self.model_type == 3:
            word_p1_inp = []
            maxp1 = 0
            p1_orig_length = []
            # Construct GloVe for each word, and find max word length (for one sentence)
            for each in p1[0]:
                p1_orig_length.append(min(len(each), self.clip_length))
                aux = []
                for i in range(min(len(each), self.clip_length)):
                    if each[i] in self.word2idx:
                        aux.append(self.word2idx[each[i]])
                    else:
                        aux.append(1)
                    maxp1 = max(maxp1, len(each[i]))
                if i < self.clip_length:
                    while i < self.clip_length - 1:
                        aux.append(0)
                        i += 1
                word_p1_inp.append(torch.LongTensor(aux))

            word_p2_inp = []
            maxp2 = 0
            p2_orig_length = []
            # Construct GloVe for each word, and find max word length (for other sentence)
            for each in p2[0]:
                p2_orig_length.append(min(len(each), self.clip_length))
                aux = []
                for i in range(min(len(each), self.clip_length)):
                    if each[i] in self.word2idx:
                        aux.append(self.word2idx[each[i]])
                    else:
                        aux.append(1)
                    maxp1 = max(maxp1, len(each[i]))
                if i < self.clip_length:
                    while i < self.clip_length - 1:
                        aux.append(0)
                        i += 1
                word_p2_inp.append(torch.LongTensor(aux))
            word_p1_inp = pad_sequence(word_p1_inp, batch_first=True)
            word_p2_inp = pad_sequence(word_p2_inp, batch_first=True)

            word_p1_inp_aux = []
            maxp1 = 0
            p1_orig_length_aux = []
            # Construct GloVe for each word, and find max word length (for one sentence)
            for each in p1[1]:
                p1_orig_length_aux.append(min(len(each), self.clip_length))
                aux = []
                for i in range(min(len(each), self.clip_length)):
                    if each[i] in self.word2idx:
                        aux.append(self.word2idx[each[i]])
                    else:
                        aux.append(1)
                    maxp1 = max(maxp1, len(each[i]))
                if i < self.clip_length:
                    while i < self.clip_length - 1:
                        aux.append(0)
                        i += 1
                word_p1_inp_aux.append(torch.LongTensor(aux))

            word_p2_inp_aux = []
            maxp2 = 0
            p2_orig_length_aux = []
            # Construct GloVe for each word, and find max word length (for other sentence)
            for each in p2[1]:
                p2_orig_length_aux.append(min(len(each), self.clip_length))
                aux = []
                for i in range(min(len(each), self.clip_length)):
                    if each[i] in self.word2idx:
                        aux.append(self.word2idx[each[i]])
                    else:
                        aux.append(1)
                    maxp1 = max(maxp1, len(each[i]))
                if i < self.clip_length:
                    while i < self.clip_length - 1:
                        aux.append(0)
                        i += 1
                word_p2_inp_aux.append(torch.LongTensor(aux))
            word_p1_inp_aux = pad_sequence(word_p1_inp, batch_first=True)
            word_p2_inp_aux = pad_sequence(word_p2_inp, batch_first=True)


            # Initiliase label tensor
            #labeller = []
            #for each in label:
            #    val = int(each)
            #    if val == 0:
            #        labeller.append([1, 0])
            #    else:
            #        labeller.append([0, 1])
            #label = torch.LongTensor(labeller)
            #label = torch.LongTensor([int(each) for each in label])

            return (word_p1_inp, word_p2_inp, word_p1_inp_aux, word_p2_inp_aux, label,\
                    (p1_orig_length, p1_orig_length_aux), (p2_orig_length, p2_orig_length_aux))


    def set_labels(self, value):
        self.classes = value

    def get_label(self, label):
        label = [int(each) for each in label]
        if self.cuda == str(1):
            return torch.LongTensor(label).cuda()
        else:
            return torch.LongTensor(label)
