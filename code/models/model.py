import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from pdb import set_trace as bp
#from models.bimpm import BiMPM


class Model(object):
    def __init__(self, args, class_size):
        self.data = args['--data']
        self.wembed_size = int(args['--embed-size'])
        self.cuda = args['--cuda']

        self.glove_path = args['--glove-path']

        print("Reading Glove File")
        self.words = []
        self.word2idx = {}
        idx = 1
        vectors = []
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        vectors.append(np.zeros((300, )))
        vectors.append(np.random.rand(300))
        with open(self.glove_path) as f:
            for l in f:
                word, vec = l.split(' ', 1)
                idx += 1
                self.word2idx[word] = idx
                vect = np.asarray(np.fromstring(vec, sep=' ')).astype(np.float)
                vectors.append(vect)
                if idx == 100:
                    break
        print("Read the Glove file")
        embeddings = np.asarray(vectors)
        #self.model = BiMPM(args, self.vocab_len, class_size, embeddings, idx + 1)
        #self.criterion = torch.nn.CrossEntropyLoss()

        #if self.cuda == str(1):
        #    self.criterion = self.criterion.cuda()

    def forward(self, label, p1, p2, method='train'):
        word_tensor1, word_tensor2, label_tensor,\
                p1_orig_len, p2_orig_len = self.format_data(label, p1, p2)
        bp()
        label_tensor = label_tensor
        if self.cuda == str(1):
            word_tensor1 = word_tensor1.cuda()
            word_tensor2 = word_tensor2.cuda()
            label_tensor = label_tensor.cuda()

        predicted = 0
        loss = 0
        #predicted = self.model(word_tensor1, word_tensor2, char_tensor1, char_tensor2,\
                          #p1_orig_len, p2_orig_len)
        #loss = self.criterion(predicted, label_tensor)
        return predicted, loss

    def evaluate(self, label, p1, p2):
        return None

    def format_data(self, label, p1, p2):
        word_p1_inp = []
        maxp1 = 0
        p1_orig_length = []
        # Construct GloVe for each word, and find max word length (for one sentence)
        for each in p1:
            p1_orig_length.append(len(each))
            aux = []
            for word in each:
                if word in self.word2idx:
                    aux.append(self.word2idx[word])
                else:
                    aux.append(1)
                maxp1 = max(maxp1, len(word))
            word_p1_inp.append(torch.LongTensor(aux))

        word_p2_inp = []
        maxp2 = 0
        p2_orig_length = []
        # Construct GloVe for each word, and find max word length (for other sentence)
        for each in p2:
            p2_orig_length.append(len(each))
            aux = []
            for word in each:
                maxp2 = max(maxp2, len(word))
                if word in self.word2idx:
                    aux.append(self.word2idx[word])
                else:
                    aux.append(1)
            word_p2_inp.append(torch.LongTensor(aux))

        word_p1_inp = pad_sequence(word_p1_inp)
        word_p2_inp = pad_sequence(word_p2_inp)

        # Initiliase label tensor
        label = torch.LongTensor([int(each) for each in label])

        return (word_p1_inp, word_p2_inp, label,\
                p1_orig_length, p2_orig_length)

    def set_labels(self, value):
        self.classes = value

    def get_label(self, label):
        label = [int(each) for each in label]
        if self.cuda == str(1):
            return torch.LongTensor(label).cuda()
        else:
            return torch.LongTensor(label)
