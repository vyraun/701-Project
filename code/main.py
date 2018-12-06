# coding=utf-8

"""
Implementation of Bilateral Multi Perspective Matching in PyTorch

Usage:
    main.py vocab
    main.py train [options]
    main.py test  [options]
    main.py train --train-src=<file> --dev-src=<file> --vocab-src=<file> [options]
    main.py test --test-src=<file> --vocab-src=<file> MODEL_PATH [options] 

Options:
    -h --help                               show this screen.
    --cuda=<bool>                           use GPU [default: 0]
    --train-src=<file>                      train source file [default: ../data/quora/small_train_tokenized.tsv]
    --dev-src=<file>                        dev source file [default: ../data/quora/small_dev_tokenized.tsv]
    --test-src=<file>                       test source file [default: ../data/quora/small_test_tokenized.tsv]
    --vocab-src=<file>                      vocab source file [default: ../data/quora/vocab.pkl]
    --aux-data-train=<file>                 auxilliary file for retrieval based testing [default: ../data/quora/small_train_tokenized.tsv]
    --aux-data-dev=<file>                   auxilliary file for retrieval based testing [default: ../data/quora/small_dev_tokenized.tsv]
    --aux-data-test=<file>                  auxilliary file for retrieval based testing [default: ../data/quora/small_test_tokenized.tsv]
    --model-path=<file>                     model path [default: ../data/models/model.bin]
    --optim-path=<file>                     optimiser state path [default: ../data/models/optim.bin]
    --glove-path=<file>                     pretrained glove embedding file [default: ../data/glove/glove.840B.300d.txt]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --rnn-type=<str>                        type of rnn (lstm, gru, rnn) [default: gru]
    --embed-size=<int>                      embedding size [default: 300]
    --char-embed-size=<int>                 char embedding size [default: 20]
    --bi-hidden-size=<int>                  bidirectional lstm hidden size [default: 100]
    --char-hidden-size=<int>                character lstm hidden size [default: 50]
    --char-rnn-layers=<int>                 number of layers in character lstm [default: 1]
    --rnn-layers=<int>                      number of layers in bidi rnn [default: 1]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 2]
    --max-epoch=<int>                       max epoch [default: 50]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 5]
    --dropout=<float>                       dropout [default: 0.2]
    --data=<str>                            type of dataset [default: quora]
    --perspective=<int>                     number of perspectives for the model [default: 20]
    --char=<bool>                           whether to use character embeddings or not, default is true [default: True]
    --len-clip=<int>                        the length at which to clip the sentences [default: 50]
    --model-type=<int>                      takes three options 1, 2 and 3 [default: 3]
"""

from docopt import docopt
from pdb import set_trace as bp
from models.model import Model
import utils.loader as loader
import torch
import torch.nn as nn
import time
import sys

def train(args):
    train_path = args['--train-src']
    dev_path = args['--dev-src']
    train_aux_path = args['--aux-data-train']
    dev_aux_path = args['--aux-data-dev']
    vocab_path = args['--vocab-src']
    lr = float(args['--lr'])
    log_every = int(args['--log-every'])
    model_path = args['--model-path']
    optim_path = args['--optim-path']
    max_patience = int(args['--patience'])
    max_num_trials = int(args['--max-num-trial'])
    clip_grad = float(args['--clip-grad'])
    valid_iter = int(args['--valid-niter'])
    model_type = int(args['--model-type'])
    if args['--data'] == 'quora':
        if model_type == 1 or model_type == 2:
            train_data = loader.read_data(train_path, 'quora')
            dev_data = loader.read_data(dev_path, 'quora')
        elif model_type == 3:
            train_data1 = loader.read_data(train_path, 'quora')
            train_data2 = loader.read_data(train_aux_path, 'quora')
            dev_data1 = loader.read_data(dev_path, 'quora')
            dev_data2 = loader.read_data(dev_aux_path, 'quora')
            train_data = (train_data1, train_data2)
            dev_data = (dev_data1, dev_data2)
        network = Model(args, 2)

    if args['--cuda'] == str(1):
        network.model = network.model.cuda()

    epoch = 0
    train_iter = 0
    report_loss = 0
    cum_loss = 0
    rep_examples = 0
    cum_examples = 0
    batch_size = int(args['--batch-size'])
    optimiser = torch.optim.Adam(list(network.model.parameters()), lr=lr)
    begin_time = time.time()
    prev_acc = 0
    val_hist = []
    num_trial = 0
    softmax = torch.nn.Softmax(dim=1)

    if args['--cuda'] == str(1):
        softmax = softmax.cuda()

    while True:
        epoch += 1
        
        for labels, p1, p2 in loader.batch_iter(model_type, train_data, batch_size):
            optimiser.zero_grad()
            train_iter += 1
            _, iter_loss = network.forward(labels, p1, p2)
            report_loss += iter_loss.item()
            cum_loss += iter_loss.item()

            iter_loss.backward()
            nn.utils.clip_grad_norm_(list(network.model.parameters()), clip_grad)
            optimiser.step()
 
            rep_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss, %.4f, cum. examples %d, time elapsed %.2f' %\
                     (epoch, train_iter, report_loss, cum_examples, time.time() - begin_time), file=sys.stderr)

                report_loss, rep_examples = 0, 0

            if train_iter % valid_iter == 0:
                print('epoch %d, iter %d, avg. loss, %.4f, cum. examples %d, time elapsed %.2f' %\
                     (epoch, train_iter, cum_loss / train_iter, cum_examples, time.time() - begin_time), file=sys.stderr)

                cum_loss, cum_examples = 0, 0
                print('Begin Validation .. ', file=sys.stderr)
                network.model.eval()
                total_examples = 0
                total_correct = 0
                val_loss, val_examples = 0, 0
                for val_labels, valp1, valp2 in loader.batch_iter(model_type, dev_data, batch_size):
                    total_examples += len(val_labels)
                    pred, _ = network.forward(val_labels, valp1, valp2)
                    pred = softmax(pred)
                    _, pred = pred.max(dim=1)
                    label_cor = network.get_label(val_labels)
                    total_correct += (pred == label_cor).sum().float()
                final_acc = total_correct / total_examples
 
                val_hist.append(final_acc) 
                val_acc = final_acc
                print('Validation: iter %d, val_acc %.4f' % (train_iter, val_acc), file=sys.stderr)
                if val_acc > prev_acc:
                    patience = 0
                    prev_acc = val_acc
                    print('Saving model and optimiser state', file=sys.stderr)
                    torch.save(network.model, model_path)
                    torch.save(optimiser.state_dict(), optim_path)
                else:
                    patience += 1
                    print('hit patience %d' %(patience), file=sys.stderr)
                    if patience == max_patience:
                        num_trial += 1
                        print('hit #%d' %(num_trial), file=sys.stderr)
                        if num_trial == max_num_trials:
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' %(lr), file=sys.stderr)

                        network.model = torch.load(model_path)
                        if args['--cuda'] == str(1):
                            network.model = network.model.cuda()

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimiser = torch.optim.Adam(list(network.model.parameters()), lr=lr)
                        optimiser.load_state_dict(torch.load(optim_path))
                        for state in optimiser.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v
                        for group in optimiser.param_groups:
                            group['lr'] = lr

                        patience = 0
                network.model.train()

        if epoch == int(args['--max-epoch']):
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0) 

def test(args):
    test_path = args['--test-src']
    test_aux_path = args['--aux-data-test']
    model_path = args['--model-path']
    batch_size = int(args['--batch-size'])
    total_examples = 0
    total_correct = 0
    vocab_path = args['--vocab-src']
    model_type = int(args['--model-type'])
    softmax = torch.nn.Softmax(dim=1)

    if args['--data'] == 'quora':
        if model_type == 1 or model_type == 2:
            test_data = loader.read_data(test_path, 'quora')
        else:
            test_data1 = loader.read_data(test_path, 'quora')
            test_data2 = loader.read_data(test_aux_path, 'quora')
            test_data = (test_data1, test_data2)
        network = Model(args, 2)
        network.model = torch.load(model_path)

    if args['--cuda'] == str(1):
        network.model = network.model.cuda()
        softmax = softmax.cuda()

    network.model.eval()
    for labels, p1, p2 in loader.batch_iter(model_type, test_data, batch_size):
        total_examples += len(labels)
        print(total_examples)
        pred, _ = network.forward(labels, p1, p2)
        pred = softmax(pred)
        _, pred = pred.max(dim=1)
        label = network.get_label(labels)
        total_correct += (pred == label).sum().float()
    final_acc = total_correct / total_examples
    print('Accuracy of the model is %.2f' % (final_acc), file=sys.stderr)
   
def main(args):
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    elif args['vocab']:
        print("Need to use another file for this")

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
