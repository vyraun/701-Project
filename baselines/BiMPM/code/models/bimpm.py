import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class BiMPM(nn.Module):

    def __init__(self, args, vocab, class_size, wembeddings, word_vocab):
        super(BiMPM, self).__init__()

        self.c_embed_size = int(args['--char-embed-size'])
        self.w_embed_size = int(args['--embed-size'])
        self.l = int(args['--perspective'])
        self.dropout_val = float(args['--dropout'])
        self.bi_hidden = int(args['--bi-hidden-size'])
        self.char_hidden = int(args['--char-hidden-size'])
        self.rnn_type = args['--rnn-type']
        self.char_layer_size = int(args['--char-lstm-layers'])
        self.context_layer_size = int(args['--bilstm-layers'])
        self.char_inp = vocab + 100
        self.classes = class_size
        self.char_use = args['--char']

        self.wembeddings = nn.Embedding(num_embeddings=word_vocab,\
                                        embedding_dim=self.w_embed_size)

        self.wembeddings.weight.data.copy_(torch.from_numpy(wembeddings))
        self.dropout = nn.Dropout(self.dropout_val)

        if self.char_use:
            self.char_embedding = nn.Embedding(num_embeddings=self.char_inp,\
                                               embedding_dim=self.c_embed_size,\
                                               padding_idx=0)

            self.char_lstm = nn.LSTM(input_size=self.c_embed_size,\
                                     hidden_size=self.char_hidden,\
                                     num_layers=self.char_layer_size,\
                                     dropout=self.dropout_val)

            self.context_lstm = nn.LSTM(input_size=self.w_embed_size + self.char_hidden,\
                                        hidden_size=self.bi_hidden,\
                                        num_layers=self.context_layer_size,\
                                        bidirectional=True,\
                                        dropout=self.dropout_val)

        else:
            self.context_lstm = nn.LSTM(input_size=self.w_embed_size,\
                                        hidden_size=self.bi_hidden,\
                                        num_layers=self.context_layer_size,\
                                        bidirectional=True,\
                                        dropout=self.dropout_val)



        self.aggregation_lstm = nn.LSTM(input_size = self.l * 8,
                                        hidden_size = self.bi_hidden,\
                                        bidirectional=True,\
                                        dropout=self.dropout_val)


        for i in range(1, 9):
            setattr(self, f'w{i}', nn.Parameter(torch.rand(1, 1, self.bi_hidden, self.l)))

        self.ff1 = nn.Linear(self.bi_hidden * 4, self.bi_hidden * 2)
        self.ff2 = nn.Linear(self.bi_hidden * 2, self.classes)

        self.init_weights()

    def init_weights(self):
        for param in list(self.parameters()):
            nn.init.uniform_(param, -0.01, 0.01)
 
    def init_char_embed(self, c1, c2):
        c1_embed = self.char_embedding(c1)
        char_p1 = self.char_lstm(c1_embed)
        c2_embed = self.char_embedding(c2)
        char_p2 = self.char_lstm(c2_embed)
        return char_p1[0][-1], char_p2[0][-1]


    def cosine_similarity(self, prod, norm):
        # As set in PyTorch documentation
        eps = 1e-8
        norm = norm * (norm > eps).float() + eps * (norm <= eps).float()

        return prod / norm

    def full_matching(self, p1, p2, w_matrix):
        p1 = torch.stack([p1] * self.l, dim = 3)
        p1 = w_matrix * p1

        p1_seq_len = p1.size(0)
        p2 = torch.stack([p2] * p1_seq_len, dim = 0)
        p2 = torch.stack([p2] * self.l, dim = 3)
        p2 = w_matrix * p2
        result = F.cosine_similarity(p1, p2, dim=2)
        return result

    def maxpool_matching(self, p1, p2, w_matrix):
        p1 = torch.stack([p1] * self.l, dim = 3)
        p1 = w_matrix * p1
        
        p2 = torch.stack([p2] * self.l, dim = 3)
        p2 = w_matrix * p2

        p1_norm = p1.norm(p = 2, dim = 2, keepdim=True)
        p2_norm = p2.norm(p = 2, dim = 2, keepdim=True)

        full_mat = torch.matmul(p1.permute(1, 3, 0, 2), p2.permute(1, 3, 2, 0))
        deno_mat = torch.matmul(p1_norm.permute(1, 3, 0, 2), p2_norm.permute(1, 3, 2, 0))

        result, _ = self.cosine_similarity(full_mat, deno_mat).max(dim = 3)
        result = result.permute(2, 0, 1)
        return result

    def attentive_matching(self, p1, p2, w_matrix_att, w_matrix_max):
        #Perform both attentive types of matching together
        p1_norm = p1.norm(p = 2, dim = 2, keepdim=True)
        p2_norm = p2.norm(p = 2, dim = 2, keepdim=True)

        full_mat = torch.matmul(p1.permute(1,0,2), p2.permute(1, 2, 0))
        deno_mat = torch.matmul(p1_norm.permute(1, 0, 2), p2_norm.permute(1, 2, 0))
        alpha_mat = self.cosine_similarity(full_mat, deno_mat)

        _, max_index = torch.max(alpha_mat, dim=2)
        max_index = torch.stack([max_index] * self.bi_hidden, dim=2)
        
        h_mat = torch.bmm(alpha_mat, p2.transpose(1, 0))
        alpha_mat = alpha_mat.sum(dim=2, keepdim=True)
        resultant = h_mat / alpha_mat

        v1 = resultant.transpose(1, 0).unsqueeze(-1) * w_matrix_att
        v2 = p1.unsqueeze(-1) * w_matrix_att
        result_match = F.cosine_similarity(v1, v2, dim=2)

        out_mat = torch.gather(p2.transpose(1, 0), 1, max_index)
        v1 = out_mat.transpose(1, 0).unsqueeze(-1) * w_matrix_max
        v2 = p1.unsqueeze(-1) * w_matrix_max
        result_max = F.cosine_similarity(v1, v2, dim=2)
        
        return result_match, result_max

    def forward(self, p1, p2, c1, c2, p1_len, p2_len):

        p1_input = self.wembeddings(p1)
        p2_input = self.wembeddings(p2)

        if self.char_use:
            char_p1, char_p2 = self.init_char_embed(c1, c2)
            dim1, dim2 = p1.size()
            char_p1 = char_p1.view(dim1, dim2, -1)
            dim1, dim2 = p2.size()
            char_p2 = char_p2.view(dim1, dim2, -1)
            p1_input = torch.cat((p1_input, char_p1), 2)
            p2_input = torch.cat((p2_input, char_p2), 2)

            context1_full, (context1_lh, _) = self.context_lstm(p1_input)
            context2_full, (context2_lh, _) = self.context_lstm(p2_input)

        else:
            context1_full, (context1_lh, _) = self.context_lstm(p1_input)
            context2_full, (context2_lh, _) = self.context_lstm(p2_input)

        context1_forw, context1_back = torch.split(context1_full, self.bi_hidden, 2)
        context1_lh_forw, context1_lh_back = context1_lh[0], context1_lh[1]

        context2_forw, context2_back = torch.split(context2_full, self.bi_hidden, 2)
        context2_lh_forw, context2_lh_back = context2_lh[0], context2_lh[1]

        # 4 tensors from forward and backward matching (full matching)
        match_p1_forw = self.full_matching(context1_forw, context2_lh_forw, self.w1)
        match_p1_back = self.full_matching(context1_back, context2_lh_back, self.w2)
        match_p2_forw = self.full_matching(context2_forw, context1_lh_forw, self.w1)
        match_p2_back = self.full_matching(context2_back, context1_lh_back, self.w2)

        # 4 tensors from forward and backward matching (max-pooling matching)
        maxm_p1_forw = self.maxpool_matching(context1_forw, context2_forw, self.w3)
        maxm_p1_back = self.maxpool_matching(context1_back, context2_back, self.w4)
        maxm_p2_forw = self.maxpool_matching(context2_forw, context1_forw, self.w3)
        maxm_p2_back = self.maxpool_matching(context2_back, context1_back, self.w4)

        # 8 tensors from the forward and backward attentive matching and attentive max
        att_p1_forw, attm_p1_forw = self.attentive_matching(context1_forw, context2_forw, self.w5, self.w7)
        att_p1_back, attm_p1_back = self.attentive_matching(context1_back, context2_back, self.w6, self.w8)
        att_p2_forw, attm_p2_forw = self.attentive_matching(context2_forw, context1_forw, self.w5, self.w7)
        att_p2_back, attm_p2_back = self.attentive_matching(context2_back, context1_back, self.w6, self.w8)

        aggr_p1 = torch.cat([match_p1_forw, match_p1_back, maxm_p1_forw, maxm_p1_back,\
                             att_p1_forw, att_p1_back, attm_p1_forw, attm_p1_back], dim=2)

        aggr_p2 = torch.cat([match_p2_forw, match_p2_back, maxm_p2_forw, maxm_p2_back,\
                             att_p2_forw, att_p2_back, attm_p2_forw, attm_p2_back], dim=2)

        aggr_p1 = self.dropout(aggr_p1)
        aggr_p2 = self.dropout(aggr_p2)

        _, (p1_output, _) = self.aggregation_lstm(aggr_p1)
        _, (p2_output, _) = self.aggregation_lstm(aggr_p2)

        output = torch.cat([torch.cat([p1_output[0,:,:], p1_output[1,:,:]], dim=-1), \
                           torch.cat([p2_output[0,:,:], p2_output[1,:,:]], dim=-1)], dim=-1)

        output = self.dropout(output)
        output = torch.tanh(self.ff1(output))
        output = self.dropout(output)
        output = self.ff2(output)

        return output
