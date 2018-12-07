import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class Matcher(nn.Module):

    def __init__(self, args, class_size, wembeddings, word_vocab):
        super(Matcher, self).__init__()

        self.w_embed_size = int(args['--embed-size'])
        self.dropout_val = float(args['--dropout'])
        self.bi_hidden = int(args['--bi-hidden-size'])
        self.char_hidden = int(args['--char-hidden-size'])
        self.rnn_type = args['--rnn-type']
        self.context_layer_size = int(args['--rnn-layers'])
        self.classes = class_size
        self.l = int(args['--perspective'])

        self.model_type = int(args['--model-type'])

        self.mu_list = [1.0]
        for i in range(1, 10, 2):
            self.mu_list.append(i * 0.1)
            self.mu_list.append(- i * 0.1)
        self.sigma = 0.1

        #initliase word embeddings
        self.wembeddings = nn.Embedding(num_embeddings=word_vocab,\
                                        embedding_dim=self.w_embed_size)

        #initliase with pre-trained embeddings
        #self.wembeddings.weight.data.copy_(torch.from_numpy(wembeddings))

        #context representation layer
        if self.rnn_type == 'gru':
            self.context = nn.GRU(input_size=self.w_embed_size, \
                                  hidden_size=self.bi_hidden, \
                                  num_layers=self.context_layer_size, \
                                  bidirectional=True)
        if self.rnn_type == 'lstm':
            self.context = nn.LSTM(input_size=self.w_embed_size, \
                                  hidden_size=self.bi_hidden, \
                                  num_layers=self.context_layer_size, \
                                  bidirectional=True)

        #vectors for attentive matching
        for i in range(1, 5):
            setattr(self, f'w{i}', nn.Parameter(torch.rand(1, 1, self.bi_hidden, self.l)))

        #aggregation layer
        if self.rnn_type == 'gru':
            self.aggregation = nn.GRU(input_size=self.l * 4, \
                                    hidden_size=self.bi_hidden, \
                                    num_layers=self.context_layer_size, \
                                    bidirectional=True)
        if self.rnn_type == 'lstm':
            self.aggregation = nn.LSTM(input_size=self.l * 4, \
                                    hidden_size=self.bi_hidden, \
                                    num_layers=self.context_layer_size, \
                                    bidirectional=True)


        #dropout layer
        self.dropout = nn.Dropout(self.dropout_val)

        self.attentive_linear = nn.Linear(self.bi_hidden * 4, len(self.mu_list))
        if self.model_type == 1:
            self.ltr = nn.Linear(len(self.mu_list), self.classes)
        elif self.model_type == 2:
            self.ltr = nn.Linear(len(self.mu_list) * 2, self.classes)
        elif self.model_type == 3:
            self.ltr = nn.Linear(len(self.mu_list) * 3, self.classes)
        self.init_weights()
        self.wembeddings.weight.data.copy_(torch.from_numpy(wembeddings))

    def init_weights(self):
        for param in list(self.parameters()):
            nn.init.uniform_(param, -0.01, 0.01)
 
    def cosine_similarity(self, prod, norm):
        # As set in PyTorch documentation
        eps = 1e-8
        norm = norm * (norm > eps).float() + eps * (norm <= eps).float()

        return prod / norm

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

    def forward(self, p1, p2, p1_len, p2_len):
        if self.model_type == 3:
            p1, p1_aux = p1
            p2, p2_aux = p2
            p1_len, p1_len_aux = p1_len
            p2_len, p2_len_aux = p2_len

        p1_input = self.wembeddings(p1)
        p2_input = self.wembeddings(p2)

        if self.model_type == 3 or self.model_type == 2:
            if self.rnn_type == 'gru':
                context1_full, context1_lh = self.context(p1_input.transpose(0, 1))
                context2_full, context2_lh = self.context(p2_input.transpose(0, 1))
            if self.rnn_type == 'lstm':
                context1_full, (context1_lh, _) = self.context(p1_input.transpose(0, 1))
                context2_full, (context2_lh, _) = self.context(p2_input.transpose(0, 1))

            context1_forw, context1_back = torch.split(context1_full, self.bi_hidden, 2)
            context1_lh_forw, context1_lh_back = context1_lh[0], context1_lh[1]
            context2_forw, context2_back = torch.split(context2_full, self.bi_hidden, 2)
            context2_lh_forw, context2_lh_back = context2_lh[0], context2_lh[1]

        p1_norm = p1_input.norm(p=2, dim=2, keepdim=True)
        p2_norm = p2_input.norm(p=2, dim=2, keepdim=True)

        p1_len = p1_len.unsqueeze(-1)
        p2_len = p2_len.unsqueeze(-1)
        masked_mat = torch.bmm(p1_len, p2_len.transpose(1, 2))

        if self.model_type == 3:
            p1_len_aux = p1_len_aux.unsqueeze(-1)
            p2_len_sux = p2_len_aux.unsqueeze(-1)
            masked_mat_aux = torch.bmm(p1_len, p2_len.transpose(1, 2))

        norm_mat = torch.bmm(p1_norm, p2_norm.transpose(1, 2))
        trans_matrix = torch.bmm(p1_input, p2_input.transpose(1, 2))

        # Translation Matrix Constructed here (using cosine similarity)
        trans_matrix = self.cosine_similarity(trans_matrix, norm_mat)
        trans_matrix = trans_matrix * masked_mat
        trans_features = 0
        for i, each in enumerate(self.mu_list):
            trans_matrix_res = torch.exp(-((trans_matrix - each) ** 2) / (2 * (self.sigma) ** 2))
            trans_matrix_res = torch.log1p(trans_matrix_res.sum(dim=2))
            trans_matrix_res = trans_matrix_res.sum(dim=1, keepdim=True)
            if i == 0:
                trans_features = trans_matrix_res
            else:
                trans_features = torch.cat([trans_features, trans_matrix_res], dim=1)

        trans_features = self.dropout(trans_features)

        if self.model_type == 1:
            output = self.ltr(trans_features)
            return output

        # For Attentive Matching
        att_p1_forw, attm_p1_forw = self.attentive_matching(context1_forw, context2_forw, self.w1, self.w2)
        att_p1_back, attm_p1_back = self.attentive_matching(context1_back, context2_back, self.w3, self.w4)
        att_p2_forw, attm_p2_forw = self.attentive_matching(context2_forw, context1_forw, self.w1, self.w2)
        att_p2_back, attm_p2_back = self.attentive_matching(context2_back, context1_back, self.w3, self.w4)

        aggr_p1 = torch.cat([attm_p1_forw, attm_p1_back, att_p1_forw, att_p1_back], dim=2)
        aggr_p2 = torch.cat([attm_p2_forw, attm_p2_back, att_p2_forw, att_p2_back], dim=2)

        if self.rnn_type == 'gru':
            _, p1_output = self.aggregation(aggr_p1)
            _, p2_output = self.aggregation(aggr_p2)
        if self.rnn_type == 'lstm':
            _, (p1_output, _) = self.aggregation(aggr_p1)
            _, (p2_output, _) = self.aggregation(aggr_p2)


        output = torch.cat([torch.cat([p1_output[0,:,:], p1_output[1,:,:]], dim=-1), \
                           torch.cat([p2_output[0,:,:], p2_output[1,:,:]], dim=-1)], dim=-1)

        output = self.dropout(output)
        output = self.attentive_linear(output)
        output = self.dropout(output)
        # Both functions combined together
        if self.model_type == 2:
            output = torch.cat([output, trans_features], dim=-1)
            output = self.ltr(output)
            return output

        p1_aux_input = self.wembeddings(p1_aux)
        p2_aux_input = self.wembeddings(p2_aux)
        p1_aux_norm = p1_aux_input.norm(p=2, dim=2, keepdim=True)
        p2_aux_norm = p2_aux_input.norm(p=2, dim=2, keepdim=True)

        norm_mat = torch.bmm(p1_aux_norm, p2_aux_norm.transpose(1, 2))
        trans_matrix = torch.bmm(p1_aux_input, p2_aux_input.transpose(1, 2))

        # Translation Matrix Constructed here (using cosine similarity)
        trans_matrix = self.cosine_similarity(trans_matrix, norm_mat)
        trans_matrix = trans_matrix * masked_mat_aux
        trans_aux_features = 0
        for i, each in enumerate(self.mu_list):
            trans_matrix_res = torch.exp(-((trans_matrix - each) ** 2) / (2 * (self.sigma) ** 2))
            trans_matrix_res = torch.log1p(trans_matrix_res.sum(dim=2))
            trans_matrix_res = trans_matrix_res.sum(dim=1, keepdim=True)
            if i == 0:
                trans_aux_features = trans_matrix_res
            else:
                trans_aux_features = torch.cat([trans_aux_features, trans_matrix_res], dim=1)

        if self.model_type == 3:
            output = torch.cat([output, trans_features, trans_aux_features], dim=-1)
            self.dropout(output)
            output = self.ltr(output)
            return output

