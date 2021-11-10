import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=F.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act

        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))

    def forward(self, H_T):
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)


class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=F.tanh):
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))

    def forward(self, H):
        z1 = self.act(H.mm(self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)


class decoder2(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)
        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return z


# hypergraph conv
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor

        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


"""
class HGNN1(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN1, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)

    def forward(self, x, G):
        x = F.dropout(x, self.dropout)
        x = F.tanh(self.hgc1(x, G))
        return x
"""


# HGNN with 2 hypergraph conv layers
class HGNN2(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, n_node, emb_dim, dropout=0.5):
        super(HGNN2, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.feat = nn.Embedding(n_node, emb_dim)
        self.feat_idx = torch.arange(n_node).cuda()
        nn.init.xavier_uniform_(self.feat.weight)

    def forward(self, x, G):
        x = self.feat(self.feat_idx)
        #x = F.dropout(x, self.dropout)
        x = F.tanh(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class self_Attention(nn.Module):
    def __init__(self, num, num_in, num_hidden):
        super(self_Attention, self).__init__()
        self.num = num
        self.num_in = num_in
        self.hidden = num_hidden
        self.act1 = F.tanh
        self.act2 = F.softmax
        self.Wr = nn.Parameter(torch.zeros(size=(self.num_in, self.hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.Wr.data, gain=0)
        self.b1 = nn.Parameter(torch.zeros(self.hidden, dtype=torch.float))
        self.P = nn.Parameter(torch.zeros(size=(self.hidden, 1), dtype=torch.float))
        nn.init.xavier_uniform_(self.P.data, gain=0)
        self.Mr = nn.Parameter(torch.zeros(size=(self.num, self.num), dtype=torch.float))

    # def reset_parameters(self):
    #     self.Wr.data.normal_(std=1.0 / math.sqrt(self.num_in))
    #     self.wr.data.normal_(std=1.0 / math.sqrt(self.num_in))
    #     self.Mr.data.normal_(std=1.0 / math.sqrt(self.num_in))

    def forward(self, embedding):
        alpha = self.act1(embedding.mm(self.Wr + self.b1)).mm(self.P)
        # emb = embedding * alpha
        return alpha

