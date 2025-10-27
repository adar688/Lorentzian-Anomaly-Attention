import os
import sys

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.hgcn.layers.hyplayers import HGATConv, HGCNConv, TemporalAttentionLayer
from script.hgcn.manifolds import PoincareBall, Hyperboloid, Euclidean


class Dynhat(nn.Module):
    def __init__(self, args, time_length):
        super(Dynhat, self).__init__()
        self.manifold_name = args.manifold
        
        
        # self.manifold = PoincareBall()
        # if self.manifold_name == 'PoincareBall':
        #     agg_feat_size = args.nhid
        #     self.manifold = PoincareBall()
        if self.manifold_name == 'Hyperboloid':
            
            if args.temporal_attention_layer_heads == 1:
                agg_feat_size = args.nhid + 1
            else: 
                # 考虑到Attention head要进行split, 因此需要为偶数
                agg_feat_size = args.nhid + 1 * args.temporal_attention_layer_heads
            self.manifold = Hyperboloid()
        # elif self.manifold_name == 'Euclidean':
        #     agg_feat_size = args.nhid
        #     self.manifold = Euclidean()

        if args.fix_curvature:
            # self.c = 1
            self.c = Parameter(torch.ones(1).to(args.device), requires_grad=False)
        # else:
        #     self.c = Parameter((torch.ones(1)).to(args.device), requires_grad=True)

        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)

        self.linear = nn.Linear(args.nfeat, args.nout)
        
        self.structural_heads = args.heads
        self.temporal_heads = args.temporal_attention_layer_heads
        # if args.aggregation == 'deg':
        #     self.layer1 = HGCNConv(self.manifold, agg_feat_size, agg_feat_size, self.c, self.c,
        #                            dropout=args.dropout)
        #     self.layer2 = HGCNConv(self.manifold, agg_feat_size, agg_feat_size, self.c, self.c, dropout=args.dropout)
        if args.aggregation == 'att':
            self.layer1 = HGATConv(self.manifold, agg_feat_size, agg_feat_size, self.c, self.c,
                                heads=self.structural_heads, dropout=args.dropout, att_dropout=args.dropout, concat=True)
            self.layer2 = HGATConv(self.manifold, agg_feat_size * self.structural_heads, agg_feat_size, self.c, self.c,
                                    heads=self.structural_heads, dropout=args.dropout, att_dropout=args.dropout, concat=False)
        
        self.cell_hidden = torch.ones(args.num_nodes, agg_feat_size).to(args.device)
        self.cell_hidden2 = torch.ones(args.num_nodes, agg_feat_size).to(args.device)
        # if args.seq_model == 'RNN':
        #     # self.seq_model = nn.RNN(args.nout, args.nout, 1, batch_first = True)
        #     self.seq_model = nn.RNN(agg_feat_size, agg_feat_size, 1, batch_first = True)
        # elif args.seq_model == 'RNNCell':
        #     self.seq_model = nn.RNNCell(agg_feat_size, agg_feat_size)
        # elif args.seq_model == 'LSTM':
        #     # self.seq_model = nn.LSTM(args.nout, args.nout, 1, batch_first = True)
        #     self.seq_model = nn.LSTM(agg_feat_size, agg_feat_size, 1, batch_first = True)
        # elif args.seq_model == 'BiLSTM':
        #     # self.seq_model = nn.LSTM(args.nout, args.nout, 1, batch_first = True)
        #     self.seq_model = nn.LSTM(agg_feat_size, agg_feat_size, 1, batch_first = True, bidirectional = True)
        # elif args.seq_model == 'LSTMCell':
        #     self.seq_model = nn.LSTMCell(agg_feat_size, agg_feat_size)
        # elif args.seq_model == 'GRU':
        #     self.seq_model = nn.GRU(agg_feat_size, agg_feat_size, 1, batch_first = True)
        self.seq_model = TemporalAttentionLayer(
                input_dim=agg_feat_size, 
                n_heads=self.temporal_heads, 
                num_time_steps=time_length,  
                attn_drop=0,  # dropout
                residual=False  
                )
        self.seq_model.name = args.seq_model
        self.nhid = args.nhid
        self.nout = args.nout
        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.cell_hidden)
        glorot(self.cell_hidden2)
    

    def initHyperX(self, x, temporal_heads, c=1.0):
        if self.manifold_name == 'Hyperboloid':
            o = torch.zeros_like(x)
            # temporal_heads防止多头出现奇数 无法整除
            x = torch.cat([o[:, 0: 1 * temporal_heads], x], dim=1)
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c), c)
        return x


    def forward(self, edge_index, x=None, weight=None):
        x = self.linear(x)
        x = self.initHyperX(x, self.temporal_heads, self.c)
        x = self.manifold.proj(x, self.c)
        x = self.layer1(x, edge_index)
        x = self.manifold.proj(x, self.c)
        x = self.layer2(x, edge_index)
        x = self.toTangentX(x, self.c)

        # if self.seq_model.name == 'RNN' or self.seq_model.name == 'LSTM' or self.seq_model.name == 'BiLSTM' or self.seq_model.name == 'GRU':
        #     # 增加一个batch_size维度
        #     x=torch.unsqueeze(x, 0) # [input_size, feature_size]=> [1, 184, 16]
        #     x, h_n = self.seq_model(x)
        #     # # 减少一个batch_size维度
        #     x = torch.squeeze(x, 0) # [batch_size, input_size, feature_size]=> [184, 16]
        #     x = self.toHyperX(x, self.c)  # to hyper space
        # elif self.seq_model.name == 'RNNCell':
        #     cell_hidden = self.seq_model(x, self.cell_hidden)
        #     self.cell_hidden = cell_hidden.detach()
        #     x = self.toHyperX(cell_hidden, self.c)
        # elif self.seq_model.name == 'LSTMCell':
        #     cell_hidden, cell_hidden2 = self.seq_model(x, (self.cell_hidden, self.cell_hidden2))
        #     self.cell_hidden = cell_hidden.detach()
        #     self.cell_hidden2 = cell_hidden2.detach()
        #     x = self.toHyperX(cell_hidden, self.c)
        return x
