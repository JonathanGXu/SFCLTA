#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：HGAE.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/3/5 15:05 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from Datasets.augmentation import augmentate

class EncoderLayer(nn.Module):
    def __init__(self,input_dim,layer_num,hidden_dims, dropout=0.1,device='cuda'):
        """
        割边/增边，完成 Graph Rewiring，类似于带有增边的且不随机的 mask strategy
        :param input_dim:
        :param layer_num:
        :param hidden_dims:
        :param dropout:
        """
        super(EncoderLayer, self).__init__()
        assert layer_num == len(hidden_dims)
        self.layer_num = layer_num
        self.dropout = dropout
        self.device = device
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GCN(input_dim, hidden_dims[0]))

        for i in range(layer_num-1):
            self.encoder_layers.append(GCN(hidden_dims[i], hidden_dims[i+1]))


        self.loss_fc = nn.BCEWithLogitsLoss()
    def forward(self,x,adj):
        """

        :param x: feature matrix [node size, feature dim]
        :param adj: adjacent matrix [node size, node size]
        :return:
        """

        x_ = x
        x = x.squeeze(dim=0)


        for layer in self.encoder_layers:
            x = layer(x, adj)

        for layer in self.encoder_layers:
            x_ = x_.squeeze(dim=0)
            adj_ = torch.matmul(x_,x_.T)

            x_ = layer(x_,adj_)



        return torch.cat([x,x_],dim=0)

    def recon_loss(self,pred,real=None):
        pred = pred.reshape(-1)
        real_shape = pred.shape
        real = torch.cat(
            [torch.ones(real_shape[0] // 2, ).to(self.device), torch.zeros(real_shape[0] // 2, ).to(self.device)])

        return self.loss_fc(pred, real)

    def embed(self, x, adj):

        for layer in self.encoder_layers:
            x = layer(x, adj)

        return x


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=True):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if len(out.shape) ==3:
            out = torch.squeeze(out,dim=0)
        return self.act(out)

class Assignment(nn.Module):
    def __init__(self,input_dim,output_dim,strategy="hard",device="cuda"):
        super(Assignment, self).__init__()
        self.assign_block = GCN(input_dim,output_dim)
        if strategy == "hard":
            self.assgin_strategy = "hard"
        else:
            self.assgin_strategy = "soft"

    def forward(self, x, adj):
        assign_matrix = self.assign_block(x,adj)
        if self.assgin_strategy == "hard":
            assign_matrix = F.gumbel_softmax(assign_matrix,tau=1,hard=True)
            # mean_degree = torch.diag(torch.pow(assign_matrix.sum(dim=0), -1))
            # x = torch.matmul(mean_degree,x)
        else:
            assign_matrix = F.softmax(assign_matrix,dim=1)


        return torch.matmul(assign_matrix.T,x),torch.matmul(torch.matmul(assign_matrix.T,adj),assign_matrix)

class Sub_Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_num,node_num,assign_ratio=0.25,mask_ratio=0.1,dropout=0.5,device='cuda'):
        super(Sub_Encoder, self).__init__()

        self.layer_num = layer_num
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        self.device = device
        self.hidden_dim = hidden_dim

        self.encoder_layers = nn.ModuleList()
        self.subgraph_mask = nn.ModuleList()
        self.graph_node = []

        self.encoder_layers.append(GCN(input_dim,hidden_dim[0]))

        for i in range(self.layer_num-1):
            self.encoder_layers.append(GCN(hidden_dim[i],hidden_dim[i+1]))
            self.subgraph_mask.append(Assignment(input_dim=hidden_dim[i],
                                                 output_dim=int(node_num*np.power(assign_ratio,i+1)),
                                                 strategy="soft"
                                                 )
                                      )
            self.graph_node.append(int(node_num*np.power(assign_ratio,i+1)))
    def forward(self,x,adj):
        self.cluster_x = []
        for i,layer in enumerate(self.encoder_layers):

            x = layer(x,adj)
            if i != self.layer_num-1:
                x, adj = self.subgraph_mask[i](x,adj)
                self.cluster_x.append((x,adj))
                # x = self.mask_subgraph(x,i)
            x = F.dropout(x,p=self.dropout,training=self.training)

        return x, adj

    def recon_loss(self):
        layer_loss = []
        for x,adj in self.cluster_x:

            i,j = torch.where(adj>0)

            length = len(i)
            sim_list =[]
            for index in range(length):
                cos_sim = torch.cosine_similarity(x[i[index]].unsqueeze(dim=0),x[j[index]].unsqueeze(dim=0))
                weighted_cos_sim = cos_sim * adj[i[index]][j[index]]
                sim_list.append(float(weighted_cos_sim))

            layer_loss.append(sum(sim_list))
        return sum(layer_loss)/len(layer_loss)

    def mask_subgraph(self,x,i):
        node_size = self.graph_node[i]
        mask_num = int(self.mask_ratio*node_size)
        no_mask_num = node_size - mask_num

        n_rand = torch.randn(node_size,).to(self.device)
        val,_ = torch.topk(n_rand,no_mask_num)
        mask_index = torch.where(n_rand < val[-1])
        I = torch.diag(torch.ones(node_size,)).to(self.device)
        I[mask_index] = 0
        x = torch.spmm(I, x)

        return x

class Decoder(nn.Module):
    def __init__(self,output_dim,hidden_dim,layer_num,graph_node,dropout=0.5,device='cuda'):
        super(Decoder,self).__init__()

        hidden_dim= hidden_dim[::-1]
        self.graph_node = graph_node[::-1]
        self.layer_num= layer_num
        self.dropout = dropout
        self.device = device

        self.decoder_layers = nn.ModuleList()
        self.generalizing_layers = nn.ModuleList()
        for i in range(self.layer_num-1):
            self.decoder_layers.append(GCN(hidden_dim[i],hidden_dim[i+1]))
            self.generalizing_layers.append(Assignment(hidden_dim[i+1],self.graph_node[i+1],strategy="soft"))

        self.decoder_layers.append(GCN(hidden_dim[-1],output_dim))

    def forward(self,x,adj):
        for i,layer in enumerate(self.decoder_layers):
            x = layer(x,adj)
            if i != self.layer_num-1:
                x,adj= self.generalizing_layers[i](x,adj)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class HGAE(nn.Module):
    def __init__(self,input_dim,hidden_dim,node_num,assign_ratio,dropout=0,device='cuda'):
        super(HGAE, self).__init__()

        self.dropout = dropout
        self.device = device
        self.layer_num = len(hidden_dim)

        self.encoder= Sub_Encoder(input_dim,hidden_dim,self.layer_num,node_num=node_num,assign_ratio=assign_ratio,
                                  dropout=self.dropout,device=self.device)
        node_list = self.encoder.graph_node
        self.decoder = Decoder(input_dim,hidden_dim,self.layer_num,[node_num]+node_list,
                               dropout=self.dropout,device=self.device)

        self.loss_fc1 = nn.MSELoss()
    def recon_loss(self,pred,real=None):
        real = self.real
        loss_sim = self.loss_fc1(pred, real)
        loss_homo = self.encoder.recon_loss()

        return loss_sim+loss_homo


    def forward(self,x,adj):
        x = torch.squeeze(x,dim=0)
        self.real = x
        x,adj = self.encoder(x,adj)
        output= self.decoder(x,adj)

        return output


    def embed(self,x,adj):
        x, adj = self.encoder(x, adj)
        output = self.decoder(x, adj)

        return output


class Heter_GCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,data_aug=0.0,data_aug_source=None,dropout=0.5,device="cuda"):
        super(Heter_GCN,self).__init__()

        self.dropout = dropout
        self.device = device

        self.low_gcn = GCN(input_dim,hidden_dim)
        self.high_gcn = GCN(input_dim,hidden_dim)

        self.loss_fc1= nn.BCEWithLogitsLoss()
        self.loss_fc2 = nn.MSELoss()

        self.disc = Discriminator(n_h=hidden_dim)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()


        self.decoder = nn.Linear(hidden_dim*2,input_dim)

        if data_aug > 0:
            self.data_aug = True
            self.adj_aug = augmentate(data_aug_source,data_aug)
            self.adj_aug = self.adj_aug.to(self.device)
    def forward(self,x,adj):
        x = torch.squeeze(x,dim=0)
        x_low = self.low_gcn(x,adj)

        if self.data_aug is False:
            d_indices = torch.sum(adj,dim=1).indices().repeat(2,1)
            d = torch.sparse_coo_tensor(d_indices,torch.sum(adj,dim=1).values(),adj.size()).to(self.device)
            x_high = self.high_gcn(x,d-adj)
        else:
            d_indices = torch.sum(self.adj_aug, dim=1).indices().repeat(2, 1)
            d = torch.sparse_coo_tensor(d_indices, torch.sum(self.adj_aug, dim=1).values(), self.adj_aug.size()).to(self.device)
            x_high = self.high_gcn(x, d - self.adj_aug)

        x_cat = torch.cat([x_low,x_high],dim=1)

        c = self.read(x_low.unsqueeze(dim=0), None)
        c = self.sigm(c)
        ret = self.disc(c, x_low.unsqueeze(dim=0), x_high.unsqueeze(dim=0), None, None)

        recon_x = self.decoder(x_cat)
        self.loss2 = self.loss_fc2(recon_x,x)

        return ret

    def recon_loss(self,pred,real=None):
        pred = pred.reshape(-1)
        real_shape = pred.shape
        real = torch.cat(
            [torch.ones(real_shape[0] // 2, ).to(self.device), torch.zeros(real_shape[0] // 2, ).to(self.device)])
        l1 = self.loss_fc1(pred, real)
        loss = l1+self.loss2
        return loss

    def embed(self,x,adj):
        x = torch.squeeze(x, dim=0)
        x_low = self.low_gcn(x, adj)

        return x_low

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
