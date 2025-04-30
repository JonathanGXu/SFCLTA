#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：VGAE.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/4/15 14:38 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.5,device="cuda"):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.output_dim = output_dim
        self.device= device
        self.conv1 = GCNConv(input_dim,hidden_dim)
        self.gcn_mean  = GCNConv(hidden_dim,output_dim)
        self.gcn_logstddev = GCNConv(hidden_dim,output_dim)


    def forward(self,x,adj):
        x = torch.squeeze(x, dim=0)
        hidden = self.conv1(x,adj)
        self.mean = self.gcn_mean(hidden,adj)
        self.logstd = self.gcn_logstddev(hidden,adj)
        gaussian_noise = torch.randn(x.size(0), self.output_dim).to(self.device)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean

        return sampled_z


class VGAE(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.5):
        super(VGAE, self).__init__()
        self.encoder = Encoder(input_dim,hidden_dim,output_dim,dropout)
    def forward(self,x,adj):

        x = self.encoder(x,adj)
        adj = torch.sparse.mm(x, x.T)
        adj = F.sigmoid(adj)
        return adj


    def recon_loss(self, pred, real):
        pred_adj, adj_label = pred, real
        norm = pred_adj.shape[0] * pred_adj.shape[0] / float((pred_adj.shape[0] * pred_adj.shape[0] - pred_adj.sum()) * 2)
        # pred_adj = pred_adj.to_dense()
        loss = norm*F.binary_cross_entropy(pred_adj.reshape(-1),adj_label.to_dense().reshape(-1))
        kl_divergence = 0.5 / pred_adj.size(0) * (1 + 2 * self.encoder.logstd - self.encoder.mean ** 2 - torch.exp(self.encoder.logstd) ** 2).sum(1).mean()

        loss -= kl_divergence

        return loss

    def embed(self,x,adj):
        x = self.encoder(x, adj)
        return x


