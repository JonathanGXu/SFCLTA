#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：Resistance.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/4/9 17:16 
'''

import math
import torch
import numpy as np
import networkx as nx


def calculate_resistance(adj):
    """

    :param adj: adjacent matrix [batch size, node num, node num]
    :return:
    """
    deg = torch.diag(torch.sum(adj,dim=2))
    laplacian = deg - adj
    laplacian_pseudoinverse = torch.pinverse(laplacian,rcond=1e-15)

    adj_shape = adj.shape
    batch_size = adj_shape[0]
    node_num= adj.shape[-1]
    adj_selfloops = torch.zeros(adj_shape)+torch.diag(torch.ones(adj_shape[-1]))
    norm_lap_self = adj_selfloops - torch.matmul(torch.matmul(torch.pow(deg,-0.5),adj),torch.pow(deg,-0.5))
    hat_lap_pse = torch.pinverse(norm_lap_self,rcond=1e-15)

    for i in range(node_num):
        for j in range(node_num):
            node_i = torch.zeros(batch_size,node_num,1)
            node_i[:,i,:] = 1
            node_j = torch.zeros(batch_size,node_num,1)
            node_j[:,j,:] = 1
            node = node_i * torch.pow(torch.sum(adj),-0.5) - node_j*torch.pow(torch.sum(adj),-0.5)
            r_i_j = torch.matmul(torch.matmul(node.permute(0,2,1), hat_lap_pse),node)
