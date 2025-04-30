#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：augmentation.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/4/24 10:54 
'''

import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx

def augmentate(data,augmentation_ratio=0.1):
    """

    :param dataset:
    :return:
    """

    x = data.x
    edge_index = data.edge_index
    node_size = data.num_nodes

    G = nx.Graph()
    G.add_nodes_from(range(node_size))
    G.add_edges_from(edge_index.cpu().numpy().T)

    index_select_rand = torch.randn(node_size,)
    select_node_size = int(augmentation_ratio * node_size)
    _,node_index = torch.topk(index_select_rand,select_node_size)
    select_node= x[node_index]

    for i in range(select_node.shape[0]):
        node = select_node[i]
        sim_value= torch.cosine_similarity(node,x,dim=1)
        _, target_id = torch.topk(sim_value,select_node_size)
        add_edge = [[node_index[i],int(tar)] for tar in target_id[1:]]
        G.add_edges_from(add_edge)

    aug_edge_index = np.array(G.edges()).T
    edge_num = aug_edge_index.shape[-1]
    sp_edge_matrix = torch.sparse_coo_tensor(torch.Tensor(aug_edge_index), torch.ones(edge_num,), torch.Size((data.num_nodes,data.num_nodes)))


    return sp_edge_matrix
