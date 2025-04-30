#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：dataset.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/4/12 12:18 
'''

from torch_geometric.datasets import Planetoid,Reddit,PPI,Amazon
import scipy.sparse as sp
import numpy as np
import torch

from Datasets.preprocess import load_data,normalize_adj,sparse_mx_to_torch_sparse_tensor,preprocess_features

def init_dataset(dataset_name, supervised=False,device="cuda",print_data=True,root="./Datasets/"):
    # Cora CiteSeer PubMed
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root+dataset_name, name=dataset_name)
        if print_data is True:
            print_dataset(dataset)

        adj, features, labels, idx_train, idx_val, idx_test = load_data(root=root+dataset_name+"/"+dataset_name,
                                                                        dataset_str=dataset_name.lower())

        if supervised is True:
            train, val, test = split_dataset(adj, features, labels,
                                             idx_train, idx_val, idx_test, supervised=supervised,device=device)

            return train, val, test, dataset.num_node_features, dataset.num_classes

        else:
            features, adj, labels, idx_train, idx_val, idx_test = split_dataset(adj, features, labels,
                                                idx_train, idx_val, idx_test, supervised=supervised,device=device)

            return features, adj, labels, idx_train, idx_val,idx_test,dataset.num_classes


    # Reddit
    if dataset_name == 'Reddit':
        dataset = Reddit(root=root + 'Reddit')

    # Amazon
    elif dataset_name == "Computers":
        dataset = Amazon(root=root + 'Computers',name='Computers')

    elif dataset_name == "Photo":
        dataset = Amazon(root=root + 'Photo', name='Photo')

    elif dataset_name == "PPI":
        dataset = PPI(root=root + 'PPI')

    else:
        raise ValueError("There is no datasets named {}".format(dataset_name))

    if print_data is True:
        print_dataset(dataset)

    return dataset

def split_dataset(adj, features, labels, idx_train, idx_val, idx_test,supervised=True,sparse=True,device='cuda'):
    features, _ = preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if device == 'cuda':
        print('Using CUDA')

        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    if supervised is True:
        train_nodes = torch.index_select(features,dim=1, index=idx_train)
        val_nodes = torch.index_select(features,dim=1, index=idx_val)
        test_nodes = torch.index_select(features,dim=1, index=idx_test)

        train_lbls = torch.argmax(labels[0, idx_train], dim=1)
        val_lbls = torch.argmax(labels[0, idx_val], dim=1)
        test_lbls = torch.argmax(labels[0, idx_test], dim=1)

        if sparse is True:
            train_sp_adj = sp_adj.index_select(0,idx_train).index_select(1,idx_train)
            val_sp_adj = sp_adj.index_select(0, idx_val).index_select(1, idx_val)
            test_sp_adj = sp_adj.index_select(0, idx_test).index_select(1, idx_test)

            return (train_nodes, train_sp_adj,train_lbls), (val_nodes,val_sp_adj,val_lbls), (test_nodes,test_sp_adj,test_lbls)

        else:
            train_adj = torch.index_select(torch.index_select(adj,dim=1, index=idx_train),dim=2,index=idx_train)
            val_adj = torch.index_select(torch.index_select(adj, dim=1, index=idx_val), dim=2, index=idx_val)
            test_adj = torch.index_select(torch.index_select(adj, dim=1, index=idx_test), dim=2, index=idx_test)

            return (train_nodes, train_adj, train_lbls), (val_nodes, val_adj, val_lbls), (test_nodes, test_adj, test_lbls)

    else:
        if sparse is True:

            return features, sp_adj, labels, idx_train, idx_val, idx_test

        else:
            return features, adj, labels, idx_train, idx_val, idx_test

def print_dataset(dataset):

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print("The first graph of the dataset. Note that, some datasets have only one graph.")
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print('=============================================================')

    check_graph_format(dataset)

def check_graph_format(dataset):
    data = dataset[0]
    print(data.x)

    print(data.edge_index)

    print(data.y)


if __name__ == '__main__':
    init_dataset("Reddit",root="./")
