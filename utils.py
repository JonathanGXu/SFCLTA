#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/4/16 21:20 
'''


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy