#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/4/9 16:08 
'''

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                        f'Training Accuracy: {accuracy["train"] * 100:.2f}% \n'
                        f' Validation Accuracy: {accuracy["val"] * 100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()

# G = to_networkx(data, to_undirected=True)
# visualize(G, color=data.y)