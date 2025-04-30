#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：HGAE-SGC 
@File    ：Classifier.py
@IDE     ：PyCharm 
@Author  ：Zhuo
@Date    ：2025/4/15 13:03 
'''


import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
