import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from .Classifier import Classifier

class DeepGNN(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim,class_num,num_layers,task_name,dropout=0.5,module="GCN"):
        super(DeepGNN, self).__init__()
        self.dropout = dropout
        self.task = task_name
        self.loss_fc= nn.CrossEntropyLoss()
        assert self.task in ["node_cls", "graph_cls"]
        self.multi_layers = nn.ModuleList()
        assert num_layers == len(hidden_dim)+1
        if module == "GCN":
            if num_layers == 1:
                self.multi_layers.append(GCNConv(input_dim,output_dim))
            else:
                self.multi_layers.append(GCNConv(input_dim,hidden_dim[0]))
                for i in range(num_layers-2):
                    self.multi_layers.append(GCNConv(hidden_dim[i],hidden_dim[i+1]))
                self.multi_layers.append(GCNConv(hidden_dim[-1],output_dim))

        self.mlp = Classifier(output_dim, class_num)


    def loss(self, pred, label):
        return self.loss_fc(pred, label)

    def forward(self,x,adj):
        x = torch.squeeze(x, dim=0)
        for layer in self.multi_layers:
            x = layer(x,adj)
            x = F.relu(x)
            x = F.dropout(x,self.dropout,training=self.training)
        if self.task == "graph_cls":
            x = torch.mean(x,dim=1)

        x = self.mlp(x)
        x = F.log_softmax(x, dim=-1)

        return x
