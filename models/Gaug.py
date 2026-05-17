import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import degree
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx

class GSL(nn.Module):
    def __init__(self,input_dim,hidden_dim,layers_num=1,dropout=0.5,device="cuda"):
        super(GSL, self).__init__()

        self.layers_num = layers_num

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder.append(GCNConv(input_dim,hidden_dim))
        for i in range(layers_num-1):
            self.encoder.append(GCNConv(hidden_dim,hidden_dim))
        for i in range(layers_num-1):
            self.decoder.append(GCNConv(hidden_dim,hidden_dim))
        self.decoder.append(GCNConv(hidden_dim,input_dim))

        self.sigmoid = nn.Sigmoid()
    def forward(self,x,adj):
        for encoder_layer in self.encoder:
            x = encoder_layer(x,adj)
        for decoder_layer in self.decoder:
            x = decoder_layer(x,adj)
        hat_adj = self.sigmoid(torch.mm(x,x.T))
        return hat_adj

class ESGSL(nn.Module):
    def __init__(self,x,edge_index,alpha=0.5):
        super(ESGSL, self).__init__()

        self.graph = Data(x=x,edge_index=edge_index)
        self.node_size = self.graph.num_nodes
        self.feature_dim = self.graph.num_features

        self.low_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.high_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.alpha = alpha
        mask = self.select_anchor_nodes(topk_ratio=0.005)
    def select_anchor_nodes(self, topk_ratio=0.005):

        deg = degree(self.graph.edge_index[0], num_nodes=self.graph.num_nodes)
        aug_node_id = torch.topk(deg, k=int(self.graph.num_nodes * topk_ratio)).indices
        aug_node_id = aug_node_id.tolist()
        nodes_list = []
        mask = torch.zeros(self.node_size, )

        for anchor_node in aug_node_id:
            neighbor_nodes, edge_index, mapping, edge_mask = k_hop_subgraph(
                anchor_node,
                1,
                self.graph.edge_index,
                num_nodes=self.graph.num_nodes
            )
            nodes_list+= neighbor_nodes
            mask[anchor_node] = 1.0

        mask = torch.diag(mask)
        return mask

    def feature_aug(self,x,adj):


        degree = torch.diag(adj.sum(dim=1))
        L = degree - adj

        eigvals, eigvecs = torch.linalg.eigh(L)

        x_hat = torch.mm(eigvecs.t(), x)

        low_x = torch.mm(eigvecs[:, :self.k], self.low_proj(x_hat[:self.k]))

        high_x = torch.mm(eigvecs[:, self.k:], self.high_proj(x_hat[self.k:]))

        fea_aug = torch.mm(self.mask,(self.alpha * low_x + self.alpha * high_x))

        return fea_aug + x

    def forward(self, x, adj):

        x = self.feature_aug(x,adj)

        return x


class DynamicAdjGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.time_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, hidden_dim)
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, H_prev):

        h_src = H_prev.unsqueeze(1).expand(-1, H_prev.size(0), -1)  # [n, n, d]

        pair_feat = h_src
        A_prob = self.edge_predictor(pair_feat).squeeze(-1)  # [n, n]

        return torch.where(A_prob > 0.5, A_prob, torch.zeros_like(A_prob))


    def high_freq_preserve(self, adj, X, k=5):

        D = torch.diag(adj.sum(1))
        L = D - adj
        L_norm = torch.eye(adj.size(0)) - torch.mm(torch.inv(D), adj)

        eigvals, eigvecs = torch.linalg.eigh(L_norm)

        _, idx = torch.topk(eigvals.abs(), k, largest=True)
        U_high = eigvecs[:, idx]  # 高频基向量

        X_hat = torch.mm(U_high.t(), X)
        energy = torch.norm(X_hat, p='fro')

        return -energy

    def temporal_smoothness(self,A_curr, A_prev):

        delta_A = A_curr - self.to_dense(A_prev,A_curr.shape[0])

        return torch.norm(delta_A, p=1)  # L1范数促进稀疏变化


    def to_dense(self,sparse_adj,node_size):
        edge_index = sparse_adj.indices
        adjacency_matrix = torch.zeros(node_size, node_size, dtype=torch.uint8)
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
        adjacency_matrix[edge_index[1], edge_index[0]] = 1
        return adjacency_matrix


class ImplicitAdjacency(nn.Module):
    def __init__(self, input_dim, proj_dim,aug_data_source,indices="Anchor",device="cuda"):
        super().__init__()

        self.device = device

        self.data_source = aug_data_source
        self.noed_size = aug_data_source.x.shape[0]


        self.relation_gen = nn.Sequential(
            nn.Linear(2 * input_dim, proj_dim),

            nn.Linear(proj_dim, 1)
        )

        if indices == "Anchor":
            indices = self.rand_sample_bynode(self.data_source,augmentation_ratio=0.005)
            self.src, self.dst = indices[:, 0], indices[:, 1]
            self.indices = indices.T

    def forward(self, node_embs):


        # print(src.shape)
        emb_pairs = torch.cat([node_embs[self.src], node_embs[self.dst]], dim=1)

        weights = F.sigmoid(self.relation_gen(emb_pairs)).squeeze()

        sp_edge_matrix = SafeSparseEnhance.apply(self.indices, weights, self.noed_size)

        return sp_edge_matrix

    def random_pair_sampling(self, n, sample_size=100):
        if sample_size is None:
            sample_size = int(4 * n * int(math.log(n)))  # O(n log n)复杂度

        probs = torch.ones(n, n) - torch.eye(n)

        return torch.multinomial(probs.flatten(), sample_size).reshape(-1, 2)

    def rand_sample_bynode(self,data, augmentation_ratio=0.001):


        x = data.x
        node_size = data.num_nodes

        index_select_rand = torch.randn(node_size, )
        select_node_size = int(augmentation_ratio * node_size)
        _, node_index = torch.topk(index_select_rand, select_node_size)
        select_node = x[node_index]

        add_edge_list = []
        for i in range(select_node.shape[0]):
            node = select_node[i]
            sim_value = torch.cosine_similarity(node, x, dim=1)
            _, target_id = torch.topk(sim_value, select_node_size)
            add_edge = [[node_index[i], int(tar)] for tar in target_id[1:]]
            add_edge_list+= add_edge


        return torch.LongTensor(add_edge_list).to(self.device)


class SafeSparseEnhance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weights, num_nodes):
        indices = edge_index.detach().clone().long()
        sparse_adj = torch.sparse_coo_tensor(
            indices,
            edge_weights,
            (num_nodes, num_nodes),

        )
        sparse_adj = sparse_adj.coalesce()

        ctx.save_for_backward(edge_weights, indices)
        ctx.num_nodes = num_nodes

        return sparse_adj

    @staticmethod
    def backward(ctx, grad_output):
        edge_weights, indices = ctx.saved_tensors

        grad_weights = None
        grad_index = None

        if ctx.needs_input_grad[1]:
            grad_weights = torch.zeros_like(edge_weights)

            for i in range(indices.shape[1]):
                row, col = indices[:, i]
                grad_weights[i] = grad_output[row, col]
        return None, grad_weights, None

