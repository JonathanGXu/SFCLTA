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


class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x





class VGAE_torch(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5, device="cuda"):
        super(VGAE_torch,self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device


        self.gcn_1 = GCNConv(input_dim,hidden_dim)
        self.active = nn.ReLU()
        self.gcn_2 = GCNConv(hidden_dim,hidden_dim)

        self.mu_layer= nn.Linear(hidden_dim,hidden_dim)
        self.logvar_layer = nn.Linear(hidden_dim,hidden_dim)

    def encoder(self,x,adj):
        h = self.gcn_1(x, adj)
        h = self.active(h)
        h = self.gcn_2(h, adj)

        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        return mu,logvar

    def decoder(self,x):
        recon_adj = F.sigmoid(torch.matmul(x,x.t()))
        return recon_adj

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    def forward(self,x,adj):

        if len(x.shape) >= 3:
            x = x.squeeze(dim=0)
        mu, logvar =self.encoder(x,adj)
        z = self.reparameterize(mu, logvar)
        recon_adj = self.decoder(z)
        loss = self.calculate_loss(recon_adj,adj,mu,logvar)
        self.loss = loss
        return z
    def calculate_loss(self,recon_adj,adj,mu,logvar):

        adj = adj.to_dense()
        loss = F.binary_cross_entropy_with_logits(recon_adj,adj,reduction='mean')
        norm = adj.numel() / (2 * (adj.sum() + 1e-8))
        E = loss * norm

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl / mu.size(0)

        return E+kl

    def recon_loss(self,pred, real=None):

        return self.loss

    def embed(self,x,adj):
        mu, logvar = self.encoder(x, adj)
        z = self.reparameterize(mu, logvar)
        return z