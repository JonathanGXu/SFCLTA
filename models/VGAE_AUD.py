import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

class CGAE(nn.Module):
    def __init__(self,input_dim,hidden_dim,dropout=0.5,device="cuda"):
        super(CGAE, self).__init__()
        self.input_dim =input_dim
        self.hidden_dim =hidden_dim
        self.dropout = dropout
        self.device = device

        self.encoder = GCN(hidden_dim,hidden_dim*2)
        self.decoder = GCN(hidden_dim,input_dim)

    def guide(self,x,adj):
        pyro.module("encoder",self.encoder)
        with pyro.plate("data",x.shape[0]):
            self.condatitional(x,adj)
            z = self.encoder(x, self.hat_adj)
            z_loc, z_logvar = z[:, :self.hidden_dim], z[:, self.hidden_dim:]
            z_scale = torch.exp(0.5 * z_logvar)
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    def model(self,x,adj):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros(x.shape[0], self.hidden_dim).to(self.device)
            z_scale = torch.ones(x.shape[0], self.hidden_dim).to(self.device)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            x_recon = self.decoder(z,self.hat_adj)
            pyro.sample("obs", dist.Normal(x_recon, 0.1).to_event(1), obs=x.flatten(1))

    def condatitional(self,x,adj):
        h = F.sigmoid(torch.sparse.mm(x, x.T))
        self.hat_adj = torch.mul(h, adj)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)