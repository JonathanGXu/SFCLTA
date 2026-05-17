import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from Datasets.augmentation import augmentate
from models.Gaug import GSL,ESGSL, DynamicAdjGenerator, ImplicitAdjacency

class EncoderLayer(nn.Module):
    def __init__(self,input_dim,layer_num,hidden_dims, dropout=0.1,device='cuda'):

        super(EncoderLayer, self).__init__()
        assert layer_num == len(hidden_dims)
        self.layer_num = layer_num
        self.dropout = dropout
        self.device = device
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GCN(input_dim, hidden_dims[0]))

        for i in range(layer_num-1):
            self.encoder_layers.append(GCN(hidden_dims[i], hidden_dims[i+1]))


        self.loss_fc = nn.BCEWithLogitsLoss()
    def forward(self,x,adj):

        x_ = x
        x = x.squeeze(dim=0)


        for layer in self.encoder_layers:
            x = layer(x, adj)

        for layer in self.encoder_layers:
            x_ = x_.squeeze(dim=0)
            adj_ = torch.matmul(x_,x_.T)

            x_ = layer(x_,adj_)



        return torch.cat([x,x_],dim=0)

    def recon_loss(self,pred,real=None):
        pred = pred.reshape(-1)
        real_shape = pred.shape
        real = torch.cat(
            [torch.ones(real_shape[0] // 2, ).to(self.device), torch.zeros(real_shape[0] // 2, ).to(self.device)])

        return self.loss_fc(pred, real)

    def embed(self, x, adj):

        for layer in self.encoder_layers:
            x = layer(x, adj)

        return x


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
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

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=True):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if len(out.shape) ==3:
            out = torch.squeeze(out,dim=0)
        return self.act(out)

class Assignment(nn.Module):
    def __init__(self,input_dim,output_dim,strategy="hard",device="cuda"):
        super(Assignment, self).__init__()
        self.assign_block = GCN(input_dim,output_dim)
        if strategy == "hard":
            self.assgin_strategy = "hard"
        else:
            self.assgin_strategy = "soft"

    def forward(self, x, adj):
        assign_matrix = self.assign_block(x,adj)
        if self.assgin_strategy == "hard":
            assign_matrix = F.gumbel_softmax(assign_matrix,tau=1,hard=True)
            # mean_degree = torch.diag(torch.pow(assign_matrix.sum(dim=0), -1))
            # x = torch.matmul(mean_degree,x)
        else:
            assign_matrix = F.softmax(assign_matrix,dim=1)


        return torch.matmul(assign_matrix.T,x),torch.matmul(torch.matmul(assign_matrix.T,adj),assign_matrix)

class Sub_Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_num,node_num,assign_ratio=0.25,mask_ratio=0.1,dropout=0.5,device='cuda'):
        super(Sub_Encoder, self).__init__()

        self.layer_num = layer_num
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        self.device = device
        self.hidden_dim = hidden_dim

        self.encoder_layers = nn.ModuleList()
        self.subgraph_mask = nn.ModuleList()
        self.graph_node = []

        self.encoder_layers.append(GCN(input_dim,hidden_dim[0]))

        for i in range(self.layer_num-1):
            self.encoder_layers.append(GCN(hidden_dim[i],hidden_dim[i+1]))
            self.subgraph_mask.append(Assignment(input_dim=hidden_dim[i],
                                                 output_dim=int(node_num*np.power(assign_ratio,i+1)),
                                                 strategy="soft"
                                                 )
                                      )
            self.graph_node.append(int(node_num*np.power(assign_ratio,i+1)))
    def forward(self,x,adj):
        self.cluster_x = []
        for i,layer in enumerate(self.encoder_layers):

            x = layer(x,adj)
            if i != self.layer_num-1:
                x, adj = self.subgraph_mask[i](x,adj)
                self.cluster_x.append((x,adj))
                # x = self.mask_subgraph(x,i)
            x = F.dropout(x,p=self.dropout,training=self.training)

        return x, adj

    def recon_loss(self):
        layer_loss = []
        for x,adj in self.cluster_x:

            i,j = torch.where(adj>0)

            length = len(i)
            sim_list =[]
            for index in range(length):
                cos_sim = torch.cosine_similarity(x[i[index]].unsqueeze(dim=0),x[j[index]].unsqueeze(dim=0))
                weighted_cos_sim = cos_sim * adj[i[index]][j[index]]
                sim_list.append(float(weighted_cos_sim))

            layer_loss.append(sum(sim_list))
        return sum(layer_loss)/len(layer_loss)

    def mask_subgraph(self,x,i):
        node_size = self.graph_node[i]
        mask_num = int(self.mask_ratio*node_size)
        no_mask_num = node_size - mask_num

        n_rand = torch.randn(node_size,).to(self.device)
        val,_ = torch.topk(n_rand,no_mask_num)
        mask_index = torch.where(n_rand < val[-1])
        I = torch.diag(torch.ones(node_size,)).to(self.device)
        I[mask_index] = 0
        x = torch.spmm(I, x)

        return x

class Decoder(nn.Module):
    def __init__(self,output_dim,hidden_dim,layer_num,graph_node,dropout=0.5,device='cuda'):
        super(Decoder,self).__init__()

        hidden_dim= hidden_dim[::-1]
        self.graph_node = graph_node[::-1]
        self.layer_num= layer_num
        self.dropout = dropout
        self.device = device

        self.decoder_layers = nn.ModuleList()
        self.generalizing_layers = nn.ModuleList()
        for i in range(self.layer_num-1):
            self.decoder_layers.append(GCN(hidden_dim[i],hidden_dim[i+1]))
            self.generalizing_layers.append(Assignment(hidden_dim[i+1],self.graph_node[i+1],strategy="soft"))

        self.decoder_layers.append(GCN(hidden_dim[-1],output_dim))

    def forward(self,x,adj):
        for i,layer in enumerate(self.decoder_layers):
            x = layer(x,adj)
            if i != self.layer_num-1:
                x,adj= self.generalizing_layers[i](x,adj)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class HGAE(nn.Module):
    def __init__(self,input_dim,hidden_dim,node_num,assign_ratio,dropout=0,device='cuda'):
        super(HGAE, self).__init__()

        self.dropout = dropout
        self.device = device
        self.layer_num = len(hidden_dim)

        self.encoder= Sub_Encoder(input_dim,hidden_dim,self.layer_num,node_num=node_num,assign_ratio=assign_ratio,
                                  dropout=self.dropout,device=self.device)
        node_list = self.encoder.graph_node
        self.decoder = Decoder(input_dim,hidden_dim,self.layer_num,[node_num]+node_list,
                               dropout=self.dropout,device=self.device)

        self.loss_fc1 = nn.MSELoss()
    def recon_loss(self,pred,real=None):
        real = self.real
        loss_sim = self.loss_fc1(pred, real)
        loss_homo = self.encoder.recon_loss()

        return loss_sim+loss_homo


    def forward(self,x,adj):
        x = torch.squeeze(x,dim=0)
        self.real = x
        x,adj = self.encoder(x,adj)
        output= self.decoder(x,adj)

        return output


    def embed(self,x,adj):
        x, adj = self.encoder(x, adj)
        output = self.decoder(x, adj)

        return output


class Heter_GCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,data_aug=0.0,data_aug_source=None,aug_method="",dropout=0.5,device="cuda"):
        super(Heter_GCN,self).__init__()

        self.dropout = dropout
        self.device = device

        self.low_gcn = GCN(input_dim,hidden_dim)
        self.high_gcn = GCN(input_dim,hidden_dim)

        self.loss_fc1= nn.BCEWithLogitsLoss()
        self.loss_fc2 = nn.MSELoss()


        self.disc = Discriminator(n_h=hidden_dim)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        self.loss_fc = InfoNCELoss()

        self.decoder = nn.Linear(hidden_dim,input_dim)

        if data_aug > 0 and aug_method == "":
            self.data_aug = True
            self.aug_method  = ""
            self.adj_aug = augmentate(data_aug_source,data_aug)
            self.adj_aug = self.adj_aug.to(self.device)

        elif aug_method == "DTAM":
            self.data_aug = True
            self.aug_method = "DTAM"
            self.adj_aug = DynamicAdjGenerator(hidden_dim=input_dim)

        elif aug_method == "IA":
            self.data_aug = True
            self.aug_method = "IA"
            self.adj_aug = ImplicitAdjacency(input_dim=input_dim ,
                                             proj_dim=hidden_dim,aug_data_source=data_aug_source)

        else:
            self.data_aug = False


    def forward(self,x,adj):
        x = torch.squeeze(x,dim=0)
        x_low = self.low_gcn(x,adj)

        d_indices = torch.sum(adj, dim=1).indices().repeat(2, 1)
        d = torch.sparse_coo_tensor(d_indices, torch.sum(adj, dim=1).values(), adj.size()).to(self.device)
        x_high = self.high_gcn(x+ 40.0 * torch.randn(x.shape).to(self.device), d - adj)

        if self.data_aug is False:
            d_indices = torch.sum(adj,dim=1).indices().repeat(2,1)
            d = torch.sparse_coo_tensor(d_indices,torch.sum(adj,dim=1).values(),adj.size()).to(self.device)
            x_high = self.high_gcn(x ,d-adj)


        elif self.aug_method == "DTAM":

            hat_adj = self.adj_aug(x)
            hat_x_high = self.high_gcn(x, torch.diag(adj.sum(1)) - hat_adj)
            hat_x_low = self.low_gcn(x, hat_adj)
            self.loss1 = self.adj_aug.high_freq_preserve(x,hat_adj)+self.adj_aug.temporal_smoothness(hat_adj,adj)+self.loss_fc(x_low, hat_x_low) + self.loss_fc(x_high, hat_x_high)

        elif self.aug_method == "IA":

            hat_adj = self.adj_aug(x)

            hat_adj = hat_adj + adj
            hat_x_low = self.low_gcn(x, hat_adj)

            hat_adj_indices = torch.sum(hat_adj, dim=1).indices().repeat(2, 1)
            hat_adj_d = torch.sparse_coo_tensor(hat_adj_indices, torch.sum(hat_adj, dim=1).values(), hat_adj.size()).to(self.device)
            hat_x_high = self.high_gcn(x, hat_adj_d - hat_adj)

            self.loss1 =  self.loss_fc(x_low, hat_x_low) + self.loss_fc(x_high, hat_x_high)


        else:
            d_indices = torch.sum(self.adj_aug, dim=1).indices().repeat(2, 1)
            d = torch.sparse_coo_tensor(d_indices, torch.sum(self.adj_aug, dim=1).values(), self.adj_aug.size()).to(self.device)
            hat_x_high = self.high_gcn(x, d - self.adj_aug)
            hat_x_low = self.low_gcn(x,self.adj_aug)
            self.loss1 = self.loss_fc(x_low,hat_x_low) + self.loss_fc(x_high,hat_x_high)

        x_cat = torch.cat([x_low,x_high],dim=1)

        c = self.read(x_low.unsqueeze(dim=0), None)
        c = self.sigm(c)
        ret = self.disc(c, x_low.unsqueeze(dim=0), x_high.unsqueeze(dim=0), None, None)

        recon_x = self.decoder(x_low)
        self.loss2 = self.loss_fc2(recon_x,x)

        return ret

    def recon_loss(self,pred,real=None):
        pred = pred.reshape(-1)
        real_shape = pred.shape
        real = torch.cat(
            [torch.ones(real_shape[0] // 2, ).to(self.device), torch.zeros(real_shape[0] // 2, ).to(self.device)])
        l1 = self.loss_fc1(pred, real)

        if self.data_aug is True:
            loss = self.loss1 + self.loss2
        else:
            loss =  self.loss2

        return loss

    def embed(self,x,adj):
        x = torch.squeeze(x, dim=0)
        x_low = self.low_gcn(x, adj)

        return x_low

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)




class ESGNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,data_aug=0.0,data_aug_source=None,dropout=0.5,device="cuda"):
        super(ESGNN, self).__init__()

        self.data_aug = data_aug

        self.lpf = GCN(input_dim,hidden_dim)
        self.hpf = GCN(input_dim,hidden_dim)

        self.dropout = dropout
        self.device = device

        self.loss_fc1 = InfoNCELoss()
        self.loss_fc2 = InfoNCELoss()


        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        if data_aug > 0:
            self.data_aug = True
            self.lpf_aug = GSL(input_dim,hidden_dim)

    def embed(self, x, adj):
        x_low = self.lpf(x,adj)

        return x_low

    def forward(self,x,adj):
        x = torch.squeeze(x, dim=0)
        if self.data_aug>0:
            lpf_adj = self.lpf_aug(x,adj)
        else:
            lpf_adj = adj

        x_low = self.lpf(x,adj)

        hat_x_low = self.lpf(x,lpf_adj)

        loss = self.loss_fc1(x_low, hat_x_low)

        return loss


    def recon_loss(self, pred, real=None):
        return pred


class InfoNCELoss(nn.Module):


    def __init__(self, temperature=0.1, reduction='mean', negative_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f" {reduction}")
        if negative_mode not in ['all', 'one_way']:
            raise ValueError(f" {negative_mode}")

    def forward(self, z1, z2):

        batch_size, feature_dim = z1.shape

        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        sim_matrix = torch.mm(z1, z2.T) / self.temperature

        labels = torch.arange(batch_size).to(z1.device)

        if self.negative_mode == 'all':
            loss = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)
            loss += F.cross_entropy(sim_matrix.T, labels, reduction=self.reduction)
            return loss / 2

        elif self.negative_mode == 'one_way':

            return F.cross_entropy(sim_matrix, labels, reduction=self.reduction)

    def __repr__(self):
        return (f"{self.__class__.__name__}(temperature={self.temperature}, "
                f"reduction={self.reduction}, negative_mode={self.negative_mode})")