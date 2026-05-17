
import os
from tqdm import tqdm
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch_geometric.utils import to_undirected

from Datasets.dataset import init_dataset
from models.DGI import DGI
from models.deepGNN import DeepGNN
from models.VGAE import VGAE,VGAE_torch
from models.HGAE import EncoderLayer,HGAE, Heter_GCN,ESGNN

from models.Classifier import Classifier
from sklearn import svm
from sklearn.metrics import f1_score


def get_argparse():
    parser = argparse.ArgumentParser(description='GNNs.')

    parser.add_argument("--downstream_task",
                        default="node_cls",
                        help="[ node_cls ] or [ graph_cls ]")
    parser.add_argument("--dataset",
                        default="Computers",
                        help="[ Cora, CiteSeer, PubMed, Reddit, Computers, Photo, PPI, Roman, Ratings ]")
    parser.add_argument("--dataset_path",
                        default="./Datasets/",
                        help="Dataset Path. (e.g. ./Datasets/")
    parser.add_argument("--print_data",type=bool,
                        default=True,
                        help="Print Details of Dataset or not.")

    parser.add_argument("--model_name",
                        default="Heter_GCN",
                        help="supervised: [ GCN, GAT ]; unsupervised: [ VGAE, DGI, GraphMAE, Heter_GCN, VGAE_Pyro ]; Ours:CLA，Heter_GCN, ESGNN")
    parser.add_argument("--print_model",type=bool,
                        default=False,
                        help="Print Details of Model or not.")
    parser.add_argument("--k_folds",
                        default=5,
                        help="K-Fold Cross Validation Params.")
    parser.add_argument("--augmentation_ratio",
                        default=0.1,
                        type=float,
                        help="Augmentation Ratio belongs to [0,1]. 0 is no augmentation.")

    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--shuffle",default=True,type=bool)
    parser.add_argument("--learning_rate",default=0.0001,type=float)
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--dropout",default=0.5,type=float)

    parser.add_argument("--num_layers",default=3,type=int,
                        help="Number of gnn layers in model.")
    parser.add_argument("--hidden_dim", default=128, type=int,
                        help="The hidden dim of graph representation.")
    parser.add_argument("--output_dim",default=128,type=int,
                        help="The output dim of graph representation.")
    parser.add_argument('--assign_ratio',default=0.25,type=float,
                        help="The ratio of assignment.")

    parser.add_argument("--device",default="cuda",
                        help="[ cuda ] or [ cpu ].")
    parser.add_argument("--multi_gpu",default=False,
                        help="Multiple GPUs in training.")

    # parser.set_defaults(dataset_path="Datasets/NSE-Tata-Global-Beverages-Limited.csv")

    return parser.parse_args()
def run_supervised(args):
    print("Beginning supervised learning...")

    train_data, val_data, test_data, input_dim,output_cls = init_dataset(dataset_name=args.dataset,
                                                    supervised=True,device=args.device,print_data=args.print_data)

    # dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=args.shuffle)

    hidden_dim = [input_dim//(4* (i+1)) for i in range(args.num_layers-1)]

    model = DeepGNN(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=args.output_dim,class_num=output_cls,
                    num_layers=args.num_layers,task_name=args.downstream_task,dropout=args.dropout,module=args.model_name)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

    model.to(device=args.device)
    # Train
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(test_data[0],test_data[1])
        loss = model.loss(pred=pred,label=test_data[2])
        loss.backward()
        optimizer.step()

    # Val
    preds = model(val_data[0],val_data[1])
    preds = torch.argmax(preds, dim=1)
    acc =  torch.sum(preds == val_data[2]).float() / val_data[2].shape[0]
    print("Val acc: {}".format(acc))

    # Test
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    for i in range(50):
        preds = model(train_data[0], train_data[1])
        preds = torch.argmax(preds, dim=1)
        acc = torch.sum(preds == train_data[2]).float() / train_data[2].shape[0]
        accs.append(acc * 100)
        print(acc)
        tot += acc

    print('Average accuracy:', tot / 50)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())

def run_unsupervised(args):
    print("Beginning unsupervised learning...")
    features, adj, labels, idx_train, idx_val, idx_test,output_cls,dataset = init_dataset(dataset_name=args.dataset,
                                                    supervised=False,device=args.device,print_data=args.print_data)
    data = dataset[0]
    input_dim = features.shape[-1]
    if args.model_name == "VGAE":
        model = VGAE(input_dim=input_dim,hidden_dim=args.hidden_dim,output_dim=args.output_dim,dropout=args.dropout)

    elif args.model_name == "DGI":
        model = DGI(n_in=input_dim, n_h=args.output_dim, activation='prelu')

    elif args.model_name == "CLA":
        model = EncoderLayer(input_dim=input_dim,layer_num=1,hidden_dims=[args.output_dim],dropout=args.dropout)

    elif args.model_name == "Heter_GCN":
        model =   Heter_GCN(input_dim=input_dim,hidden_dim=args.output_dim,data_aug=args.augmentation_ratio,
                            data_aug_source=data,dropout=args.dropout,device=args.device)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.to(device=args.device)

    print("======Train !!======")
    #Train
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()

        pred = model(features, adj)
        # VGAE (pred=pred,real=adj)
        # DGI (pred=pred,real=None)
        # CLA (pred=pred,real=None)

        loss = model.recon_loss(pred=pred,real=None)

        loss.backward()
        optimizer.step()
    #

    embeds = model.embed(features,adj)
    if type(embeds) is tuple:
        embeds = embeds[0]
        train_embs = embeds[0,idx_train]
        val_embs = embeds[0,idx_val]
        test_embs = embeds[0,idx_test]
    else:
        train_embs = embeds[idx_train]
        val_embs = embeds[idx_val]
        test_embs = embeds[idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    tot = tot

    accs = []
    xent = nn.CrossEntropyLoss()
    
    for _ in range(50):
        log = Classifier(args.output_dim, output_cls)

        opt = optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        log.to(args.device)
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            lg_loss = xent(logits, train_lbls)

            lg_loss.backward(retain_graph=True)
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.cpu())
        # print("Acc {}".format(acc))
        tot += acc.cpu()
    std = np.std(accs)
    mean_acc= np.mean(accs)
    print("Acc {}".format(mean_acc))
    print("Std {}".format(std))

def run_KFold_unsupervised(args):



    print("Beginning {}-Fold cross validation unsupervised learning...".format(args.k_folds))

    dataset = init_dataset(dataset_name=args.dataset)
    input_dim =dataset.num_features
    output_cls = dataset.num_classes
    data = dataset[0]
    node_size = data.x.shape[0]
    if data.is_undirected():
        edge_index = data.edge_index
    else:
        print('### Input graph {} is directed'.format(args.dataset))
        edge_index = to_undirected(data.edge_index)
    # data.full_adj_t = SparseTensor.from_edge_index(edge_index).t()

    data.full_adj = torch.sparse_coo_tensor(edge_index, torch.ones(data.num_edges,), torch.Size((data.num_nodes,data.num_nodes)))

    labels = data.y.view(-1)

    data = data.to(args.device)

    if args.model_name == "VGAE":
        model = VGAE(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout)

    elif args.model_name == "DGI":
        model = DGI(n_in=input_dim, n_h=args.output_dim, activation='prelu')

    elif args.model_name == "CLA":
        model = EncoderLayer(input_dim=input_dim, layer_num=1, hidden_dims=[ args.output_dim])

    elif args.model_name == "HGAE":
        model= HGAE(input_dim=input_dim,hidden_dim=[args.hidden_dim, args.output_dim],node_num=data.num_nodes,
                    assign_ratio=args.assign_ratio,dropout=args.dropout,device=args.device)

    elif args.model_name == "Heter_GCN":
        model =   Heter_GCN(input_dim=input_dim,hidden_dim=args.output_dim,data_aug=args.augmentation_ratio,
                            data_aug_source=data,dropout=args.dropout,device=args.device)
    elif args.model_name == "VGAE_torch":
        model = VGAE_torch(input_dim=input_dim,hidden_dim=args.hidden_dim,
                           dropout=args.dropout,device=args.device)
    elif args.model_name == "ESGNN":
        model = ESGNN(input_dim=input_dim,hidden_dim=args.hidden_dim,
                      data_aug=args.augmentation_ratio,data_aug_source=[data.x,edge_index])

    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.to(device=args.device)

    print("======Train !!======")
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()

        pred = model(data.x.unsqueeze(dim=0), data.full_adj)
        # VGAE (pred=pred,real=adj)
        # DGI (pred=pred,real=None)
        loss = model.recon_loss(pred=pred, real=None)



        loss.backward()


        optimizer.step()

    print("Training is OK.")

    embeds = model.embed(data.x, data.full_adj)
    if type(embeds) is tuple:
        embeds = embeds[0]

    if len(embeds.size()) == 3:
        embeds = embeds.squeeze(dim=0)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    accs = []
    f1_mac = []
    f1_mic = []
    embeds = embeds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for train_index, test_index in kf.split(embeds):
        train_X, train_y = embeds[train_index], labels[train_index]
        test_X, test_y = embeds[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')

        acc = np.sum(preds == test_y) / test_y.shape[0]
        accs.append(acc)
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    std = np.std(accs)
    accs = np.mean(accs)

    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac,
          'acc=%.4f' % accs,
          'std=%.4f'%std)


def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)

def run_model():
    args = get_argparse()
    if args.dataset in ["Cora","CiteSeer","PubMed"] and args.model_name in ["GCN", "GAT"]:
        run_supervised(args)

    elif args.dataset in ["Cora","CiteSeer","PubMed"]:
        run_unsupervised(args)

    else:
        run_KFold_unsupervised(args)



if __name__ == '__main__':
    print("OK!")
    run_model()
