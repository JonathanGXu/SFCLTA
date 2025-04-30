# Node Classification (Node-Level Task)

###  Preliminary

**Node Classification Task.**

Given a graph $G = (V, E)$ where $V$ is the node set and $E$ is the edge set, the node classification aims to category the nodes into $k$ classes, which is defined as follows.

$$ Y = \phi (V, E, \Theta), $$
where $\phi(\cdot)$ is the classification model, $Y = \{ y_{1}, y_{2}, ..., y_{k} \}$ is the label set with $k$ classes, and the $\Theta$ is the hyper-parameter set. 

### Set Up

We followed the experimental set-up by GraphMAE 
![GraphMAE Set-Up](Images/Node_cls_setup.png "Node_cls_setup")

**(1) Supervised Learning.** Several supervised learning methods GCN or GAT train on the training datasets, and test on the testing datasets. For example, previous experimental set-up in [DGI](https://github.com/PetarV-/DGI) has proposed a datasets split-up (e.g. Cora, CiteSeer, PubMed) for supervision.

**(2) Un-supervised Learning.** First, we train the graph encoder models to generate node representations under un-supervision. Then, we adopt the linear module as the node classifier (i.e. the subsequent module to classify the nodes), and freeze the parameters of the graph encoder models.

### Related Work 

    Note that, some papers do not release their codes for reproduction. We only focus the papers with their codes.

___

##### Supervised Baselines
**(1) GCN (ICLR-17)**
    paper: [GCN_paper](https://arxiv.org/pdf/1609.02907).
    code: [GCN_code](https://github.com/tkipf/gcn).

**(2) GAT (ICLR-18)**
    paper: [GAT_paper](https://arxiv.org/abs/1710.10903).
    code: [GAT_code](https://github.com/PetarV-/GAT).

___

##### Unsupervised Baselines
**(1) VGAE (NIPS-16)**
    paper: [VGAE_paper](https://arxiv.org/abs/1611.07308).
    code: [VGAE_code](https://github.com/tkipf/gae).

**(2) DGI (ICLR-19)**
    paper: [DGI_paper](https://openreview.net/forum?id=rklz9iAcKQ).
    code: [DGI_code](https://github.com/PetarV-/DGI).

**(3) GraphMAE (SIGKDD-22)**
    paper: [GraphMAE_paper](https://arxiv.org/abs/2205.10803).
    code: [GraphMAE_code](https://github.com/THUDM/GraphMAE).

**(4)  GraphMAE2 (WWW-23)**
    paper: [GraphMAE2_paper](https://arxiv.org/abs/2304.04779).
    code: [GraphMAE2_code](https://github.com/THUDM/GraphMAE2).

**（5） S2GAE (WSDM-23)**
    paper: [S2GAE_paper](https://dl.acm.org/doi/abs/10.1145/3539597.3570404).
    code: [S2GAE_code](https://github.com/qiaoyu-tan/S2GAE).

**(5) 24/25**

**(6) 24/25**

**(7) 25**

___

