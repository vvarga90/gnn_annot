
# 
# GNN_annot IJCNN 2021 implementation
#   GatedGCN network customzied for label propagation.
#   @author Viktor Varga, 
#       based on
#           Dwivedi et al. 2020, "Benchmarking Graph Neural Networks",
#           and its implementation: https://github.com/graphdeeplearning/benchmarking-gnns.git
#       AND
#           Bresson et al. 2017, "Residual Gated Graph ConvNets"
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from label_estimation.graphs.gated_gcn_layer import GatedGCNLayer
import config as Config

class GatedGCN(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        in_dim_edge = net_params['in_dim_edge']
        in_dim_node = net_params['in_dim_node']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.device = net_params['device']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.loss_name = net_params['loss_name']

        # h input embedding
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.graph_norm,
                        self.batch_norm, self.residual) for layer_idx in range(n_layers)])
        self.MLP_layer = MLPReadoutClassifier(input_dim=hidden_dim, L=2)

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e, snorm_n, snorm_e)
        
        # output
        g.ndata['h'] = h
        h = self.MLP_layer(h)
        return h
        
    def loss(self, pred, label):
        # pred: (n_samples, 1) fl32
        # label: (n_samples, 1) fl32
        if Config.GNN_BINARY_LOSS == 'bce':
            loss = nn.BCELoss()(pred, label)
        elif Config.GNN_BINARY_LOSS == 'iou_bin':
            loss = BinaryIOULoss()(pred, label)
        else:
            assert False
        return loss

class MLPReadoutClassifier(nn.Module):
    # Based on MLPReadout layer, binary classification output (single probability value)

    def __init__(self, input_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l, input_dim//2**(l+1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear( input_dim//2**L, 1, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        y = torch.sigmoid(y)
        return y

class BinaryIOULoss(nn.Module):

    # Rahman et al. 2016, "Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation"
    #   adapted to binary case, computing mean IoU from both bg and fg IoUs

    def __init__(self):
        super(BinaryIOULoss, self).__init__()

    def forward(self, pred, label):
        # pred: (n_samples, 1) of fl32
        # label: (n_samples, 1) of fl32

        pred = pred.reshape(-1)
        label = label.reshape(-1)
        pred_inv = 1. - pred
        label_inv = 1. - label

        # IoU fg
        inter_fg = pred * label
        union_fg = pred + label - inter_fg
        inter_fg = inter_fg.sum(dim=0)
        union_fg = union_fg.sum(dim=0)
        iou_fg = inter_fg / (union_fg + 1e-8)

        # IoU bg
        inter_bg = pred_inv * label_inv
        union_bg = pred_inv + label_inv - inter_bg
        inter_bg = inter_bg.sum(dim=0)
        union_bg = union_bg.sum(dim=0)
        iou_bg = inter_bg / (union_bg + 1e-8)

        return 1. - 0.5*(iou_fg + iou_bg)
