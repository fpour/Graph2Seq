"""
Graph Encoder of Graph2Seq Architecture

Date:
    - Jan. 28, 2023
"""
import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log

from conv_layer import GCNConv_BiD



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5,
                 use_gdc=False, gnn_mode='gcn'):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        conv_layers_list = []
        self.PGE = torch.nn.Linear(out_channels, out_channels)  # Pooling-based Graph Embedding

        for i in range(num_layers):
            # decide which layer
            if i == 0:
                num_in_ch = in_channels
                num_out_ch = hidden_channels
            elif i == num_layers - 1:
                num_in_ch = hidden_channels
                num_out_ch = out_channels
            else:
                num_in_ch = hidden_channels
                num_out_ch = hidden_channels

            # decide which layer type
            if gnn_mode == 'gcn':
                layer = GCNConv(num_in_ch, num_out_ch, cached=True, normalize=not use_gdc)
            elif gnn_mode == 'bi_gcn':
                layer = GCNConv_BiD(num_in_ch, num_out_ch)
            else:
                raise ValueError("Undefined GNN model! (Not implemented yet!!!)")

            conv_layers_list.append(layer)
        self.conv_layers = torch.nn.ModuleList(conv_layers_list)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv_layer in enumerate(self.conv_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == self.num_layers - 1:
                x = conv_layer(x, edge_index, edge_weight)
            else:
                x = conv_layer(x, edge_index, edge_weight).relu()
        pooled_ge = torch.max(self.PGE(x), dim=0)
        return x, pooled_ge




def train_gnn(model, optimizer, data, epochs):
    """
    train a GNN object
    NOTE: this is for local testing
    """
    best_val_acc = 0
    for epoch in range(epochs):
        # train
        model.train()
        optimizer.zero_grad()
        out, pooled_ge = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # evaluate
        train_acc, val_acc = eval_gnn(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc)



def eval_gnn(model, data):
    with torch.no_grad():
        model.eval()
        pred, pooled_ge = model(data.x, data.edge_index, data.edge_attr)
        pred = pred.argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def test_gnn(model, data):
    # final test
    with torch.no_grad():
        model.eval()
        pred, pooled_ge = model(data.x, data.edge_index, data.edge_attr)
        pred = pred.argmax(dim=-1)
        test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

    # print("DEBUG: Test: pooled_ge:", pooled_ge)

    return test_acc



def main():
    """
    To test the functionality of graph encoder
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--gnn', type=str, default='gcn', choices=['gcn', 'bi_gcn'], help='The GNN architecture.')
    parser.add_argument('--gnn_hidden_channels', type=int, default=80, help='Number of GNN hidden channels.')
    parser.add_argument('--gnn_num_layers', type=int, default=7, help='Number of hidden layers for the GNN.')
    args = parser.parse_args()
    print("DEBUG: args:\n", args)

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    # if args.use_gdc:
    #     transform = T.GDC(
    #         self_loop_weight=1,
    #         normalization_in='sym',
    #         normalization_out='col',
    #         diffusion_kwargs=dict(method='ppr', alpha=0.05),
    #         sparsification_kwargs=dict(method='topk', k=128, dim=0),
    #         exact=True,
    #     )
    #     data = transform(data)

    # define model
    model = GNN(dataset.num_features, args.gnn_hidden_channels, dataset.num_classes, num_layers=args.gnn_num_layers,
                dropout=args.dropout, use_gdc=args.use_gdc, gnn_mode=args.gnn)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train & validation
    train_gnn(model, optimizer, data, args.epochs)

    # test
    test_acc = test_gnn(model, data)
    log(Final_Test_ACC=test_acc)




if __name__ == '__main__':
    main()