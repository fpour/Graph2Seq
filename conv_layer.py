
"""
Convolutional Layer:
    GCN Layer Bi-Directional

Adopted from here: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
Date:
    - Jan. 28, 2023
"""

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import math


class GCNConv_BiD(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.even = out_channels % 2 == 0
        out_channels = math.ceil(out_channels / 2)
        self.lin = Linear(in_channels, int(out_channels), bias=False)
        self.bias = Parameter(torch.Tensor(int(out_channels)))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_index_T = torch.stack([col, row], dim=0)

        # Step 4-5: Start propagating messages.
        h_forward = self.propagate(edge_index, x=x, norm=norm)
        h_backward = self.propagate(edge_index_T, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        h_forward += self.bias
        h_backward += self.bias

        # out = torch.cat((h_forward, h_backward), dim=1)
        if self.even:
            out = torch.cat((h_forward, h_backward), dim=1)
        else:
            out = torch.cat((h_forward, h_backward[:, :-1]), dim=1)

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j