"""
Parse Arguments

Date:
    - Jan. 28, 2023
"""
import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser("Interface for Graph2Seq")

    # General
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio.')

    # The Encoder
    parser.add_argument('--gnn', type=string, default='gcn', choices=['gcn', 'bi_gcn'], help='The GNN architecture.')
    parser.add_argument('--gnn_hidden_channels', type=int, default=16, help='Number of GNN hidden channels.')
    parser.add_argument('--gnn_num_layers', type=int, default=6, help='Number of hidden layers for the GNN.')
    # parser.add_argument('--use_gdc', action='store_true', help='Whether to use GDC.')

    # The Decoder
    parser.add_argument('--dec_hidden_state_size', type=int, default=80, help='Number of decoder hidden channels.')

    # General Training
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size.')
    # parser.add_argument('--wandb', action='store_true', help='Track experiment')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv