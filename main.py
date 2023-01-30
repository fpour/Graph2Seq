"""
Graph2Seq Model

Date:
    - Jan. 28, 2023
"""

import torch
import torch.nn.functional as F
import numpy as np

import parser
from params import *
from graph_encoders import GNN
from attention_decoder import AttnDecoderRNN
from train import train
from eval import evaluate
from utils import *

import time





def main():
    """
    The main flow of Graph2Seq
    """
    args, sys_argv = get_args()

    # ====================================================================================
    # ===================================== Set Arguments
    # === General
    DROPOUT = args.dropout
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # === GNN
    GNN_MODE = args.gnn
    GNN_HID_CH = args.gnn_hidden_channels
    # GNN_GDC = args.use_gdc
    GNN_NUM_LAYERS = args.gnn_num_layers

    # === Attention-Decoder
    DEC_HID_STATE_SIZE = args.dec_hidden_state_size


    # ====================================================================================
    # ===================================== Data Loading & Processing
    """
    Assumption on Data Processing:
        1. Data contains three splits (i.e., train, validation, & test) covering the whole datasets
        2. Each data split (i.e., train, validation, & test) consists of a list of pairs (graph, sentence).
           The length of the list specifies the number of instances in the split. 
           Each instance corresponds to one 'graph (of a SQL query)' that maps to one 'interpretation'.
        3. There should be a 'graph_lang' (~input_lang) that maps each node of the graph to its node id.
        4. There should be a 'output_lang' that maps each word in a sentence to its corresponding id.
        5. 'graph': it includes the input graph in the convention format accepted by PyTorch Geometric.
                    The node ids comes from 'graph_lang'.
        6. 'sentence': this is a English sentence. The word ids come from 'output_lang'.
    """

    data_train, data_val, data_test = [], [], []
    input_lang = Lang('graph')  # Input language helps in mapping of graph nodes to their ids
    output_lang = Lang('snt')  # output language helps in mapping of words to their ids

    # ====================================================================================
    # ===================================== Model Definition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # graph encoder
    graph_encoder = GNN(num_node_init_feats, GNN_HID_CH, DEC_HID_STATE_SIZE,
                        num_layers=GNN_NUM_LAYERS, dropout=DROPOUT,
                        gnn_mode=GNN_MODE)
    graph_encoder = graph_encoder.to(device)
    enc_optimizer = torch.optim.Adam(graph_encoder.parameters(), lr=LR)

    # sequence decoder
    dec_out_ch = output_lang.n_words  # @TODO: should be set based on the available data
    attn_decoder = AttnDecoderRNN(DEC_HID_STATE_SIZE, dec_out_ch, dropout_p=DROPOUT)
    attn_decoder = attn_decoder.to(device)
    dec_optimizer = torch.optim.Adam(attn_decoder.parameters(), lr=LR)

    # criterion
    criterion = nn.NLLLoss()

    # ====================================================================================
    # ===================================== Train & Validation
    train(data_train, data_val, graph_encoder, attn_decoder, enc_optimizer, dec_optimizer, criterion,
          EPOCHS, BATCH_SIZE, device, output_lang,
          max_length=MAX_LENGTH)


    # ====================================================================================
    # ===================================== Test
    start_time_test = time.time()
    test_perf_metric = evaluate(encoder, decoder, data_test, output_lang, max_length=MAX_LENGTH)
    print("INFO: Final Test performance: {}".format(test_perf_metric))
    print("INFO: \tTest elapsed time: {}s.".format(str(time.time() - start_time_test)))





if __name__ == '__main__':
    main()