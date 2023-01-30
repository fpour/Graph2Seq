"""
Graph2Seq: Evaluation procedure

Date:
    - Jan. 28, 2023
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import math
import time
import random

from utils import *
from params import *



def compute_perf_metric(y_true_tensors, y_true_words, y_pred_tensors, y_pred_words):
    """
    compute the performance of a prediction tasks
    :param y_true: the actual target output
    :param y_pred: the predicted output
    :return: the value of a metric measuring the performance of a prediction task
    @TODO: BLEU or some other performance metric should be calculated here
    """
    metric = 0
    return metric


def evaluate(encoder, decoder, data_pairs, output_lang, max_length=MAX_LENGTH):
    """
    evaluate a model on a batch of data pairs
    """
    y_true_tensors, y_true_words, y_pred_tensors, y_pred_words = [], [], [], []

    for data in data_pairs:

        with torch.no_grad():

            input_graph = data[0]
            target_tensor = data[1]

            y_true_tensors.append(target_tensor)
            y_true_words.append(output_lang.index2word[y_true_tensor])

            encoder = encoder.eval()
            decoder = decoder.eval()

            # encoder
            node_encs, pooled_ge = encoder(input_graph.x, input_graph.edge_index, input_graph.edge_attr)

            # decoder
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = pooled_ge

            decoded_words, decoder_outputs = [], []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs=node_encs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                decoder_outputs.append(topi.item())
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            # record the final prediction
            y_pred_tensors.append(decoder_outputs)
            y_pred_words.append(decoded_words)

    metric = compute_perf_metric(y_true_tensors, y_true_words, y_pred_tensors, y_pred_words)

    return metric


