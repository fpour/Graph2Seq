"""
Graph2Seq: Training procedure

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
import numpy as np

from utils import *
from params import *
from eval import evaluate

# set random seed
random.seed(RAND_SEED)


def train(data_train, data_val, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          epochs, batch_size, device, output_lang,
          max_length=MAX_LENGTH):
    """
    Training Process with consideration of several epochs and mini-batches
    :param data: the whole training data
    """
    num_instance = len(data_train)
    num_batch = math.ceil(num_instance / batch_size)
    print("INFO: Number of training instances: {}".format(num_instance))
    print("INFO: Number of batches per epoch: {}".format(num_batch))

    for epoch_idx in range(epochs):
        start_time_epoch = time.time()
        batch_losses = []
        train_batch_perf = []
        for b_idx in range(num_batch):
            start_idx = b_idx * batch_size
            end_idx = min(num_instance - 1, start_idx + batch_size)

            training_pairs = data_train[start_idx: end_idx]

            # =============== training
            encoder.train()
            decoder.train()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = 0

            for tr_pair in training_pairs:
                input_graph = tr_pair[0]
                target_tensor = tr_pair[1]

                # encoder
                node_encs, pooled_ge = encoder(input_graph.x, input_graph.edge_index, input_graph.edge_attr)

                # decoder
                target_length = target_tensor.size(0)
                decoder_input = torch.tensor([[SOS_token]], device=device)
                decoder_hidden = pooled_ge

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                if use_teacher_forcing:
                    # Teacher forcing: Feed the target as the next input
                    for di in range(target_length):
                        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                                    encoder_outputs=node_encs)
                        loss += criterion(decoder_output, target_tensor[di])
                        decoder_input = target_tensor[di]  # Teacher forcing
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    for di in range(target_length):
                        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                                    encoder_outputs=node_encs,
                                                                                    max_length=max_length)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze().detach()  # detach from history as input

                        loss += criterion(decoder_output, target_tensor[di])
                        if decoder_input.item() == EOS_token:
                            break

            # back-propagate the loss
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            with torch.no_grad():
                batch_losses.append(loss.item())
                train_perf_metric = evaluate(encoder, decoder, data_train, output_lang, max_length)
                train_batch_perf.append(train_perf_metric)

        # =============== validation
        val_perf_metric = evaluate(encoder, decoder, data_val, output_lang, max_length)

        # ========== Print out some info...
        print("INFO: Epoch: {}, Elapsed time: {}s.".format(epoch_idx, str(time.time() - start_time_epoch)))
        print("INFO: \tEpoch mean loss: {}.".format(np.mean(batch_losses)))
        print("INFO: \tTraining performance: {}.".format(np.mean(train_batch_perf)))
        print("INFO: \tValidation performance: {}.".format(val_perf_metric))
