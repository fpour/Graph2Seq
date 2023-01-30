
"""
Attention-based Decoder

Date:
    - Jan. 28, 2023
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils import *
from params import MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttnDecoderRNN(nn.Module):
    """
    Adopted from PyTorch Tutorials
    link: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model
    """
    def __init__(self, hidden_channels, output_channels, dropout_p=0.5, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_channels, self.hidden_channels)
        self.attn = nn.Linear(self.hidden_channels * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_channels, self.hidden_channels)
        self.out = nn.Linear(self.hidden_channels, self.output_channels)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights