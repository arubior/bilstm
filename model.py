"""Bi-LSTM network."""
# pylint: disable=W0221
# pylint: disable=E1101
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class BiLSTM(nn.Module):
    """Bi-LSTM architecture definition.

    Args:
        - input_dim: dimension of the input
        - hidden_dim: dimension of the hidden/output layer
        - batch_first: parameter of the PackedSequence data

    """

    def __init__(self, input_dim, hidden_dim, batch_first=False, dropout=0):
        """Create the network."""
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                            batch_first=batch_first, bidirectional=True,
                            dropout=dropout)


    def forward(self, data, hidden):
        """Do a forward pass."""
        return self.lstm(data, hidden)

    def init_hidden(self, batch_size):
        """Initialize the hidden state and cell state."""
        return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)))


