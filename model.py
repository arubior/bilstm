"""Bi-LSTM network."""
import torch
import torch.nn as nn
import torch.autograd as autograd


class Network(nn.Module):
    """Bi-LSTM network."""

    def __init__(self, input_dim, hidden_dim):
        """Create the network."""
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

    def forward(self, data, hidden):
        """Do a forward pass."""
        return self.lstm(data, hidden)

    def init_hidden(self):
        """Initialize the hidden state and cell state."""
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

