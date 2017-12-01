"""Bi-LSTM network."""
import torch
import torch.nn as nn
import torch.autograd as autograd


class Network(nn.Module):
    """Bi-LSTM network."""

    def __init__(self, input_dim, hidden_dim, batch_first=False):
        """Create the network."""
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=batch_first, bidirectional=True)

    def forward(self, data, hidden):
        """Do a forward pass."""
        return self.lstm(data, hidden)

    def init_hidden(self, batch_size):
        """Initialize the hidden state and cell state."""
        return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)))

