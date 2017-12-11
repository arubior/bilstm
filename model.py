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

    def __init__(self, input_dim, hidden_dim, batch_first=False):
        """Create the network."""
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                            batch_first=batch_first, bidirectional=True)

    def forward(self, data, hidden):
        """Do a forward pass."""
        return self.lstm(data, hidden)

    def init_hidden(self, batch_size):
        """Initialize the hidden state and cell state."""
        return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)))


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Extracted from: hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin -
                                                                      euclidean_distance,
                                                                      min=0.0), 2))


        return loss_contrastive
