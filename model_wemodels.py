"""Bi-LSTM network adapted for wemodels."""
# pylint: disable=W0221
# pylint: disable=E1101
import torch
import torch.nn as nn
import torch.autograd as autograd
from wemodels.pytorch.wenet import WENet


class BiLSTM(WENet):
    """Bi-LSTM architecture definition.

    Args:
        - input_dim: dimension of the input
        - hidden_dim: dimension of the hidden/output layer
        - batch_first: parameter of the PackedSequence data. Shape of input
        data varies as follows:
            if True: batch_size x max_seq_length x data_dimension
            if False: max_seq_length x batch_size x data_dimension

    """

    def __init__(self, input_dim, hidden_dim, batch_first=False):
        """Create the network."""
        super(BiLSTM, self).__init__({})
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

    def set_dataparallel(self):
        """Set layers to be processed by multiple gpus."""
        self.lstm = nn.DataParallel(self.lstm).cuda()

    def set_fine_tune_level(self, level=1):
        """Abstract method of disabling gradient propagation.

        Disable gradient propagation throught the layers at
        different ``level``.
        """
        print "Fine tune level doesn't make sense in a 1-layer network."
