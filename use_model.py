"""Script for using the model in model.py with sequence inputs."""
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model import Network


def create_packed_seq(data_dim, seq_lens, batch_first=False):
    """Create a packed input of sequences for a RNN."""
    seqs = [autograd.Variable(torch.randn(data_dim, sl)) for sl in seq_lens]
    t_seqs = []
    for seq in seqs:
        if seq.size()[1] < max(seq_lens):
            t_seqs.append(torch.cat((seq, torch.zeros(data_dim, max(seq_lens) - seq.size()[1])), 1))
        else:
            t_seqs.append(seq)

    t_seqs = torch.stack(t_seqs)  # t_seqs is (batch size, data_dim, max length)
    t_seqs = t_seqs.permute(2, 0, 1)  # now it is (batch size, max length, data_dim)

    return pack_padded_sequence(t_seqs, seq_lens, batch_first=batch_first)


def main():
    """Forward sequences."""
    seq_lens = [5, 5, 3, 1]  # batch size = 4, max length = 5
    batch_size = len(seq_lens)
    input_dim = 100
    hidden_dim = 300
    data = create_packed_seq(input_dim, seq_lens)
    batch_first = True

    model = Network(input_dim, hidden_dim, batch_first)
    hidden = model.init_hidden(batch_size)
    out, hidden = model.forward(data, hidden)
    out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output are the sequence lengths
    print out


if __name__ == '__main__':
    main()
