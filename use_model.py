"""Script for using the BiLSTM model in model.py with sequence inputs."""
import numpy as np
import torch
import torch.autograd as autograd
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model import BiLSTM
from transforms import ImageTransforms

def create_random_packed_seq(data_dim, seq_lens, batch_first=False):
    """Create a random packed input of sequences for a RNN."""
    seqs = [autograd.Variable(torch.randn(data_dim, sl)) for sl in seq_lens]
    t_seqs = []
    for seq in seqs:
        if seq.size()[1] < max(seq_lens):
            t_seqs.append(torch.cat((seq, torch.zeros(data_dim, max(seq_lens) - seq.size()[1])), 1))
        else:
            t_seqs.append(seq)

    t_seqs = torch.stack(t_seqs)  # t_seqs is (batch size, data_dim, max length)
    if batch_first:
        t_seqs = t_seqs.permute(0, 2, 1)  # now it is (batch size, max length, data_dim)
    else:
        t_seqs = t_seqs.permute(2, 0, 1)  # now it is (max length, max length, data_dim)

    return pack_padded_sequence(t_seqs, seq_lens, batch_first=batch_first)


def get_feats(model, inputs):
    """Extract features from inputs.

    Args:
        - model: CNN model.
        - inputs: tensor of N images.

    Returns:
        - tensor of N features.

    """
    return model(inputs)


def create_packed_seq(model, img_transform, data, batch_first=False):
    """Create a packed input of sequences for a RNN.

    Args:
        - model: a model to extract features from data.
        - img_transform: a function to preprocess images.
        - data: a list (with length batch_size) of sequence CNN images (shaped seq_len x img_dim)
        - [batch_first] : determines the shape of the output.

    Returns:
        - torch PackedSequence (batch_size x max_seq_len x img_dim if batch_first = True,
                                max_seq_len x batch_size x img_dim otherwise)

    """
    # First, get all inputs and keep the information about the sequence they belong to.
    images = []
    seq_tags = torch.Tensor()
    for seq_tag, seq_imgs in data:
        for img in seq_imgs:
            inp = img_transform(img)
            inputs = torch.cat((inputs, inp))
            seq_tags.append(seq_tag)

    # Then, extract their features.
    feats = model(torch.Variable(inputs))



    seqs = [autograd.Variable(torch.randn(data_dim, sl)) for sl in seq_lens]
    t_seqs = []
    for seq in seqs:
        if seq.size()[1] < max(seq_lens):
            t_seqs.append(torch.cat((seq, torch.zeros(data_dim, max(seq_lens) - seq.size()[1])), 1))
        else:
            t_seqs.append(seq)

    t_seqs = torch.stack(t_seqs)  # t_seqs is (batch size, data_dim, max length)
    if batch_first:
        t_seqs = t_seqs.permute(0, 2, 1)  # now it is (batch size, max length, data_dim)
    else:
        t_seqs = t_seqs.permute(2, 0, 1)  # now it is (max length, max length, data_dim)

    return pack_padded_sequence(t_seqs, seq_lens, batch_first=batch_first)


def loss_forward(feats, hidden, seq_lens):
    """Compute the forward loss of a batch.

    Args:
        - feats: a batch inputs of the LSTM (padded) - for now, batch first (batch_size x max_seq_len x feat_dimension)
        - hidden: outputs of the LSTM for the same batch - for now, batch first (batch_size x max_seq_len x hidden_dim)

    Returns:
        - autograd.Variable containing the forward loss value for a batch.

    """
    n_seqs = len(seq_lens)
    loss = autograd.Variable(torch.zeros(1))
    for i, seq_len in enumerate(seq_lens):
        seq_loss = 0
        seq_feats = torch.cat((feats[i, :seq_len, :],
                               torch.zeros(1, feats.size()[2])))
        seq_hiddens = hidden[i, :seq_len, :hidden.size()[2] // 2]  # Get the forward hidden state
        for j in xrange(seq_len):
            prob = np.exp(torch.dot(seq_hiddens[j], seq_feats[j + 1])) / \
                torch.sum(np.exp(torch.mm(seq_hiddens[j].unsqueeze(0), feats.permute(1, 0))))
            seq_loss += prob
            seq_loss /= -seq_len
        loss += seq_loss

    return loss/n_seqs


def loss_backward(feats, hidden, seq_lens):
    """Compute the backward loss of a batch.

    Args:
        - feats: a batch inputs of the LSTM (padded) - for now, batch first (batch_size x max_seq_len x feat_dimension)
        - hidden: outputs of the LSTM for the same batch - for now, batch first (batch_size x max_seq_len x hidden_dim)

    Returns:
        - autograd.Variable containing the backward loss value for a batch.

    """
    n_seqs = len(seq_lens)
    loss = autograd.Variable(torch.zeros(1))
    for i, seq_len in enumerate(seq_lens):
        seq_loss = 0
        seq_feats = torch.cat((torch.zeros(1, feats.size()[2]),
                               feats[i, :seq_len, :]))
        seq_hiddens = hidden[i, :seq_len, hidden.size()[2] // 2:]  # Get the backward hidden state
        for j in xrange(seq_len):
            prob = (seq_hiddens[j] * seq_feats[j]) / torch.sum(seq_hiddens[j] * feats)
            seq_loss += prob
            seq_loss /= -seq_len
        loss += seq_loss

    return loss/n_seqs


def main():
    """Forward sequences."""
    seq_lens = [5, 5, 3, 1]  # batch size = 4, max length = 5
    batch_size = len(seq_lens)
    input_dim = 100
    hidden_dim = 512
    inception = models.inception_v3(pretrained=True)
    data = create_packed_seq(inception, input_dim, seq_lens)
    data = create_random_packed_seq(input_dim, seq_lens)
    batch_first = True

    model = BiLSTM(input_dim, hidden_dim, batch_first)
    hidden = model.init_hidden(batch_size)
    out, hidden = model.forward(data, hidden)
    out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output: the sequence lengths
    print out


if __name__ == '__main__':
    main()
