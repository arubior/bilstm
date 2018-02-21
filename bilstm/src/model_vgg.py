"""Full Bi-LSTM network."""
# pylint: disable=W0221
# pylint: disable=E1101
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
from torchvision.models.vgg import model_urls
from torchvision import transforms

model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')


class FullBiLSTM(nn.Module):
    """Bi-LSTM architecture definition.

    Args:
        - input_dim: (int) dimension of the input.
        - hidden_dim: (int) dimension of the hidden/output layer.
        - vocab_size: (int) size of the text vocabulary
        - [batch_first]: (bool) parameter of the PackedSequence data.
        - [dropout]: (float) dropout value for LSTM.
        - [freeze]: (bool) whether to freeze or not the CNN part.

    """

    # Disable too-many-arguments.
    # pylint: disable=R0913
    def __init__(self, input_dim, hidden_dim, vocab_size,
                 batch_first=False, dropout=0, freeze=False):
        """Create the network."""
        super(FullBiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.vocab_size = vocab_size
        self.textn = nn.Linear(vocab_size, input_dim)
        self.cnn = models.vgg16_bn(pretrained=True)
        if freeze:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.cnn.classifier._modules['6'] = nn.Linear(4096, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                            batch_first=self.batch_first, bidirectional=True,
                            dropout=dropout)

    # Disable too-many-arguments and too-many-locals.
    # pylint: disable=R0913
    # pylint: disable=R0914
    def forward(self, images, seq_lens, im_lookup_table, txt_lookup_table, hidden, texts):
        """Do a forward pass.

        The forward pass implies:
            - A normal forward of the images through a CNN.
            - A pass of the texts through a text embedding.
            - Transforming the image features to a pytorch PackedSequence.
            - Doing the forward pass through the LSTM.

        Args:
            - images: autograd Variable with the images of the batch.
            - seq_lens: torch tensor with a list of the sequence lengths.
            - im_lookup_table: list of lists with indices of the images.
            - txt_lookup_table: list of lists with indices of the words in the texts.
            - hidden: hidden variables for the LSTM.
            - texts: autograd Variable with a list of one-hot encoding matrices for
                texts (M words x N vocab_size).

        Returns:
            - features extracted from the CNN (PackedSequence).
            - (im_feats, txt_feats): network features for images and texts in the batch.
            - (out, hidden): outputs and hidden states of the LSTM.

        """
        # Get image features:
        im_feats = self.cnn(images)
        # L2 norm as here: https://github.com/xthan/polyvore/blob/master/polyvore/polyvore_model_bi.py#L328
        im_feats = torch.nn.functional.normalize(im_feats, p=2, dim=1)

        # Get word features:
        word_feats = self.textn(texts)
        # L2 norm as here: https://github.com/xthan/polyvore/blob/master/polyvore/polyvore_model_bi.py#L330
        # word_feats = torch.nn.functional.normalize(word_feats, p=2, dim=1)

        # Mean of word descriptors for each text:
        # txt_feats = [torch.mean(word_feats[i[0]:i[-1] + 1], 0)
        #              for batch in txt_lookup_table for i in batch]
        txt_feats_matrix = autograd.Variable(torch.zeros(len(images), word_feats.size()[1]))
        if im_feats[0].is_cuda:
            txt_feats_matrix = txt_feats_matrix.cuda()
        table_idxs = [y for x in txt_lookup_table for y in x]
        for i in range(txt_feats_matrix.size(0)):
            # txt_feats_matrix[i, :] = feat
            txt_feats_matrix[i, ] = torch.mean(word_feats[table_idxs[i]], 0)
        txt_feats_matrix = torch.nn.functional.normalize(txt_feats_matrix, p=2, dim=1)

        # Pack the sequences:
        packed_feats = self.create_packed_seq(im_feats, seq_lens, im_lookup_table)
        # Forward the sequence through the LSTM:
        return packed_feats, (im_feats, txt_feats_matrix), self.lstm(packed_feats, hidden)
    # pylint: enable=R0913
    # pylint: enable=R0914

    def im_forward(self, images, seq_lens, im_lookup_table, hidden):
        """Do a forward pass only with images.

        The image forward pass implies:
            - A normal forward of the images through a CNN.
            - Transforming the image features to a pytorch PackedSequence.
            - Doing the forward pass through the LSTM.

        Args:
            - images: autograd Variable with the images of the batch.
            - seq_lens: torch tensor with a list of the sequence lengths.
            - im_lookup_table: list of lists with indices of the images.
            - hidden: hidden variables for the LSTM.

        Returns:
            - (out, hidden): outputs and hidden states of the LSTM.

        """
        # Get image features:
        im_feats, _ = self.cnn(images)
        # Pack the sequences:
        packed_feats = self.create_packed_seq(im_feats, seq_lens, im_lookup_table)
        # Forward the sequence through the LSTM:
        return self.lstm(packed_feats, hidden)

    def init_hidden(self, batch_size):
        """Initialize the hidden state and cell state."""
        return (autograd.Variable(torch.rand(2, batch_size, self.hidden_dim) * 2 * 0.08),  # https://github.com/xthan/polyvore/blob/master/polyvore/polyvore_model_bi.py#L55
                autograd.Variable(torch.rand(2, batch_size, self.hidden_dim) * 2 * 0.08))
        """
        return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)),
        autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)))
        """

    def create_packed_seq(self, feats, seq_lens, im_lookup_table):
        """Create a packed input of sequences for a RNN.

        Args:
            - feats: torch.Tensor with data features (N imgs x feat_dim).
            - seq_lens: sequence lengths.
            - im_lookup_table: list of image indices from seqs2batch.
            - data: list (with length batch_size) of sequences of images (shaped seq_len x img_dim).

        Returns:
            - torch PackedSequence (batch_size x max_seq_len x img_dim if batch_first = True,
                                    max_seq_len x batch_size x img_dim otherwise).

        """
        # Manually create the padded sequence.
        if feats.is_cuda:
            seqs = autograd.Variable(torch.zeros((len(seq_lens), max(seq_lens),
                                                  feats.size()[1]))).cuda()
        else:
            seqs = autograd.Variable(torch.zeros((len(seq_lens), max(seq_lens), feats.size()[1])))

        for i, seq_len in enumerate(seq_lens):  # Iterate over batch
            for j in range(max(seq_lens)):  # Iterate over sequence
                if j < seq_len:
                    seqs[i, j] = feats[im_lookup_table[i][j]]
                else:
                    seqs[i, j] = autograd.Variable(torch.zeros(feats.size()[1]))

        # In order to be packed, sequences must be ordered from larger to shorter.
        seqs = seqs[sorted(range(len(seq_lens)), key=lambda k: seq_lens[k], reverse=True), :]
        ordered_seq_lens = sorted(seq_lens, reverse=True)

        # seqs is (batch size, max length, data_dim)
        if not self.batch_first:
            seqs = seqs.permute(1, 0, 2)  # now it is (max length, max length, data_dim)

        return pack_padded_sequence(seqs, ordered_seq_lens, batch_first=self.batch_first)
