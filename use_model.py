"""Script for using the BiLSTM model in model.py with sequence inputs."""
# pylint: disable=E1101
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import torchvision.models as models
from model import BiLSTM
from transforms import ImageTransforms


########################################################
# DATA LOADER
# ~~~~~~~~~~~

itr = {'train': ImageTransforms(227, 5, 224),
       'test': ImageTransforms(224)}

ttr = TextTransforms()

img_train_tf = lambda x: torchvision.transforms.ToTensor()(itr['train'].random_crop(
                     itr['train'].random_rotation(itr['train'].random_horizontal_flip(
                     itr['train'].resize(x)))))
img_test_val_tf = lambda x: torchvision.transforms.ToTensor()(itr['test'].resize(x))

txt_train_tf = lambda x: ttr.random_delete(ttr.normalize(x))
txt_test_val_tf = lambda x: ttr.normalize(x)

img_transforms = {'train': img_train_tf,
                  'test': img_test_val_tf,
                  'val': img_test_val_tf}

txt_transforms = {'train': txt_train_tf,
                  'test': txt_test_val_tf,
                  'val': txt_test_val_tf}


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
    images = torch.Tensor()
    seq_lens = torch.zeros(len(data))
    lookup_table = []
    count = 0
    for seq_tag, seq_imgs in data:
        seq_lookup = []
        for img in seq_imgs:
            images = torch.cat((images,
                                torchvision.transforms.ToTensor()(img_transform(img))))
            seq_lookup.append(count)
            count += 1
            seq_lens[seq_tag] += 1
        lookup_table.append(seq_lookup)

    # Then, extract their features.
    feats = model(torch.Variable(images))

    # Manually create the padded sequence.
    seqs = torch.Tensor(len(data), max(seq_lens), feats.size()[1])
    for i in range(len(data)):  # Iterate over batch
        for j in range(max(seq_lens)):  # Iterate over sequence
            if j < seq_lens[i]:
                seqs[i, j] = feats[lookup_table[i][j]]
            else:
                seqs[i, j] = torch.zeros(feats.size()[1])

    # In order to be packed, sequences must be ordered from larger to shorter.
    seqs = feats[sorted(range(len(seq_lens)), key=lambda k: seq_lens[k], reverse=True), :]

    # seqs is (batch size, data_dim, max length)
    if batch_first:
        seqs = seqs.permute(0, 2, 1)  # now it is (batch size, max length, data_dim)
    else:
        seqs = seqs.permute(2, 0, 1)  # now it is (max length, max length, data_dim)

    return pack_padded_sequence(seqs, seq_lens, batch_first=batch_first)


def loss_forward(feats, hidden, seq_lens):
    """Compute the forward loss of a batch.

    Args:
        - feats: a batch inputs of the LSTM (padded) - for now, batch first
                (batch_size x max_seq_len x feat_dimension)
        - hidden: outputs of the LSTM for the same batch - for now, batch first
                (batch_size x max_seq_len x hidden_dim)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--multigpu', nargs='*', default=[])
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    batch_first = True

    seq_lens = [5, 5, 3, 1]  # batch size = 4, max length = 5
    batch_size = len(seq_lens)
    input_dim = 100
    hidden_dim = 512
    margin = 0.2
    inception = models.inception_v3(pretrained=True)
    im_transform = ImageTransforms(size=299)
    data = create_packed_seq(inception, im_transform, input_dim, seq_lens)
    data = create_random_packed_seq(input_dim, seq_lens)

    model = BiLSTM(input_dim, hidden_dim, batch_first)
    if args.cuda:
        model = model.cuda()
    if args.multigpu:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=args.multigpu)

    # hidden = model.init_hidden(batch_size)
    # out, hidden = model.forward(data, hidden)
    # out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output: the sequence lengths
    # print out

    img_dir = 'datasets/polyvore/images'
    json_dir = 'datasets/polyvore/label'
    json_files = {'train': 'train_no_dup.json',
                  'test': 'test_no_dup.json',
                  'val': 'valid_no_dup.json'}

    image_datasets = {x: PolyvoreDataset(os.path.join(json_dir, json_files[x]),
                                         img_dir,
                                         img_transform=img_transforms[x],
                                         txt_transform=txt_transforms[x])
                      for x in ['train', 'test', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                batch_size=batch_size, shuffle=True, num_workers=1)
                   for x in ['train', 'test', 'val']}


    optimizer = optim.SGD(model.parameters(), lr=0.2, weight_decay=1e-4)

    numepochs = 200
    tic = time.time()
    for epoch in range(numepochs):  # again, normally you would NOT do 300 epochs, it is toy data
        for batch in dataloaders['train']:
            # Clear gradients, reset hidden state.
            model.zero_grad()
            hidden = model.init_hidden(batch_size)

            # Prepare data
            packed_batch = create_packed_seq(model, img_transforms['train'],
                                             batch, batch_first=batch_first)

            # Forward pass
            # neg_log_likelihood = model.neg_log_likelihood(autograd.Variable(sentence), targets)
            out, hidden = model.forward(packed_batch, hidden)
            out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output: seq lengths

            # loss = model.ContrastiveLoss(margin)
            loss = loss_forward(feats, hidden, seq_lens) + loss_backward(feats, hidden, seq_lens)

            loss.backward()
            optimizer.step()

            writer.add_scalar('data/loss', neg_log_likelihood.data[0], epoch) # data grouping by `slash`
        sys.stdout.write("Epoch %i/%i took %f seconds\r" % (epoch, numepochs, time.time() - tic))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
