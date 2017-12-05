"""Script for using the BiLSTM model in model.py with sequence inputs."""
# pylint: disable=E1101
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import torchvision.models as models
from datasets import PolyvoreDataset
from model import BiLSTM
from transforms import ImageTransforms, TextTransforms
from tensorboardX import SummaryWriter
from datasets import collate_seq

WRITER = SummaryWriter()

########################################################
# DATA LOADER
# ~~~~~~~~~~~

itr = {'train': ImageTransforms(305, 5, 299),
       'test': ImageTransforms(299)}

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


def create_packed_seq(model, data, batch_first=False):
    """Create a packed input of sequences for a RNN.

    Args:
        - model: a model to extract features from data.
        - data: a list (with length batch_size) of sequences of images (shaped seq_len x img_dim)
        - [batch_first] : determines the shape of the output.

    Returns:
        - torch PackedSequence (batch_size x max_seq_len x img_dim if batch_first = True,
                                max_seq_len x batch_size x img_dim otherwise)

    """
    # First, get all inputs and keep the information about the sequence they belong to.
    images = torch.Tensor()
    img_data = [i['images'] for i in data]
    seq_lens = torch.zeros(len(img_data)).int()
    lookup_table = []
    count = 0
    for seq_tag, seq_imgs in enumerate(img_data):
        seq_lookup = []
        for img in seq_imgs:
            images = torch.cat((images, img.unsqueeze(0)))
            seq_lookup.append(count)
            count += 1
            seq_lens[seq_tag] += 1
        lookup_table.append(seq_lookup)

    feats, _ = model(autograd.Variable(images).cuda())

    # Manually create the padded sequence.
    seqs = autograd.Variable(torch.zeros((len(img_data), max(seq_lens), feats.size()[1]))).cuda()
    for i in range(len(img_data)):  # Iterate over batch
        for j in range(max(seq_lens)):  # Iterate over sequence
            if j < seq_lens[i]:
                seqs[i, j] = feats[lookup_table[i][j]]
            else:
                seqs[i, j] = torch.zeros(feats.size()[1])

    # In order to be packed, sequences must be ordered from larger to shorter.
    seqs = seqs[sorted(range(len(seq_lens)), key=lambda k: seq_lens[k], reverse=True), :]
    ordered_seq_lens = sorted(seq_lens, reverse=True)

    # seqs is (batch size, max length, data_dim)
    if not batch_first:
        seqs = seqs.permute(1, 0, 2)  # now it is (max length, max length, data_dim)

    return pack_padded_sequence(seqs, ordered_seq_lens, batch_first=batch_first)


def lstm_losses(feats, hidden, seq_lens):
    """Compute the forward and backward loss of a batch.

    Args:
        - feats: a batch inputs of the LSTM (padded) - for now, batch first
                (batch_size x max_seq_len x feat_dimension)
        - hidden: outputs of the LSTM for the same batch - for now, batch first
                (batch_size x max_seq_len x hidden_dim)

    Returns:
        Tuple containing two autograd.Variable values: the forward and backward losses for a batch.

    """
    fw_loss = autograd.Variable(torch.zeros(1))
    bw_loss = autograd.Variable(torch.zeros(1))
    for i, seq_len in enumerate(seq_lens):
        fw_seq_loss = 0
        fw_seq_feats = torch.cat((feats[i, :seq_len, :],
                                  torch.zeros(1, feats.size()[2])))
        fw_seq_hiddens = hidden[i, :seq_len, :hidden.size()[2] // 2]  # Get forward hidden state

        bw_seq_loss = 0
        bw_seq_feats = torch.cat((torch.zeros(1, feats.size()[2]),
                                  feats[i, :seq_len, :]))
        bw_seq_hiddens = hidden[i, :seq_len, hidden.size()[2] // 2:]  # Get backward hidden state

        for j in xrange(seq_len):
            fw_prob = np.exp(torch.dot(fw_seq_hiddens[j], fw_seq_feats[j + 1])) / \
                torch.sum(np.exp(torch.mm(fw_seq_hiddens[j].unsqueeze(0), feats.permute(1, 0))))
            fw_seq_loss += fw_prob
            fw_seq_loss /= -seq_len

            bw_prob = (bw_seq_hiddens[j] * bw_seq_feats[j]) / torch.sum(bw_seq_hiddens[j] * feats)
            bw_seq_loss += bw_prob
            bw_seq_loss /= -seq_len

        fw_loss += fw_seq_loss
        bw_loss += bw_seq_loss

    return (fw_loss/len(seq_lens), bw_loss/len(seq_lens))


def main():
    """Forward sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--multigpu', nargs='*', default=[])
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    batch_first = True

    # seq_lens = [5, 5, 3, 1]  # batch size = 4, max length = 5
    # batch_size = len(seq_lens)
    batch_size = 3
    input_dim = 100
    hidden_dim = 512
    margin = 0.2
    inception_emb = models.inception_v3(pretrained=True)
    inception_emb.fc = nn.Linear(2048, 512)

    model = BiLSTM(input_dim, hidden_dim, batch_first)
    if args.cuda:
        model = model.cuda()
        inception_emb = inception_emb.cuda()
    if args.multigpu:
        model = model.cuda()
        inception_emb = inception_emb.cuda()
        model = nn.DataParallel(model, device_ids=args.multigpu)
        inception_emb = nn.DataParallel(inception_emb, device_ids=args.multigpu)

    # hidden = model.init_hidden(batch_size)
    # out, hidden = model.forward(data, hidden)
    # out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output: the sequence lengths
    # print out

    img_dir = 'data/images'
    json_dir = 'data/label'
    json_files = {'train': 'train_no_dup.json',
                  'test': 'test_no_dup.json',
                  'val': 'valid_no_dup.json'}

    image_datasets = {x: PolyvoreDataset(os.path.join(json_dir, json_files[x]),
                                         img_dir,
                                         img_transform=img_transforms[x],
                                         txt_transform=txt_transforms[x])
                      for x in ['train', 'test', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=1,
                                                  collate_fn=collate_seq)
                   for x in ['train', 'test', 'val']}

    optimizer = optim.SGD(model.parameters(), lr=0.2, weight_decay=1e-4)

    numepochs = 200
    tic = time.time()
    for epoch in range(numepochs):  # again, normally you would NOT do 300 epochs, it is toy data
        for i_batch, batch in enumerate(dataloaders['train']):
            # Clear gradients, reset hidden state.
            model.zero_grad()
            hidden = model.init_hidden(batch_size)

            # Prepare data
            packed_batch = create_packed_seq(inception_emb, batch, batch_first=batch_first)
            import epdb; epdb.set_trace()
            """
            # Forward pass
            # neg_log_likelihood = model.neg_log_likelihood(autograd.Variable(sentence), targets)
            out, hidden = model.forward(packed_batch, hidden)
            out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output: seq lengths

            # loss = model.ContrastiveLoss(margin)
            loss = loss_forward(feats, hidden, seq_lens) + loss_backward(feats, hidden, seq_lens)

            loss.backward()
            optimizer.step()

            WRITER.add_scalar('data/loss', loss.data[0], epoch)
            """
        sys.stdout.write("Epoch %i/%i took %f seconds\r" % (epoch, numepochs, time.time() - tic))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
    WRITER.close()
