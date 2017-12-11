"""Script for using the BiLSTM model in model.py with sequence inputs."""
# pylint: disable=E1101
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import torchvision.models as models
from model import BiLSTM
from losses import LSTMLosses, SBContrastiveLoss
from transforms import ImageTransforms, TextTransforms
from datasets import PolyvoreDataset
from datasets import collate_seq
from tensorboardX import SummaryWriter

WRITER = SummaryWriter()

########################################################
# DATA LOADER
# ~~~~~~~~~~~

itr = {'train': ImageTransforms(305, 5, 299, 0.5),
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


def seqs2batch(data):
    """Get a list of images from a list o sequences.

    Args:
        - data: list of sequences (shaped batch_size x seq_len, with seq_len variable).

    Returns:
        - images: list of images.
        - seq_lens: list of sequence lengths.
        - lookup_table: list (shaped batch_size x seq_len, with seq_len variable) containing
            the indices of images in the image list.

    """
    # Get all inputs and keep the information about the sequence they belong to.
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

    return images, seq_lens, lookup_table


def create_packed_seq(model, data, cuda, batch_first):
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
    images, seq_lens, lookup_table = seqs2batch(data)

    if cuda:
        feats, _ = model(autograd.Variable(images).cuda())
    else:
        feats, _ = model(autograd.Variable(images))

    # Manually create the padded sequence.
    if cuda:
        seqs = autograd.Variable(torch.zeros((len(data), max(seq_lens), feats.size()[1]))).cuda()
    else:
        seqs = autograd.Variable(torch.zeros((len(data), max(seq_lens), feats.size()[1])))
    for i in range(len(data)):  # Iterate over batch
        for j in range(max(seq_lens)):  # Iterate over sequence
            if j < seq_lens[i]:
                seqs[i, j] = feats[lookup_table[i][j]]
            else:
                seqs[i, j] = autograd.Variable(torch.zeros(feats.size()[1]))

    # In order to be packed, sequences must be ordered from larger to shorter.
    seqs = seqs[sorted(range(len(seq_lens)), key=lambda k: seq_lens[k], reverse=True), :]
    ordered_seq_lens = sorted(seq_lens, reverse=True)

    # seqs is (batch size, max length, data_dim)
    if not batch_first:
        seqs = seqs.permute(1, 0, 2)  # now it is (max length, max length, data_dim)

    return pack_padded_sequence(seqs, ordered_seq_lens, batch_first=batch_first)




def main():
    """Forward sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--multigpu', nargs='*', default=[])
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    batch_first = True

    batch_size = 2
    input_dim = 512
    hidden_dim = 512
    margin = 0.2
    tic = time.time()
    inception_emb = models.inception_v3(pretrained=True)
    inception_emb.fc = nn.Linear(2048, 512)
    print("inception loading took %.2f secs" % (time.time() - tic))

    tic = time.time()
    model = BiLSTM(input_dim, hidden_dim, batch_first, dropout=0.7)
    if args.cuda:
        model = model.cuda()
        inception_emb = inception_emb.cuda()
    if args.multigpu:
        model = model.cuda()
        inception_emb = inception_emb.cuda()
        model = nn.DataParallel(model, device_ids=args.multigpu)
        inception_emb = nn.DataParallel(inception_emb, device_ids=args.multigpu)
    print("models to cuda took %.2f secs" % (time.time() - tic))

    img_dir = 'data/images'
    json_dir = 'data/label'
    json_files = {'train': 'train_no_dup.json',
                  'test': 'test_no_dup.json',
                  'val': 'valid_no_dup.json'}

    tic = time.time()
    image_datasets = {x: PolyvoreDataset(os.path.join(json_dir, json_files[x]),
                                         img_dir,
                                         img_transform=img_transforms[x],
                                         txt_transform=txt_transforms[x])
                      for x in ['train', 'test', 'val']}
    print("image_datasets took %.2f secs" % (time.time() - tic))

    tic = time.time()
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4,
                                                  collate_fn=collate_seq)
                   for x in ['train', 'test', 'val']}
    print("dataloaders took %.2f secs" % (time.time() - tic))

    tic = time.time()
    optimizer = optim.SGD(model.parameters(), lr=0.2, weight_decay=1e-4)
    criterion = LSTMLosses(batch_first, args.cuda)
    contrastive_criterion = SBContrastiveLoss(margin)
    print("optimizer took %.2f secs" % (time.time() - tic))

    numepochs = 200
    tic = time.time()
    for epoch in range(numepochs):  # again, normally you would NOT do 300 epochs, it is toy data
        for i_batch, batch in enumerate(dataloaders['train']):
            tic_b = time.time()
            # Clear gradients, reset hidden state.
            model.zero_grad()
            hidden = model.init_hidden(batch_size)
            if args.cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())

            # Prepare data
            packed_batch = create_packed_seq(inception_emb, batch,
                                             args.cuda, batch_first=batch_first)
            out, hidden = model.forward(packed_batch, hidden)
            out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output: seq lengths
            fw_loss, bw_loss = criterion(packed_batch, out)
            # cont_loss = contrastive_criterion()
            loss = fw_loss + bw_loss  # + cont_loss

            # loss = SBContrastiveLoss(margin)
            loss.backward()
            optimizer.step()

            WRITER.add_scalar('data/loss', loss.data[0], i_batch)
            WRITER.add_scalar('data/loss_FW', fw_loss.data[0], i_batch)
            WRITER.add_scalar('data/loss_BW', bw_loss.data[0], i_batch)
            print("\033[1;31mEpoch %d - Batch %d (%.2f secs)\033[0m" % (epoch, i_batch, time.time() - tic_b))
            print("\033[1;36m----------------------\033[0m")
            print("\033[1;92mForward loss: %.2f <==> Backward loss: %.2f\033[0m" % (fw_loss.data[0], bw_loss.data[0]))
            print("\033[1;4;92mTOTAL LOSS: %.2f\033[0m" % loss.data[0])
            print("\033[1;36m----------------------\033[0m")

        print("\033[1;30mEpoch %i/%i: %f seconds\033[0m" % (epoch, numepochs, time.time() - tic))


if __name__ == '__main__':
    main()
    WRITER.close()
