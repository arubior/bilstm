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
from utils import seqs2batch, ImageTransforms, TextTransforms
from model import FullBiLSTM
from losses import LSTMLosses, SBContrastiveLoss
from datasets import PolyvoreDataset
from datasets import collate_seq
from tensorboardX import SummaryWriter

WRITER = SummaryWriter()
torch.manual_seed(1)

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


def main():
    """Forward sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--multigpu', nargs='*', default=[])
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    batch_first = True

    batch_size = 10
    input_dim = 512
    hidden_dim = 512
    margin = 0.2
    tic = time.time()
    # inception_emb = models.inception_v3(pretrained=True)
    # inception_emb.fc = nn.Linear(2048, 512)
    # print("inception loading took %.2f secs" % (time.time() - tic))

    tic = time.time()
    # model = BiLSTM(input_dim, hidden_dim, batch_first, dropout=0.7)
    model = FullBiLSTM(input_dim, hidden_dim, batch_first, dropout=0.7)
    if args.cuda:
        model.cuda()
        # inception_emb.cuda()
    if args.multigpu:
        model.cuda()
        # inception_emb.cuda()
        model = nn.DataParallel(model, device_ids=args.multigpu)
        # inception_emb = nn.DataParallel(inception_emb, device_ids=args.multigpu)
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
            hidden = model.init_hidden(len(batch))

            # Get a list of images and texts from sequences:
            images, seq_lens, lookup_table = seqs2batch(batch)

            if args.cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
                images = autograd.Variable(images).cuda()
            else:
                images = autograd.Variable(images)

            packed_batch, (out, hidden) = model.forward(images, seq_lens, lookup_table, hidden)
            out, _ = pad_packed_sequence(out, batch_first=batch_first)  # 2nd output: seq lengths
            fw_loss, bw_loss = criterion(packed_batch, out)
            # cont_loss = contrastive_criterion()
            loss = fw_loss + bw_loss  # + cont_loss

            loss.backward()
            optimizer.step()

            WRITER.add_scalar('data/loss', loss.data[0], (epoch + 1) * i_batch)
            WRITER.add_scalar('data/loss_FW', fw_loss.data[0], (epoch + 1) * i_batch)
            WRITER.add_scalar('data/loss_BW', bw_loss.data[0], (epoch + 1) * i_batch)
            print("\033[1;31mEpoch %d - Batch %d (%.2f secs)\033[0m" % (epoch, i_batch, time.time() - tic_b))
            print("\033[1;36m----------------------\033[0m")
            print("\033[0;92mForward loss: %.2f <==> Backward loss: %.2f\033[0m" % (fw_loss.data[0], bw_loss.data[0]))
            print("\033[0;4;92mTOTAL LOSS: %.2f\033[0m" % loss.data[0])
            print("\033[1;36m----------------------\033[0m")

        print("\033[1;30mEpoch %i/%i: %f seconds\033[0m" % (epoch, numepochs, time.time() - tic))


if __name__ == '__main__':
    main()
    WRITER.close()
