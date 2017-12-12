"""Script for using the BiLSTM model in model.py with sequence inputs."""
# pylint: disable=E1101
# pylint: disable=C0325
# pylint: disable=W0403
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_packed_sequence
import torchvision
from src.utils import seqs2batch, ImageTransforms, TextTransforms
from src.model import FullBiLSTM
from src.losses import LSTMLosses, SBContrastiveLoss
from src.datasets import PolyvoreDataset
from src.datasets import collate_seq
from tensorboardX import SummaryWriter

WRITER = SummaryWriter()
torch.manual_seed(1)

########################################################
# DATA LOADER
# ~~~~~~~~~~~

IMG_TRF = {'train': ImageTransforms(305, 5, 299, 0.5),
           'test': ImageTransforms(299)}

TXT_TRF = TextTransforms()

IMG_TRAIN_TF = lambda x: torchvision.transforms.ToTensor()(IMG_TRF['train'].random_crop(
    IMG_TRF['train'].random_rotation(IMG_TRF['train'].random_horizontal_flip(
        IMG_TRF['train'].resize(x)))))
IMG_TEST_VAL_TF = lambda x: torchvision.transforms.ToTensor()(IMG_TRF['test'].resize(x))

TXT_TRAIN_TF = lambda x: TXT_TRF.random_delete(TXT_TRF.normalize(x))
# pylint: disable=W0108
TXT_TEST_VAL_TF = lambda x: TXT_TRF.normalize(x)
# pylint: enable=W0108

IMG_TRANSFORMS = {'train': IMG_TRAIN_TF,
                  'test': IMG_TEST_VAL_TF,
                  'val': IMG_TEST_VAL_TF}

TXT_TRANSFORMS = {'train': TXT_TRAIN_TF,
                  'test': TXT_TEST_VAL_TF,
                  'val': TXT_TEST_VAL_TF}


def main():
    """Forward sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, help='batch size', default=10)
    parser.add_argument('--cuda', dest='cuda', help='use cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', help="don't use cuda", action='store_false')
    parser.add_argument('--batch_first', dest='batch_first', action='store_true')
    parser.add_argument('--no-batch_first', dest='batch_first', action='store_false')
    parser.add_argument('--multigpu', nargs='*', default=[], help='list of gpus to use')
    parser.set_defaults(cuda=True)
    parser.set_defaults(batch_first=True)
    args = parser.parse_args()

    input_dim = 512
    hidden_dim = 512
    margin = 0.2

    model = FullBiLSTM(input_dim, hidden_dim, args.batch_first, dropout=0.7)
    if args.cuda:
        print("Switching model to gpu")
        model.cuda()
    if args.multigpu:
        print("Switching model to multigpu")
        model.cuda()
        model = nn.DataParallel(model, device_ids=args.multigpu)

    img_dir = 'data/images'
    json_dir = 'data/label'
    json_files = {'train': 'train_no_dup.json',
                  'test': 'test_no_dup.json',
                  'val': 'valid_no_dup.json'}

    dataloaders = {x: torch.utils.data.DataLoader(
        PolyvoreDataset(os.path.join(json_dir, json_files[x]), img_dir,
                        img_transform=IMG_TRANSFORMS[x], txt_transform=TXT_TRANSFORMS[x]),
        batch_size=args.batch_size,
        shuffle=True, num_workers=4,
        collate_fn=collate_seq)
                   for x in ['train', 'test', 'val']}

    optimizer = optim.SGD(model.parameters(), lr=0.2, weight_decay=1e-4)
    criterion = LSTMLosses(args.batch_first, args.cuda)
    contrastive_criterion = SBContrastiveLoss(margin)

    numepochs = 20
    n_iter = 0
    tic = time.time()
    for epoch in range(numepochs):
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
            out, _ = pad_packed_sequence(out, batch_first=args.batch_first)
            fw_loss, bw_loss = criterion(packed_batch, out)
            # cont_loss = contrastive_criterion()
            loss = fw_loss + bw_loss  # + cont_loss

            loss.backward()
            optimizer.step()

            WRITER.add_scalar('data/loss', loss.data[0], n_iter)
            WRITER.add_scalar('data/loss_FW', fw_loss.data[0], n_iter)
            WRITER.add_scalar('data/loss_BW', bw_loss.data[0], n_iter)

            n_iter += 1

            print("\033[1;31mEpoch %d - Batch %d (%.2f secs)\033[0m" %
                  (epoch, i_batch, time.time() - tic_b))
            print("\033[1;36m----------------------\033[0m")
            print("\033[0;92mForward loss: %.2f <==> Backward loss: %.2f\033[0m" %
                  (fw_loss.data[0], bw_loss.data[0]))
            print("\033[0;4;92mTOTAL LOSS: %.2f\033[0m" % loss.data[0])
            print("\033[1;36m----------------------\033[0m")

        print("\033[1;30mEpoch %i/%i: %f seconds\033[0m" % (epoch, numepochs, time.time() - tic))


if __name__ == '__main__':
    main()
    WRITER.close()
