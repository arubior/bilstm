"""Script for using the BiLSTM model in model.py with sequence inputs."""
# pylint: disable=E1101
# pylint: disable=C0325
# pylint: disable=W0403
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_packed_sequence
import torchvision
from src.utils import seqs2batch, ImageTransforms, TextTransforms, create_vocab
from src.model import FullBiLSTM
from src.losses import LSTMLosses, SBContrastiveLoss
from src.datasets import PolyvoreDataset
from src.datasets import collate_seq
from src.evaluation import Evaluation
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

GRADS = {}
def save_grad(name):
    """Save gradient value. To be called from "register_hook"."""
    def hook(grad):
        GRADS[name] = grad
    return hook

def config(net_params, data_params, opt_params, batch_params, cuda_params):
    """Get parameters to configure the experiment and prepare the needed variables.

    Args:
        - net_params: list containing:
            input_dimension for LSTM (int)
            output_dimension for LSTM (int)
            margin for contrastive loss (float)
            size of the vocabulary (int)
            load_path for loading weights (str) (None by default)
        - data_params: list containing:
            path to the directory where images are (string)
            path to the directory wher jsons are (string)
            names of the train, test and validation jsons (dictionary)
        - opt_params: list containing:
            learning rate value (float)
            weight decay value (float)
        - batch_params: list containing:
            batch_size (int)
            batch_first (bool) for the LSTM sequences
        - cuda_params:
            cuda (bool): whether to use GPU or not
            multigpu (list of int): indices of GPUs to use

    Returns:
        - model: pytorch model to train
        - dataloaders: data iterators
        - scheduler: scheduler of the optimizer function to train
        - criterion: loss equation to train

    """
    input_dim, hidden_dim, margin, vocab_size, load_path = net_params
    img_dir, json_dir, json_files = data_params
    learning_rate, weight_decay = opt_params
    batch_size, batch_first = batch_params
    cuda, multigpu = cuda_params

    model = FullBiLSTM(input_dim, hidden_dim, vocab_size, batch_first, dropout=0.7)
    if load_path is not None:
        print("Loading weights from %s" % load_path)
        model.load_state_dict(torch.load(load_path))
    if cuda:
        print("Switching model to gpu")
        model.cuda()
    if multigpu:
        print("Switching model to multigpu")
        model.cuda()
        model = nn.DataParallel(model, device_ids=multigpu)

    dataloaders = {x: torch.utils.data.DataLoader(
        PolyvoreDataset(os.path.join(json_dir, json_files[x]), img_dir,
                        img_transform=IMG_TRANSFORMS[x], txt_transform=TXT_TRANSFORMS[x]),
        batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=collate_seq)
                   for x in ['train', 'test', 'val']}

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = LSTMLosses(batch_first, cuda)
    contrastive_criterion = SBContrastiveLoss(margin)

    return model, dataloaders, optimizer, criterion, contrastive_criterion


def train(train_params, dataloaders, cuda, batch_first, epoch_params):
    """Train the model.

    """
    model, criterion, contrastive_criterion, optimizer, scheduler, vocab = train_params
    numepochs, nsave, save_path = epoch_params

    n_iter = 0
    tic = time.time()
    torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
    for epoch in range(numepochs):
        print("Epoch %d - lr = %.4f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        for batch in dataloaders['train']:

            # tic_b = time.time()
            # Clear gradients, reset hidden state.
            model.zero_grad()
            hidden = model.init_hidden(len(batch))

            # Get a list of images and texts from sequences:
            images, texts, seq_lens, im_lookup_table, txt_lookup_table = seqs2batch(batch, vocab)

            if cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
                images = autograd.Variable(images).cuda()
                texts = autograd.Variable(texts).cuda()
            else:
                images = autograd.Variable(images)
                texts = autograd.Variable(texts)

            packed_batch, (im_feats, txt_feats), (out, hidden) = model.forward(images,
                                                                               seq_lens,
                                                                               im_lookup_table,
                                                                               txt_lookup_table,
                                                                               hidden,
                                                                               texts)
            out, _ = pad_packed_sequence(out, batch_first=batch_first)
            fw_loss, bw_loss = criterion(packed_batch, out)
            cont_loss = contrastive_criterion(im_feats, txt_feats)
            lstm_loss = fw_loss + bw_loss
            loss = lstm_loss# + cont_loss

            if np.isnan(loss.cpu().data.numpy()) or lstm_loss.cpu().data[0] < 0:
                import epdb; epdb.set_trace()

            im_feats.register_hook(save_grad('im_feats'))
            loss.backward()
            # Gradient clipping
            optimizer.step()

            print("\033[1;31mloss %d: %.3f\033[0m" % (n_iter, loss.data[0]))
            print("\033[1;31mcont_loss %d: %.3f\033[0m" % (n_iter, cont_loss.data[0]))
            print("\033[1;34mLSTM loss: %.3f ||| Contr. loss: %.3f\033[0m" % (loss.data[0] -
                                                                              cont_loss.data[0],
                                                                              cont_loss.data[0]))
            print("\033[1;31mBatch size = %d\033[0m" % len(batch))
            for i, b in enumerate(batch):
                print("Seq %d - len = %d" % (i, len(b['texts'])))

            dists = torch.sum(1 - F.cosine_similarity(im_feats, txt_feats))/im_feats.size()[0]
            print("\033[1;31n_iter: %d: mdists: %.3f\033[0m" % (n_iter, dists.data[0]))

            WRITER.add_scalar('data/loss', loss.data[0], n_iter)
            WRITER.add_scalar('data/lstm_loss', lstm_loss.data[0], n_iter)
            WRITER.add_scalar('data/cont_loss', cont_loss.data[0], n_iter)
            WRITER.add_scalar('data/pos_dists', dists.data[0], n_iter)
            WRITER.add_scalar('grads/im_feats', torch.mean(torch.cat([torch.norm(t)
                                                                      for t in GRADS['im_feats']])),
                              n_iter)

            # print("\033[1;31mBatch %d took %.2f secs\033[0m" % (n_iter, time.time() - tic_b))
            # print("\033[1;36m----------------------\033[0m")
            # print("\033[0;92mForward loss: %.2f <==> Backward loss: %.2f\033[0m" %
                  # (fw_loss.data[0], bw_loss.data[0]))
            # print("\033[0;4;92mTOTAL LOSS: %.2f\033[0m" % loss.data[0])
            # print("\033[1;36m----------------------\033[0m")

            n_iter += 1

        if not epoch % nsave:
            print("Epoch %d (%d iters) -- Saving model in %s" % (epoch, n_iter, save_path))
            torch.save(model.state_dict(), "%s_%d" % (save_path, n_iter))
            evaluate(model, criterion)

        print("\033[1;30mEpoch %i/%i: %f seconds\033[0m" % (epoch, numepochs, time.time() - tic))


def main():
    """Forward sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, help='batch size', default=10)
    parser.add_argument('--save_path', '-sp', type=str, help='path to save the model',
                        default='models/model.pth')
    parser.add_argument('--load_path', '-mp', type=str, help='path to load the model',
                        default=None)
    parser.add_argument('--cuda', dest='cuda', help='use cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', help="don't use cuda", action='store_false')
    parser.add_argument('--batch_first', dest='batch_first', action='store_true')
    parser.add_argument('--no-batch_first', dest='batch_first', action='store_false')
    parser.add_argument('--multigpu', nargs='*', default=[], help='list of gpus to use')
    parser.set_defaults(cuda=True)
    parser.set_defaults(batch_first=True)
    parser.set_defaults(create_vocab=False)
    args = parser.parse_args()

    filenames = {'train': 'train_no_dup.json',
                 'test': 'test_no_dup.json',
                 'val': 'valid_no_dup.json'}

    tic = time.time()
    print("Reading all texts and creating the vocabulary")
    all_texts = [TXT_TEST_VAL_TF(t['name']) for d in json.load(open(os.path.join('data/label',
                                                   filenames['train']))) for t in d['items']]
    vocab = create_vocab(all_texts)
    print("Vocabulary creation took %.2fsecs - %d words" % (time.time() - tic, len(vocab)))

    model, dataloaders, optimizer, criterion, contrastive_criterion = config(
        net_params=[512, 512, 0.2, len(vocab), args.load_path],
        data_params=['data/images', 'data/label',
                     filenames],
        opt_params=[0.2, 1e-4],
        batch_params=[args.batch_size, args.batch_first],
        cuda_params=[args.cuda, args.multigpu])
    print("before training: lr = %.4f" % optimizer.param_groups[0]['lr'])

    scheduler = StepLR(optimizer, 2, 0.5)

    train([model, criterion, contrastive_criterion, optimizer, scheduler, vocab],
          dataloaders, args.cuda, args.batch_first,
          [20, 3, args.save_path])


if __name__ == '__main__':
    main()
    WRITER.close()
