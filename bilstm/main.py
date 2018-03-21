"""Script for using the BiLSTM model in model.py with sequence inputs."""
# pylint: disable=E1101
# pylint: disable=C0325
# pylint: disable=W0403
import os
import ast
import time
import json
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_packed_sequence
import torchvision
from src.utils import seqs2batch, ImageTransforms, TextTransforms, create_vocab, write_tensorboard
from src.model import FullBiLSTM as inception
from src.model_vgg import FullBiLSTM as vgg
from src.model_squeezenet import FullBiLSTM as squeezenet
from src.losses import LSTMLosses, SBContrastiveLoss
from src.datasets import PolyvoreDataset
from src.datasets import collate_seq
from tensorboardX import SummaryWriter

torch.manual_seed(1)

########################################################
# DATA LOADER
# ~~~~~~~~~~~

GRADS = {}

TXT_TRF = TextTransforms()
# pylint: disable=W0108
TXT_TEST_VAL_TF = lambda x: TXT_TRF.normalize(x)
# pylint: enable=W0108

def save_grad(name):
    """Save gradient value. To be called from "register_hook"."""
    def hook(grad):
        """Get gradient value."""
        GRADS[name] = grad
    return hook


def config(net_params, data_params, opt_params, cuda_params):
    """Get parameters to configure the experiment and prepare the needed variables.

    Args:
        - net_params: list containing:
            input_dimension for LSTM (int)
            output_dimension for LSTM (int)
            margin for contrastive loss (float)
            size of the vocabulary (int)
            load_path for loading weights (str) (None by default)
            freeze (bool) whether or not freezing the cnn layers
        - data_params: dictionary with keys:
            'img_dir': path to the directory where images are (string)
            'json_dir': path to the directory wher jsons are (string)
            'json_files': names of the train, test and validation jsons (dictionary)
            'batch_size': batch_size (int)
            'batch_first': batch_first (bool) for the LSTM sequences
        - opt_params: dictionary with keys:
            'learning_rate': learning rate value (float)
            'weight_decay': weight decay value (float)
        - cuda_params: dictionary with keys:
            'cuda': (bool): whether to use GPU or not
            'multigpu': (list of int): indices of GPUs to use

    Returns:
        - model: pytorch model to train
        - dataloaders: data iterators
        - scheduler: scheduler of the optimizer function to train
        - criterion: loss equation to train

    """
    model_type, input_dim, hidden_dim, margin, vocab_size, load_path, freeze = net_params


    if model_type == 'inception':

        model = inception(input_dim, hidden_dim, vocab_size, data_params['batch_first'],
                           dropout=0.7, freeze=freeze)
        img_size = 299
        img_trf = {'train': ImageTransforms(img_size + 6, 5, img_size, 0.5),
                   'test': ImageTransforms(img_size)}
        img_train_tf = lambda x: torchvision.transforms.ToTensor()(img_trf['train'].random_crop(
            img_trf['train'].random_rotation(img_trf['train'].random_horizontal_flip(
                img_trf['train'].resize(x)))))
        img_test_val_tf = lambda x: torchvision.transforms.ToTensor()(img_trf['test'].resize(x))

    elif model_type == 'vgg':

        model = vgg(input_dim, hidden_dim, vocab_size, data_params['batch_first'],
                           dropout=0.7, freeze=freeze)
        img_size = 224
        norm_trf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img_trf = {'train': ImageTransforms(img_size + 6, 5, img_size, 0.5),
                   'test': ImageTransforms(img_size)}
        img_train_tf = lambda x: norm_trf(torchvision.transforms.ToTensor()(img_trf['train'].random_crop(
            img_trf['train'].random_rotation(img_trf['train'].random_horizontal_flip(
                img_trf['train'].resize(x))))))
        img_test_val_tf = lambda x: norm_trf(torchvision.transforms.ToTensor()(img_trf['test'].resize(x)))

    elif model_type == 'squeezenet':
        model = squeezenet(input_dim, hidden_dim, vocab_size, data_params['batch_first'],
                           dropout=0.7, freeze=freeze)
        img_size = 227
        norm_trf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img_trf = {'train': ImageTransforms(img_size + 6, 5, img_size, 0.5),
                   'test': ImageTransforms(img_size)}
        img_train_tf = lambda x: norm_trf(torchvision.transforms.ToTensor()(img_trf['train'].random_crop(
            img_trf['train'].random_rotation(img_trf['train'].random_horizontal_flip(
                img_trf['train'].resize(x))))))
        img_test_val_tf = lambda x: norm_trf(torchvision.transforms.ToTensor()(img_trf['test'].resize(x)))

    else:
        print("Please, specify a valid model type: inception, vgg or squeezenet"\
              "instead of %s" % model_type)
        return

    txt_train_tf = lambda x: TXT_TRF.random_delete(TXT_TRF.normalize(x))

    img_transforms = {'train': img_train_tf,
                      'test': img_test_val_tf,
                      'val': img_test_val_tf}

    txt_transforms = {'train': txt_train_tf,
                      'test': TXT_TEST_VAL_TF,
                      'val': TXT_TEST_VAL_TF}

    if load_path is not None:
        print("Loading weights from %s" % load_path)
        model.load_state_dict(torch.load(load_path))
    if cuda_params['cuda']:
        print("Switching model to gpu")
        model.cuda()
    if cuda_params['multigpu']:
        print("Switching model to multigpu")
        multgpu = ast.literal_eval(multigpu[0])
        model.cuda()
        model = nn.DataParallel(model, device_ids=cuda_params['multigpu'])

    dataloaders = {x: torch.utils.data.DataLoader(
        PolyvoreDataset(os.path.join(data_params['json_dir'], data_params['json_files'][x]),
                        data_params['img_dir'],
                        img_transform=img_transforms[x], txt_transform=txt_transforms[x]),
        batch_size=data_params['batch_size'],
        shuffle=True, num_workers=24,
        collate_fn=collate_seq,
        pin_memory=True)
                   for x in ['train', 'test', 'val']}

    # Optimize only the layers with requires_grad = True, not the frozen layers:
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                          lr=opt_params['learning_rate'], weight_decay=opt_params['weight_decay'])
    criterion = LSTMLosses(data_params['batch_first'], cuda_params['cuda'])
    contrastive_criterion = SBContrastiveLoss(margin)

    return model, dataloaders, optimizer, criterion, contrastive_criterion


def train(train_params, dataloaders, cuda, batch_first, epoch_params):
    """Train the model.

    """
    model, criterion, contrastive_criterion, optimizer, scheduler, vocab, freeze = train_params
    numepochs, nsave, save_path = epoch_params

    log_name = ('runs/L2/lr%.3f' % optimizer.param_groups[0]['initial_lr'])
    if freeze:
        log_name += '_frozen'
    writer = SummaryWriter(log_name)

    n_iter = 0
    tic_e = time.time()
    for epoch in range(numepochs):
        print("Epoch %d - lr = %.4f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        for batch in dataloaders['train']:

            tic = time.time()
            # Clear gradients, reset hidden state.
            model.zero_grad()
            hidden = model.init_hidden(len(batch))

            # Get a list of images and texts from sequences:
            images, texts, seq_lens, im_lookup_table, txt_lookup_table = seqs2batch(batch, vocab)

            images = autograd.Variable(images)
            texts = autograd.Variable(texts)
            if cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
                images = images.cuda()
                texts = texts.cuda()

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
            loss = lstm_loss + cont_loss

            if np.isnan(loss.cpu().data.numpy()) or lstm_loss.cpu().data[0] < 0:
                import epdb
                epdb.set_trace()

            im_feats.register_hook(save_grad('im_feats'))
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()
            print("iteration %d took %.2f secs" % (n_iter, time.time() - tic))

            print("\033[4;32miter %d\033[0m" % n_iter)
            print("\033[1;34mTotal loss: %.3f ||| LSTM loss: %.3f ||| Contr. loss: %.3f\033[0m" %
                  (loss.data[0], lstm_loss.data[0], cont_loss.data[0]))
            print("Seq lens:", [len(b['texts']) for b in batch])

            dists = torch.sum(1 - F.cosine_similarity(im_feats, txt_feats))/im_feats.size()[0]
            print("\033[0;31mmdists: %.3f\033[0m" % dists.data[0])

            write_data = {'data/loss': loss.data[0],
                          'data/lstm_loss': lstm_loss.data[0],
                          'data/cont_loss': cont_loss.data[0],
                          'data/pos_dists': dists.data[0],
                          'grads/im_feats': torch.mean(torch.cat([torch.norm(t)
                                                                  for t in GRADS['im_feats']]))}
            write_tensorboard(writer, write_data, n_iter)

            n_iter += 1

            if not n_iter % nsave:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print("Epoch %d (%d iters) -- Saving model in %s" % (epoch, n_iter, save_path))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), "%s_%d.pth" % (
                    os.path.join(save_path, 'model'), n_iter))

        print("\033[1;30mEpoch %i/%i: %f seconds\033[0m" % (epoch, numepochs, time.time() - tic_e))
    writer.close()


def main():
    """Forward sequences."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, help='batch size', default=10)
    parser.add_argument('--save_path', '-sp', type=str, help='path to save the model',
                        default='models')
    parser.add_argument('--load_path', '-mp', type=str, help='path to load the model',
                        default=None)
    parser.add_argument('--lr', '-lr', type=float, help='initial learning rate',
                        default=0.2)
    parser.add_argument('--model_type', '-t', type=str, help='type of the model: inception,'
                        'vgg or squeezenet', default='inception')
    parser.add_argument('--cuda', dest='cuda', help='use cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', help="don't use cuda", action='store_false')
    parser.add_argument('--freeze', '-fr', dest='freeze', help='freeze cnn layers',
                        action='store_true')
    parser.add_argument('--batch_first', dest='batch_first', action='store_true')
    parser.add_argument('--no-batch_first', dest='batch_first', action='store_false')
    parser.add_argument('--multigpu', nargs='*', default=[], help='list of gpus to use')
    parser.set_defaults(cuda=True)
    parser.set_defaults(freeze=False)
    parser.set_defaults(batch_first=True)
    args = parser.parse_args()

    filenames = {'train': 'train_no_dup.json',
                 'test': 'test_no_dup.json',
                 'val': 'valid_no_dup.json'}

    tic = time.time()
    print("Reading all texts and creating the vocabulary")
    # Create the vocabulary with all the texts.
    vocab = create_vocab([TXT_TEST_VAL_TF(t['name']) for d in json.load(
        open(os.path.join('data/label', filenames['train'])))
                          for t in d['items']])
    print("Vocabulary creation took %.2f secs - %d words" % (time.time() - tic, len(vocab)))
    import epdb; epdb.set_trace()

    data_params = {'img_dir': 'data/images',
                   'json_dir': 'data/label',
                   'json_files': filenames,
                   'batch_size': args.batch_size,
                   'batch_first': args.batch_first}
    opt_params = {'learning_rate': args.lr,
                  'weight_decay': 1e-4}

    model, dataloaders, optimizer, criterion, contrastive_criterion = config(
        net_params=[args.model_type, 512, 512, 0.2, len(vocab), args.load_path, args.freeze],
        data_params=data_params,
        opt_params=opt_params,
        cuda_params={'cuda': args.cuda,
                     'multigpu': args.multigpu})

    print("before training: lr = %.4f" % optimizer.param_groups[0]['lr'])

    scheduler = StepLR(optimizer, 2, 0.5)

    train([model, criterion, contrastive_criterion, optimizer, scheduler, vocab, args.freeze],
          dataloaders, args.cuda, args.batch_first,
          [100, 500, args.save_path])


if __name__ == '__main__':
    main()
