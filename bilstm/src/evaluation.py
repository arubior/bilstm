"""Evaluate results with a trained model."""
import os
import sys
import time
import argparse
import h5py
import numpy as np
from PIL import Image
import torch
from model import FullBiLSTM as inception
from model_vgg import FullBiLSTM as vgg
from model_squeezenet import FullBiLSTM as squeezenet
from losses import LSTMLosses
from utils import ImageTransforms
import torchvision
from sklearn import metrics
from wevision.transforms import padding as pad


# Disable superfluous-parens warning for python 3.
# pylint: disable=C0325

class Evaluation(object):
    """Evaluate an existing model.

    Args:
        model (pytorch model)
        weights (str): path to the saved weights.

    """

    # We need all the arguments
    # pylint: disable=R0913
    def __init__(self, model, model_type, weights, img_dir, batch_first, cuda):
        """Load the model weights."""
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.model.eval()
        self.model.load_state_dict(torch.load(weights))
        self.img_dir = img_dir
        self.model_type = model_type
        if model_type == 'inception':
            IMG_TRF = ImageTransforms(299)
            self.trf = lambda x: IMG_TRF.resize(x)
        elif model_type == 'vgg':
            TRF = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            IMG_TRF = ImageTransforms(224)
            self.trf = lambda x: TRF(torchvision.transforms.ToTensor()(IMG_TRF.resize(x)))
        elif model_type == 'squeezenet':
            TRF = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            IMG_TRF = ImageTransforms(227)
            self.trf = lambda x: TRF(torchvision.transforms.ToTensor()(IMG_TRF.resize(x)))
        else:
            print("Please, specify a valid model type: inception, vgg or squeezenet"\
                  "instead of %s" % model_type)
            return
        self.criterion = LSTMLosses(batch_first, cuda=cuda)
        self.batch_first = batch_first
        self.cuda = cuda

    def compatibility(self, sequence, test_feats, x_values, i_seq):
        """Get the compatibility score of a sequence of images.
        Right now, it computes probability among the images of the own
        sequence. Theoretically, it has to compute probability amongst
        all images in the test(?) dataset (line 77 of
        https://github.com/xthan/polyvore/blob/master/polyvore/fashion_compatibility.py)

        """
        # Disable complaints about no-member in torch
        # pylint: disable=E1101
        try:
            im_feats = torch.from_numpy(np.array([test_feats[d] for d in sequence]))
        except KeyError:
            im_feats = torch.from_numpy(np.array([test_feats[bytes(d, 'utf8')]
                                                  for d in sequence]))
            # print("Key %s not in the precomputed features." % d)
            # import epdb
            # epdb.set_trace()

        # hidden = self.model.init_hidden(1)

        if self.cuda:
            im_feats = im_feats.cuda()
            # hidden = (hidden[0].cuda(), hidden[1].cuda())

        im_feats = torch.nn.functional.normalize(im_feats, p=2, dim=1)
        out, _ = self.model.lstm(torch.autograd.Variable(im_feats).unsqueeze(0))
                                 # hidden)
        out = out.data

        fw_hiddens = out[0, :im_feats.size(0), :out.size(2) // 2]
        bw_hiddens = out[0, :im_feats.size(0), out.size(2) // 2:]

        if self.cuda:
            im_feats = im_feats.cuda()
            x_values = x_values.cuda()

        fw_logprob = torch.nn.functional.log_softmax(torch.autograd.Variable(
            torch.mm(fw_hiddens, x_values.permute(1, 0))), dim=1).data
        bw_logprob = torch.nn.functional.log_softmax(torch.autograd.Variable(
            torch.mm(bw_hiddens, x_values.permute(1, 0))), dim=1).data
        score = torch.diag(fw_logprob[:, i_seq + 2 : i_seq + 2 + fw_logprob.size(0)]).mean() +\
            torch.diag(bw_logprob[:, i_seq : i_seq + bw_logprob.size(0)]).mean()

        # pylint: enable=E1101

        return score


    def get_images(self, sequence):
        """Get a list of images from a list of names."""
        images = []
        for im_path in sequence:
            img = Image.open(os.path.join(self.img_dir, im_path.replace('_', '/') + '.jpg'))
            try:
                if img.layers == 1:  # Imgs with 1 channel are usually noise.
                    # continue
                    img = Image.merge("RGB", [img.split()[0], img.split()[0], img.split()[0]])
            except AttributeError:
                # Images with size == 1 in any dimension are useless.
                if np.any(np.array(img.size) == 1):
                    continue
            images.append(img)

        return images

    def get_img_feats(self, img_data):
        """Get the features for some images."""
        images = torch.Tensor()
        # Disable complaints about no-member in torch
        # pylint: disable=E1101
        for img in img_data:
            images = torch.cat((images, self.trf(img).unsqueeze(0)))
        # pylint: enable=E1101
        images = torch.autograd.Variable(images)
        if self.cuda:
            images = images.cuda()
        if self.model.__module__ == 'model_squeezenet':
            self.model.cnn.num_classes = self.model.input_dim
        return self.model.cnn(images)


# Disable too-many-locals. No clear way to reduce them
# pylint: disable= R0914
def main(model_name, feats_name, model_type):
    """Main function."""
    compatibility_file = 'data/label/fashion_compatibility_prediction.txt'

    data = h5py.File(feats_name, 'r')
    data_dict = dict()
    for fname, feat in zip(data['filenames'], data['features']):
        data_dict[fname] = feat

    if model_type == 'inception':
        model = inception(512, 512, 2480, batch_first=True, dropout=0.7)
    elif model_type == 'vgg':
        model = vgg(512, 512, 2480, batch_first=True, dropout=0.7)
    elif model_type == 'squeezenet':
        model = squeezenet(512, 512, 2480, batch_first=True, dropout=0.7)
    else:
        print("Please, specify a valid model type: inception, vgg or squeezenet, instead of %s" % model_type)
    evaluator = Evaluation(model, model_type, model_name, 'data/images',
                           batch_first=True, cuda=True)

    seqs = [l.replace('\n', '') for l in open(compatibility_file).readlines()]
    labels = []
    scores = []
    tic = time.time()
    feats = np.zeros((np.sum([len(s.split()[1:]) for s in seqs]) + 2*len(seqs),
                      list(data_dict.values())[2].shape[0]))

    seq_start_idx = [None] * len(seqs)
    seq_start_idx[0] = 0
    cum_lens = np.hstack((0, np.cumsum([len(s.split()[1:]) for s in seqs])))
    for i, seq in enumerate(seqs):
        sys.stdout.write("Concatenating sequences (%d/%d)\r" % (i, len(seqs)))
        sys.stdout.flush()
        if i != 0:
            seq_start_idx[i] = cum_lens[i] + 2 * i
        seqimgs = seq.split()[1:]
        # Disable complaints about no-member in torch
        # pylint: disable=E1101
        try:
            im_feats = torch.from_numpy(np.array([data_dict[d] for d in seqimgs]))
        except KeyError as e:
            im_feats = torch.from_numpy(np.array([data_dict[bytes(d, 'utf8')] for d in seqimgs]))
            # print("Key %s not in the precomputed features." % e)
            # import epdb
            # epdb.set_trace()

        feats[seq_start_idx[i] + 1 : seq_start_idx[i] + 1 + len(im_feats), :] = im_feats

    feats = torch.from_numpy(feats.astype(np.float32))
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    for i, seq in enumerate(seqs):
        seqtag = seq.split()[0]
        seqdata = seq.split()[1:]
        compat = evaluator.compatibility(seqdata, data_dict, feats, seq_start_idx[i])
        scores.append(compat)
        labels.append(int(seqtag))
        sys.stdout.write("(%d/%d) SEQ LENGTH = %d - TAG: %s - COMPAT: %.4f - %.2f min left\r" %
                         (i, len(seqs), len(seqdata), seqtag, compat,
                          (time.time() - tic)/(i + 1)*(len(seqs) - i)/60))
        sys.stdout.flush()
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    print("\033[0;31m\nModel: %s\033[0m" % model_name)
    print("\033[1;30mCompatibility AUC: %f for %d outfits\033[0m" % (metrics.auc(fpr, tpr), len(labels)))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--model_path', '-m', type=str, help='path to the model', default='')
    PARSER.add_argument('--feats_path', '-sp', type=str, help='path to the features', default='')
    PARSER.add_argument('--model_type', '-t', type=str, help='type of the model: inception, vgg or squeezenet',
                        default='inception')
    ARGS = PARSER.parse_args()
    main(ARGS.model_path, ARGS.feats_path, ARGS.model_type)
