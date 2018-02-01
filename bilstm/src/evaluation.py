"""Evaluate results with a trained model."""
import os
import sys
import time
import argparse
import h5py
import numpy as np
from PIL import Image
import torch
# from model_lstm import FullBiLSTM
from bilstm.src.model import FullBiLSTM
from bilstm.src.losses import LSTMLosses
from bilstm.src.utils import ImageTransforms
import torchvision
from sklearn import metrics


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
    def __init__(self, model, weights, img_dir, batch_first, cuda):
        """Load the model weights."""
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.model.eval()
        self.model.load_state_dict(torch.load(weights))
        self.img_dir = img_dir
        self.trf = ImageTransforms(299)
        self.criterion = LSTMLosses(batch_first, cuda=cuda)
        self.batch_first = batch_first
        self.cuda = cuda

    def compatibility(self, sequence, test_feats):
        """Get the compatibility score of a sequence of images.
        Right now, it computes probability among the images of the own
        sequence. Theoretically, it has to compute probability amongst
        all images in the test(?) dataset (line 77 of
        https://github.com/xthan/polyvore/blob/master/polyvore/fashion_compatibility.py)

        """
        # Disable complaints about no-member in torch
        # pylint: disable=E1101
        try:
            # im_feats = torch.from_numpy(np.array([test_feats[bytes(d, 'utf8')]
            # for d in sequence]))
            im_feats = torch.from_numpy(np.array([test_feats[d] for d in sequence]))
        except KeyError:
            print("Key %s not in the precomputed features." % d)
            import epdb
            epdb.set_trace()
        if self.cuda:
            im_feats = im_feats.cuda()
        out, _ = self.model.lstm(torch.autograd.Variable(im_feats).unsqueeze(0))
        out = out.data
        im_feats = torch.from_numpy(np.array(list(test_feats.values())))
        if self.cuda:
            im_feats = im_feats.cuda()
        x_fw = torch.zeros(im_feats.size(0) + 1, im_feats.size(1))
        x_bw = torch.zeros(im_feats.size(0) + 1, im_feats.size(1))
        if self.cuda:
            x_fw = x_fw.cuda()
            x_bw = x_bw.cuda()

        x_fw[:im_feats.size(0)] = im_feats
        x_bw[1 : im_feats.size(0) + 1] = im_feats
        fw_hiddens = out[0, :im_feats.size(0), :out.size(2) // 2]
        bw_hiddens = out[0, :im_feats.size(0), out.size(2) // 2:]
        fw_logprob = torch.nn.functional.log_softmax(torch.autograd.Variable(
            torch.mm(fw_hiddens, x_fw.permute(1, 0)))).data
        bw_logprob = torch.nn.functional.log_softmax(torch.autograd.Variable(
            torch.mm(bw_hiddens, x_bw.permute(1, 0)))).data
        score = torch.diag(fw_logprob[:, 1 : fw_logprob.size(0) + 1]).mean() +\
            torch.diag(bw_logprob[:, :bw_logprob.size(0)]).mean()
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
                # Images with size = 1 in any dimension are useless.
                if np.any(np.array(img.size) == 1):
                    continue
            images.append(img)

        return images

    def get_img_feats(self, img_data):
        """Get the features for some images."""
        images = torch.Tensor()
        imtr = lambda x: torchvision.transforms.ToTensor()(self.trf.resize(x))
        # Disable complaints about no-member in torch
        # pylint: disable=E1101
        for img in img_data:
            images = torch.cat((images, imtr(img).unsqueeze(0)))
        # pylint: enable=E1101
        images = torch.autograd.Variable(images)
        if self.cuda:
            images = images.cuda()
        return self.model.cnn(images)


# Disable too-many-locals. No clear way to reduce them
# pylint: disable= R0914
def main(model_name, feats_name):
    """Main function."""
    compatibility_file = 'data/label/fashion_compatibility_prediction.txt'
    jump = 1

    data = h5py.File(feats_name, 'r')
    data_dict = dict()
    for fname, feat in zip(data['filenames'], data['features']):
        data_dict[fname] = feat

    model = FullBiLSTM(512, 512, 2480, batch_first=True, dropout=0.7)
    evaluator = Evaluation(model, model_name, 'data/images',
                           batch_first=True, cuda=True)

    seqs = [l.replace('\n', '') for l in open(compatibility_file).readlines()]
    labels = []
    scores = []
    tic = time.time()
    for i, seq in enumerate(seqs[::jump]):
        seqtag = seq.split()[0]
        seqdata = seq.split()[1:]
        compat = evaluator.compatibility(seqdata, data_dict)
        scores.append(compat)
        labels.append(int(seqtag))
        sys.stdout.write("(%d/%d) SEQ LENGTH = %d - TAG: %s - COMPAT: %.4f - %.2f min left\r" %
                         (i*jump, len(seqs), len(seqdata), seqtag, compat,
                          (time.time() - tic)/(i + 1)*jump*(len(seqs)/jump - i*jump)/60))
        sys.stdout.flush()
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    print("\033[0;31m\nModel: %s\033[0m" % model_name)
    print("\033[1;30mCompatibility AUC: %f for %d outfits\033[0m" % (metrics.auc(fpr, tpr), len(labels)))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--model_path', '-m', type=str, help='path to the model', default='')
    PARSER.add_argument('--feats_path', '-sp', type=str, help='path to the features', default='')
    ARGS = PARSER.parse_args()
    main(ARGS.model_path, ARGS.feats_path)
