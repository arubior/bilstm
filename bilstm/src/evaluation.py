"""Evaluate results with a trained model."""
import os
import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from model_lstm import FullBiLSTM
from losses import LSTMLosses
from utils import ImageTransforms
import torchvision


class Evaluation(object):
    """Evaluate an existing model.

    Args:
        model (pytorch model)
        weights (str): path to the saved weights.

    """

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

    def compatibility(self, sequence):
        """Get the compatibility score of a sequence of images."""
        img_data = self.get_images(sequence)
        images = torch.Tensor()
        res = ImageTransforms(299)
        imtr = lambda x: torchvision.transforms.ToTensor()(res.resize(x))
        for img in img_data:
            images = torch.cat((images, imtr(img).unsqueeze(0)))
        images = torch.autograd.Variable(images)
        if self.cuda:
            images = images.cuda()

        im_table = [range(0, len(images))]
        try:
            packed_seq = self.model.create_packed_seq(self.model.cnn(images), [len(images)], im_table)
        except:
            import epdb; epdb.set_trace()
        # hidden = self.model.init_hidden(1)
        # if self.cuda:
            # hidden = (hidden[0].cuda(), hidden[1].cuda())
        out, _ = self.model.lstm(packed_seq)  #, hidden)
        out, _ = pad_packed_sequence(out, batch_first=self.batch_first)
        fw_loss, bw_loss = self.criterion(packed_seq, out)
        return fw_loss + bw_loss


    def get_images(self, sequence):
        """Get a list of images from a list of names."""
        images = []
        for im_path in sequence:
            img = Image.open(os.path.join(self.img_dir, im_path.replace('_', '/') + '.jpg'))
            try:
                if img.layers == 1:  # Imgs with 1 channel are usually noise.
                    continue
                    # img = Image.merge("RGB", [img.split()[0], img.split()[0], img.split()[0]])
            except AttributeError:
                # Images with size = 1 in any dimension are useless.
                if np.any(np.array(img.size) == 1):
                    continue
            images.append(img)

        return images


def main():
    """Main function."""
    model = FullBiLSTM(512, 512, 2480, batch_first=True, dropout=0.7)
    evaluator = Evaluation(model, '../models/model_lstm.pth_6928', '../data/images',
                           batch_first=True, cuda=False)
    compatibility_file = '../data/label/fashion_compatibility_prediction.txt'
    seqs = [l.replace('\n', '') for l in open(compatibility_file).readlines()]
    pos = []
    neg = []
    for seq in seqs[:10]:
        seqtag = seq.split()[0]
        seqdata = seq.split()[1:]
        if bool(seqtag):
            pos.append(compat)
        else:
            neg.append(compat)


if __name__ == '__main__':
    main()
