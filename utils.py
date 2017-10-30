import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import transform

plt.ion() # interactive mode


########################################################
# TRANSFORMS
# ~~~~~~~~~~

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')

        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = image[top: top + new_h,
                      left: left + new_w]

        return img


class RandomHorizontalFlip(object):
    """Randomly flip horizontally an image."""

    def __call__(self, image):
        if np.random.random() > 0.5:
            img = image[:, ::-1, :]
        else:
            img = image

        return img


class Normalize(object):
    """Normalize an image.

    Args:
        mean: (tuple, list or float): mean to normalize. If int, same value is used for
            all channels. If list/tuple, each value is used for one channel.
        std: (tuple, list or float): standard deviation to normalize. If int, same value is used for
            all channels. If list/tuple, each value is used for one channel.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float, tuple, list))
        if isinstance(mean, float):
            self.mean = [mean, mean, mean]
        else:
            assert len(mean) == 3
            self.mean = mean

        assert isinstance(std, (float, tuple, list))
        if isinstance(std, float):
            self.std = [std, std, std]
        else:
            assert len(std) == 3
            self.std = std

    def __call__(self, image):
        nchannels = image.shape[2]

        img = image.astype(np.float64)
        
        if nchannels == 1:
            img = (img - self.mean[0]) / self.std[0]
        else:
            for c in range(nchannels):
                img[:, :, c] = (img[:, :, c] - self.mean[c]) / self.std[c]

        img = torch.Tensor(img)

        return img

class ToCHW(object):
    """Convert image from (H x W x C) to (C x H x W)"""

    def __call__(self, image):
        img = image.transpose(0, 2).transpose(1, 2)

        return img


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
