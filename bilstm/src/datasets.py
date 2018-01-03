"""Polyvore dataset."""
# pylint: disable=E1101
# pylint: disable=R0903
import os
import json
import collections
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PolyvoreDataset(Dataset):
    """ Polyvore dataset."""

    def __init__(self, json_file, img_dir, img_transform=None, txt_transform=None):
        """
        Args:
            json_file (string): Path to the json file with the data.
            img_dir (string): Directory where the image files are located.
            transform (callable, optional): Optional transform to be applied on
                                            a sample.
        """
        self.img_dir = img_dir
        self.data = json.load(open(json_file))
        self.img_transform = img_transform
        self.txt_transform = txt_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a specific index of the dataset (for dataloader batches).

        Args:
            idx: index of the dataset.

        Returns:
            Dictionary with two fields: images and texts, containing the corresponent sequence.

        """
        set_id = self.data[idx]['set_id']
        items = self.data[idx]['items']
        images = []
        texts = []
        ignored = []
        for i in items:
            img = Image.open(os.path.join(self.img_dir, set_id, '%s.jpg' % i['index']))
            try:
                if img.layers == 1:  # Imgs with 1 channel are usually noise.
                    # ignored.append(set_id + '_%s' % i['index'])
                    # continue
                    img = Image.merge("RGB", [img.split()[0], img.split()[0], img.split()[0]])
            except AttributeError:
                # Images with size = 1 in any dimension are useless.
                ignored.append(set_id + '_%s' % i['index'])
                if np.any(np.array(img.size) == 1):
                    continue
            images.append(img)
            texts.append(i['name'])

        if self.img_transform:
            images = [self.img_transform(image) for image in images]

        if self.txt_transform:
            texts = [self.txt_transform(t) for t in texts]

        return {'images': images, 'texts': texts, 'ignored': ignored}


def collate_seq(batch):
    """Return batches as we want: with variable item lengths."""
    if isinstance(batch[0], collections.Mapping):
        # return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
        return batch
