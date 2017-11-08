import os
import json
from PIL import Image
from skimage import io
from torch.utils.data import Dataset


########################################################
# KIWIS & LLAMAS
# ~~~~~~~~~~~~~~

class KiwisLlamasDataset(Dataset):
    """ Kiwis and llamas dataset."""

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with path to images.
            root_dir (string): Directory where the txt files train and test are
                               located (and also the folders with images).
            transform (callable, optional): Optional transform to be applied on
                                            a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f.replace('\n', '') for f in
                                open(os.path.join(root_dir, txt_file)).readlines()]
        # Label 0 for kiwis, 1 for llamas:
        self.tags = [0 if 'kiwis' in f else 1 for f in self.filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'tag': self.tags[idx]}

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
        set_id = self.data[idx]['set_id']
        items = self.data[idx]['items']
        images = [Image.open(os.path.join(self.img_dir, set_id, '%s.jpg' % i['index'])) for i in items]
        texts = [i['name'] for i in items]

        if self.img_transform:
            images = [self.img_transform(image) for image in images]

        if self.txt_transform:
            texts = [self.txt_transform(t) for t in texts]

        return (images, texts)
