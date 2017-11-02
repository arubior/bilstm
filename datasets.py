import os
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
