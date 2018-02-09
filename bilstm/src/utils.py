"""Some utilities."""
# Disable no-member for torch.
# pylint: disable=E1101
# Disable superfluous-parens for python 3
# pylint: disable=C0325
import random
import torch
import PIL
import numpy as np
# import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer


# Disable too-many-locals (no clear way of reducing them).
# pylint: disable=R0914
def seqs2batch(data, word_to_ix):
    """Get a list of images and texts from a list of sequences.

    Args:
        - data: list of sequences (shaped batch_size x seq_len, with seq_len variable).
        - word_to_ix: dictionary. Keys are unique words, values are unique indices.

    Returns:
        - images: torch.Tensor of images.
        - texts: torch.Tensor of stacked one-hot encoding matrices for texts (M words x
                                                                              N vocab_size).
        - seq_lens: list of sequence lengths.
        - im_lookup_table: list (shaped batch_size x seq_len, with seq_len variable) containing
          the indices of images in the image list.
        - txt_lookup_table: list (shaped batch_size x seq_len x text_len, with seq_len and
          text_len variable) containing the indices of words in the text list.

    """
    # Get all inputs and keep the information about the sequence they belong to.
    images = torch.Tensor()
    texts = torch.Tensor()
    # img_data, txt_data = zip([(i['images'], i['texts']) for i in data])
    img_data = [i['images'] for i in data]
    txt_data = [i['texts'] for i in data]
    seq_lens = torch.zeros(len(img_data)).int()
    im_lookup_table = [None] * len(data)
    txt_lookup_table = [None] * len(data)
    count = 0
    word_count = 0
    for seq_tag, (seq_imgs, seq_txts) in enumerate(zip(img_data, txt_data)):
        im_seq_lookup = []
        txt_seq_lookup = []
        for img, txt in zip(seq_imgs, seq_txts):
            text_to_append = range(word_count, word_count + len(txt.split()))
            if not text_to_append:
                # IMAGES WITHOUT TEXT ARE NORMALLY IRRELEVANT.
                # plt.imshow(img.permute(1, 2, 0).numpy())
                # plt.show()
                continue
            images = torch.cat((images, img.unsqueeze(0)))
            texts = torch.cat((texts, get_one_hot(txt, word_to_ix)))
            im_seq_lookup.append(count)
            txt_seq_lookup.append(range(word_count, word_count + len(txt.split())))
            count += 1
            word_count += len(txt.split())
            seq_lens[seq_tag] += 1
        im_lookup_table[seq_tag] = im_seq_lookup
        txt_lookup_table[seq_tag] = txt_seq_lookup

    return images, texts, seq_lens, im_lookup_table, txt_lookup_table


def create_vocab(texts):
    """Create vocabulary for one-hot word encodings.

    Args:
        - texts: list of sentences.

    Returns:
        - word_to_ix: dictionary. Keys are unique words, values are unique indices.

    """
    idx = 0
    word_to_ix = dict()
    for word in ' '.join(texts).split():
        if word not in word_to_ix:
            word_to_ix[word] = idx
            idx += 1
    return word_to_ix


def get_one_hot(text, word_to_ix):
    """Get a matrix of one-hot encoding vectors for all words in a text.

    Args:
        - text: (str)
        - word_to_ix: dictionary. Keys are unique words, values are unique indices.

    Returns:
        - encodings: (torch.Tensor) matrix with size ((M words) x (vocab. size))

    """
    encodings = torch.zeros(len(text.split()), len(word_to_ix))
    for i, word in enumerate(text.split()):
        try:
            encodings[i, word_to_ix[word]] = 1
        except KeyError:
            print("Word %s not in vocabulary" % word)

    return encodings

class ImageTransforms(object):
    """Custom image transformations.

    Args:
        [size]: size to resize images (can be int (square) or tuple(width, height)).
        [angle]: maximum angle for rotation.
        [crop_size]: size of the random crops (again: int or tuple).

    """

    def __init__(self, size=None, angle=None, crop_size=None, hflip_ratio=None):
        if size is not None:
            assert isinstance(size, (int, tuple)), "Size must be tuple or int"
            assert size > 0, "Size must be greater than 0"
            if isinstance(size, tuple):
                assert len(size) == 2, "Size must have 1 (square) or 2 dimensions: (width, height)"
            if isinstance(size, int):
                size = (size, size)
        self.size = size

        if angle is not None:
            assert isinstance(angle, (float, int)), "Angle must be a float or int"
        self.angle = angle

        if crop_size is not None:
            assert isinstance(crop_size, (int, tuple)), "Size must be a tuple or an int"
            assert crop_size > 0, "Size must be greater than 0"
            if isinstance(crop_size, tuple):
                assert len(crop_size) == 2, "Size must have 1 (square) or 2 dim: (width, height)"
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        if hflip_ratio is not None:
            assert isinstance(hflip_ratio, (int, float)), "hflip_ratio must be int or float"
            assert hflip_ratio <= 1 and hflip_ratio >= 0, "hflip_ratio must be between [0, 1]"
        self.hflip_ratio = hflip_ratio

    def resize(self, img):
        """Resize and image.

        Args:
            img: PIL image.

        Returns:
            Resized PIL image.

        """
        assert isinstance(img, PIL.Image.Image), "Image must be a PIL.Image.Image"
        if self.size is not None:
            return img.resize(self.size)
        else:
            raise ValueError('Size is not defined')

    def random_rotation(self, img):
        """Rotate randomly an image.

        Args:
            img: PIL image.

        Returns:
            Rotated PIL image.

        """
        assert isinstance(img, PIL.Image.Image), "Image must be a PIL.Image.Image"
        if self.angle is not None:
            return img.rotate(2*(random.random() - 0.5)*self.angle)
        else:
            raise ValueError('Angle is not defined')

    def random_horizontal_flip(self, img):
        """Randomly flip horizontally an image.

        Args:
            img: PIL image.

        Returns:
            PIL image (flipped or not).

        """
        assert isinstance(img, PIL.Image.Image), "Image must be a PIL.Image.Image"
        if random.random() < self.hflip_ratio:
            return img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return img

    def random_crop(self, img):
        """Randomly crop an image.

        Args:
            img: PIL image.

        Returns:
            PIL image of the cropped part.

        """
        assert isinstance(img, PIL.Image.Image), "Image must be a PIL.Image.Image"
        width, height = img.size
        crop_x, crop_y = self.crop_size
        x_left = random.randint(0, width-crop_x - 1)
        y_top = random.randint(0, height-crop_y - 1)
        return img.crop((x_left, y_top, x_left + crop_x, y_top + crop_y))


class TextTransforms(object):
    """Custom text transformations.

    Args:
        [keep_numbers]: (boolean) whether to keep or not the numbers in the string.
        [delete_ratio]: (float between 0-1) the portion of words to be randomly removed.

    """
    def __init__(self, keep_numbers=False, delete_ratio=0):
        self.lemmatizer = WordNetLemmatizer()
        assert isinstance(keep_numbers, bool), "keep_numbers must be a boolean value"
        self.keep_numbers = keep_numbers
        assert isinstance(delete_ratio, (float, int)), "Deletion ratio must be a float or int"
        self.delete_ratio = delete_ratio

    def normalize(self, text):
        """Normalize text (remove symbols, lowercase)."""
        text = text.lower()
        text2 = ''
        for word in text.split():
            text2 += ' ' + self.lemmatizer.lemmatize(word)

        text = text2
        text = " ".join(text.split("''"))
        text = " ' ".join(text.split("'"))
        text = ' " '.join(text.split('"'))
        text = ' '.join(text.split('http'))
        text = ' '.join(text.split('https'))
        text = ' '.join(text.split('.com'))
        text = ' '.join(text.split('&quot'))
        text = ' . '.join(text.split('.'))
        text = ' '.join(text.split('<br />'))
        text = ' , '.join(text.split(', '))
        text = ' , '.join(text.split(', '))
        text = ' ( '.join(text.split('('))
        text = ' ) '.join(text.split(')'))
        text = ' ! '.join(text.split('!'))
        text = ' ? '.join(text.split('?'))
        text = ' '.join(text.split(';'))
        text = ' '.join(text.split(':'))
        text = ' - '.join(text.split('-'))
        text = ' '.join(text.split('='))
        text = ' '.join(text.split('*'))
        text = ' '.join(text.split('\n'))
        text = ' '.join(text.split('@'))
        text = ' '.join(text.split('/'))
        if not self.keep_numbers:
            text = text.replace('1', ' ').replace('2', ' ').replace('3', ' '). \
                        replace('4', ' ').replace('5', ' ').replace('6', ' '). \
                        replace('7', ' ').replace('8', ' ').replace('9', ' ').replace('0', ' ')
        return text

    def random_delete(self, text):
        """Randomly delete some words (according to the specified ratio for the class)."""
        words = text.split()
        perm = np.random.permutation(len(text.split()))
        to_delete = np.array(words)[perm[:int(random.random()*self.delete_ratio*len(words))]]

        return ' '.join([w for w in words if w not in to_delete])


def write_tensorboard(writer, data, n_iter):
    """Write several scalars in a tensorboard writer.

    Args:
        writer: SummaryWriter object from tensorboardX.
        data: dictionary with 'name for writing': data for writing.
        n_iter: number of iteration to write.

    """
    for name, value in data.items():
        writer.add_scalar(name, value, n_iter)
