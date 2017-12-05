"""Custom transformations for image and text preprocessing."""
import random
import PIL
import numpy as np
# reload(sys)
# sys.setdefaultencoding('utf8')
from nltk.stem import WordNetLemmatizer

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
        x = random.randint(0, width-crop_x - 1)
        y = random.randint(0, height-crop_y - 1)
        return img.crop((x, y, x + crop_x, y + crop_y))

class TextTransforms(object):
    def __init__(self, keep_numbers=False, delete_ratio=0):
        self.lemmatizer = WordNetLemmatizer()
        assert isinstance(keep_numbers, bool), "keep_numbers must be a boolean value"
        self.keep_numbers = keep_numbers
        assert isinstance(delete_ratio, float) or isinstance(delete_ratio, int), "Deletion ratio must be a float or int"
        self.delete_ratio = delete_ratio
        pass

    def normalize(self, text):
        text = text.lower()
        text2 = ''
        for w in text.split():
            text2 += ' ' +  self.lemmatizer.lemmatize(w)

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
        text = ' , '.join(text.split(','))
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
            text = text.replace('1',' ').replace('2',' ').replace('3',' '). \
                        replace('4',' ').replace('5',' ').replace('6',' '). \
                        replace('7',' ').replace('8',' ').replace('9',' ').replace('0',' ')
        return text

    def random_delete(self, text):
        words = text.split()
        perm = np.random.permutation(len(text.split()))
        to_delete = np.array(words)[perm[:int(random.random()*self.delete_ratio*len(words))]]

        return ' '.join([w for w in words if w not in to_delete])
