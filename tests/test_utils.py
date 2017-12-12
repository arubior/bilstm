import unittest
import torch
import nltk
from PIL import Image
from bilstm.src.utils import seqs2batch, ImageTransforms, TextTransforms
nltk.download('wordnet')



class TestUtils(unittest.TestCase):

    def test_seqs2batch(self):

        img = torch.randn(20, 20, 3)
        data = [{'images': [img, img]}]*4
        images, seq_lens, lookup_table = seqs2batch(data)

        self.assertTrue(len(seq_lens) == len(data) == len(images)/2)
        self.assertTrue(lookup_table == [[0, 1], [2, 3], [4, 5], [6, 7]])

        images, seq_lens, lookup_table = seqs2batch([])
        self.assertTrue(len(images) == len(seq_lens) == len(lookup_table) == 0)


    def test_ImageTransforms(self):

        img = Image.open('tests/image.jpg')
        img_trf = ImageTransforms(size=20, angle=5, crop_size=20, hflip_ratio=1)

        resized_img = img_trf.resize(img)
        self.assertEqual(resized_img.size, (20, 20))

        rotated_img = img_trf.random_rotation(img)
        self.assertFalse(rotated_img, img)

        croped_img = img_trf.random_crop(img)
        self.assertEqual(croped_img.size, (20, 20))

        flipped_img = img_trf.random_horizontal_flip(img)
        self.assertEqual(flipped_img.size, img.size)


    def test_TextTransforms(self):

        text = "Text with uppercase Letters, commas and num83rs."
        txt_trf = TextTransforms(keep_numbers=False, delete_ratio=0.5)

        norm_text = txt_trf.normalize(text)
        self.assertNotEqual(text, norm_text)
        self.assertEqual(sum([str(n) in text for n in range(10)]), 0)
        self.assertEqual(sum([l.isupper() for l in norm_text]), 0)

        del_text = txt_trf.random_delete(text)
        self.assertTrue(len(del_text) <= len(text))


if __name__ == '__main__':
    unittest.main()
