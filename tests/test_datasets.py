import unittest
from bilstm.src.datasets import PolyvoreDataset


class TestLosses(unittest.TestCase):

    def test_PolyvoreDataset(self):

        path = 'tests/data/data_sample.json'
        img_dir = 'tests/data'
        data = PolyvoreDataset(path, img_dir)
        self.assertEqual(len(data), 1)
        self.assertTrue('images' in data[0])


if __name__ == '__main__':
    unittest.main()
