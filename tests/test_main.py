import unittest
from bilstm.main import config, train


class TestMain(unittest.TestCase):

    def test_config_train(self):

        model, dataloaders, optimizer, criterion = config(
            net_params=[20, 20, 1],
            data_params=['tests/data', 'tests/data',
                         {'train': 'data_sample.json',
                          'test': 'data_sample.json',
                          'val': 'data_sample.json'}],
            opt_params=[0.2, 1e-4],
            batch_params=[2, True],
            cuda_params=[False, []])

        self.assertTrue('train' in dataloaders)
        self.assertTrue('test' in dataloaders)
        self.assertTrue('val' in dataloaders)

        batch = next(iter(dataloaders['train']))
        self.assertTrue('images' in batch[0])

        train([model, criterion, optimizer], batch, 0, False, True)


if __name__ == '__main__':
    unittest.main()
