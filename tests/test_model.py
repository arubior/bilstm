import unittest
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from bilstm.src.model import FullBiLSTM as model


class TestModel(unittest.TestCase):

    def test_model(self):

        input_dim = 20
        hidden_dim = 10
        size = 299
        batch_first = True
        init_seq_lens = [2, 2, 1]
        lookup_table = [[0, 1], [2, 3], [4]]
        net = model(input_dim, hidden_dim, batch_first)

        inputs = torch.autograd.Variable(
            torch.randn(5, 3, size, size).type(torch.FloatTensor),
            requires_grad=False)

        out, hidden = net(inputs, init_seq_lens, lookup_table, net.init_hidden(3))
        out, seq_lens = pad_packed_sequence(out, batch_first=batch_first)
        self.assertEqual(out.size(), torch.Size([3, 2, 20]))
        self.assertEqual(seq_lens, init_seq_lens)
        self.assertEqual(hidden[0][1], [3, 2])


if __name__ == '__main__':
    unittest.main()
