import unittest
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from bilstm.src.losses import LSTMLosses, SBContrastiveLoss


class TestLosses(unittest.TestCase):

    def test_LSTMLosses(self):

        max_seq_len = 2
        feats_len = 3

        seq_lens = [max_seq_len, max_seq_len]
        feats = torch.autograd.Variable(torch.zeros(len(seq_lens), max_seq_len, feats_len))
        packed_batch = pack_padded_sequence(feats, seq_lens, batch_first=True)

        out = torch.autograd.Variable(torch.zeros(len(seq_lens), max_seq_len, 2*feats_len))

        criterion = LSTMLosses(batch_first=True, cuda=False)
        fw_loss, bw_loss = criterion(packed_batch, out)
        self.assertAlmostEqual(fw_loss.data[0], 0.34657359)
        self.assertAlmostEqual(bw_loss.data[0], 0.34657359)

    def test_SBContrastiveLoss(self):

        feats_len = 3
        feats1 = torch.autograd.Variable(torch.zeros(1, feats_len))
        feats2 = torch.autograd.Variable(torch.zeros(1, feats_len))
        labels = 1

        closs = SBContrastiveLoss(margin=0.2)
        loss = closs(feats1, feats2, labels)
        self.assertAlmostEqual(loss.data[0], 0.03999930)


if __name__ == '__main__':
    unittest.main()
