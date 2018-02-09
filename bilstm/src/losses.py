"""Loss functions used in the Bi-LSTM paper."""
# pylint: disable=E1101
# pylint: disable=W0221
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_packed_sequence


def paper_dist(desc1, desc2):
    """Distance metric used in the paper: cosine distance with normalized vectors."""
    dists = torch.cat([torch.dot(a, b) for a, b in zip(desc1, desc2)])
    if desc1.is_cuda and desc2.is_cuda:
        dists = dists.cuda()
    return dists


class LSTMLosses(nn.Module):
    """Compute the forward and backward loss of a batch.

    Args:
        - seq_lens: sequence lengths
        - cuda: bool specifying whether to use (True) or not (False) a GPU.

    """

    def __init__(self, batch_first, cuda):
        super(LSTMLosses, self).__init__()
        self.batch_first = batch_first
        self.cuda = cuda

    # Disable too-many-locals (no clear way of reducing them).
    # pylint: disable=R0914
    def forward(self, packed_feats, hidden):
        """Compute forward and backward losses.

        Args:
            - feats: a PackedSequence batch inputs of the LSTM (padded) - for now, batch first
                    (batch_size x max_seq_len x feat_dimension)
            - hidden: outputs of the LSTM for the same batch - for now, batch first
                    (batch_size x max_seq_len x hidden_dim)

        Returns:
            Tuple containing two autograd.Variable: the forward and backward losses for a batch.

        """
        feats, seq_lens = pad_packed_sequence(packed_feats, batch_first=self.batch_first)
        fw_loss = autograd.Variable(torch.zeros(1))
        bw_loss = autograd.Variable(torch.zeros(1))
        if self.cuda:
            fw_loss = fw_loss.cuda()
            bw_loss = bw_loss.cuda()

        x_values = torch.autograd.Variable(
            (torch.zeros(sum(seq_lens) + 2*len(seq_lens), feats.size(2))))
        # x_fw = torch.autograd.Variable(
            # (torch.zeros(sum(seq_lens) + len(seq_lens), feats.size(2))))
        # x_bw = torch.autograd.Variable(
            # (torch.zeros(sum(seq_lens) + len(seq_lens), feats.size(2))))

        if self.cuda:
            x_values = x_values.cuda()
            # x_fw = x_fw.cuda()
            # x_bw = x_bw.cuda()
        start = 0

        for feat, seq_len in zip(feats, seq_lens):
            x_values[start + 1 : start + 1 + seq_len] = feat[:seq_len]
            # x_fw[start: start + seq_len] = feat[:seq_len]
            # x_bw[start+1: start + 1 + seq_len] = feat[:seq_len]
            start += (seq_len + 2)  # add 1 because of column of 0

        cum_seq_lens = [0]
        cum_seq_lens.extend([int(k) for k in torch.cumsum(torch.FloatTensor(seq_lens), 0)])

        for i, seq_len in enumerate(seq_lens):

            fw_seq_hiddens = hidden[i, :seq_len, :hidden.size()[2] // 2]  # Forward hidden states
            bw_seq_hiddens = hidden[i, :seq_len, hidden.size()[2] // 2:]  # Backward hidden states

            fw_logprob = torch.nn.functional.log_softmax(torch.mm(fw_seq_hiddens,
                                                                  x_values.permute(1, 0)), dim=1)

            seq_idx_start = 2*i + cum_seq_lens[i]

            fw_idx_start = seq_idx_start + 2
            fw_logprob_sq = fw_logprob[:, fw_idx_start : fw_idx_start + fw_logprob.size(0)]
            fw_loss += - torch.diag(fw_logprob_sq).mean()

            # backward inference
            bw_logprob = torch.nn.functional.log_softmax(torch.mm(bw_seq_hiddens,
                                                                  x_values.permute(1, 0)), dim=1)
            bw_idx_start = seq_idx_start
            bw_logprob_sq = bw_logprob[:, bw_idx_start : bw_idx_start + fw_logprob.size(0)]
            bw_loss += - torch.diag(bw_logprob_sq).mean()

        return fw_loss / len(seq_lens), bw_loss / len(seq_lens)


class ContrastiveLoss(nn.Module):
    """Standard contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Extracted from: hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, labels):
        """Compute the loss value.

        Args:
            - output1: descriptor coming from one branch of a siamese network.
            - output2: descriptor coming from the other branch of the network.
            - labels: similarity label (0 for similar, 1 for dissimilar).

        Returns:
            Contrastive loss value.

        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - labels).unsqueeze(1) * torch.pow(euclidean_distance, 2) +
                                      (labels) * torch.pow(torch.clamp(self.margin -
                                                                       euclidean_distance, min=0.0),
                                                           2))
        return loss_contrastive


class SBContrastiveLoss(nn.Module):
    """
    Stochastic bidirectional contrastive loss function.

    Based on the one used in the paper "Learning Fashion Compatibility with Bidirectional LSTMs by
    X. Han, Z. Wu, Y. Jiang and L. Davis.

    Args:
        - margin: float, margin value.

    """

    def __init__(self, margin=2.0):
        super(SBContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, descs1, descs2):
        """Forward function.

        Args:
            - descs1: (torch autograd Variable) descriptors of the first branch.
            - descs2: (torch autograd Variable) descriptors of the second branch.
            - labels: (torch autograd Variable) similarity labels (1 for similar items,
              0 for dissimilar).

        Returns:
            autograd.Variable with the stochastic bidirectional contrastive loss value computed as:
            loss = sum_f(sum_k(max(0, m - d(f, v) + d(f, v_k)) +
                   sum_v(sum_k(max(0, m - d(v, f) + d(v, f_k)),
            where sum_X denotes sumatory over X, m is the margin value, d is the distance metric,
            f and v are descs1 and descs2, v_k are non-matching descs2s for a given descs1,
            and f_k are non-matching descs1s for a given descs2.

        """
        loss = autograd.Variable(torch.Tensor([0]))
        zero_comp = autograd.Variable(torch.Tensor([0]))
        if descs1.is_cuda:
            loss = loss.cuda()
            zero_comp = zero_comp.cuda()
        # same_dists = F.cosine_similarity(descs1, descs2)
        # same_dists = paper_dist(descs1, descs2)
        dists = torch.mm(descs1, descs2.permute(1, 0))
        same_dists = torch.diag(dists)
        # Get the loss (compensate the fact that dists includes same_dists)
        desc1_loss = torch.sum(torch.max(zero_comp, self.margin - same_dists.unsqueeze(1) + dists))\
            - self.margin * len(same_dists)
        desc2_loss = torch.sum(torch.max(zero_comp, self.margin - same_dists.unsqueeze(0) + dists))\
            - self.margin * len(same_dists)

        """
        for i, desc1 in enumerate(descs1):
            idxs = [x for x in range(descs2.size()[0]) if x != i]
            # diff_dists = F.cosine_similarity(desc1.repeat(descs2.size()[0] - 1, 1),
            # descs2[idxs, :])
            # diff_dists = paper_dist(desc1.repeat(descs2.size()[0] - 1, 1), descs2[idxs, :])
            diff_dists = dists[:, idxs][i]
            loss += torch.sum(torch.max(
                self.margin - same_dists[i].repeat(descs2.size()[0] -1, 1) + diff_dists, 1)[0])
            import epdb; epdb.set_trace()

        for j, desc2 in enumerate(descs2):
            idxs = [x for x in range(descs1.size()[0]) if x != j]
            # diff_dists = F.cosine_similarity(desc2.repeat(descs1.size()[0] - 1, 1),
            # descs1[idxs, :])
            diff_dists = paper_dist(desc2.repeat(descs1.size()[0] - 1, 1), descs1[idxs, :])
            loss += torch.sum(torch.max(self.margin - same_dists[j].repeat(descs1.size()[0] -1,
                                                                           1) + diff_dists, 1)[0])
            import epdb; epdb.set_trace()
        """
        loss = desc1_loss + desc2_loss

        return loss/(descs1.size()[0]**2)
