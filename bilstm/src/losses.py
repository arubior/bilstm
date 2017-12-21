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
    dists = torch.cat([torch.dot(a/torch.norm(a), b/torch.norm(b))
                       for a, b in zip(desc1, desc2)])
    if desc1.is_cuda:
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

        X_fw = torch.autograd.Variable(
            (torch.zeros( sum(seq_lens) + len(seq_lens), feats.size(2)))).cuda()
        X_bw = torch.autograd.Variable(
            (torch.zeros( sum(seq_lens) + len(seq_lens), feats.size(2)))).cuda()
        start = 0

        for feat, seq_len in zip(feats, seq_lens):
            X_fw[start: start + seq_len] = feat[:seq_len]
            X_bw[start+1: start + 1 + seq_len] = feat[:seq_len]
            start += (seq_len + 1)  # add 1 because of column of 0

        for i, seq_len in enumerate(seq_lens):

            fw_seq_hiddens = hidden[i, :seq_len, :hidden.size()[2] // 2]  # Forward hidden states
            bw_seq_hiddens = hidden[i, :seq_len, hidden.size()[2] // 2:]  # Backward hidden states

            fw_logprob = torch.nn.functional.log_softmax(torch.mm(fw_seq_hiddens, X_fw.permute(1, 0)))
            fw_logprob_sq = fw_logprob[:, 1 : 1 + fw_logprob.size(0)]
            # fw_loss += (-torch.sum(torch.diag(fw_logprob, 1)) / seq_len)
            fw_loss += (-torch.sum(torch.diag(fw_logprob_sq)) / seq_len)

            # backward inference
            bw_logprob = torch.nn.functional.log_softmax(torch.mm(bw_seq_hiddens, X_bw.permute(1, 0)))
            bw_logprob_sq = bw_logprob[:, :fw_logprob.size(0)]
            # bw_loss += (-torch.sum(torch.diag(bw_logprob)) / seq_len)
            bw_loss += (-torch.sum(torch.diag(bw_logprob_sq)) / seq_len)
        return fw_loss / len(seq_lens), bw_loss / len(seq_lens)
        """
        for i, seq_len in enumerate(seq_lens):

            fw_seq_loss = autograd.Variable(torch.zeros(1))
            bw_seq_loss = autograd.Variable(torch.zeros(1))

            # Create a first and last vector of zeros:
            start_stop = autograd.Variable(torch.zeros(1, feats.size()[2]))

            if self.cuda:
                start_stop = start_stop.cuda()
                fw_seq_loss = fw_seq_loss.cuda()
                bw_seq_loss = bw_seq_loss.cuda()

            fw_seq_feats = torch.cat((feats[i, :seq_len, :], start_stop))
            fw_seq_hiddens = hidden[i, :seq_len, :hidden.size()[2] // 2]  # Forward hidden states

            bw_seq_feats = torch.cat((start_stop, feats[i, :seq_len, :]))
            bw_seq_hiddens = hidden[i, :seq_len, hidden.size()[2] // 2:]  # Backward hidden states

            # torch.mm(fw_seq_hiddens[j].unsqueeze(0),
                      # feats[k].permute(1, 0))

            for j in range(seq_len):
                fw_denom = torch.cat([torch.exp(torch.mm(fw_seq_hiddens[j].unsqueeze(0),
                                                         feats[k].permute(1, 0))).sum()
                                      for k in range(len(seq_lens))]).sum()
                fw_prob = torch.exp(torch.dot(fw_seq_hiddens[j], fw_seq_feats[j + 1])) / fw_denom
                fw_seq_loss += torch.log(fw_prob)

                # new_fw_loss = nn.functional.log_softmax()

                bw_denom = torch.cat([torch.exp(torch.mm(bw_seq_hiddens[j].unsqueeze(0),
                                                         feats[k].permute(1, 0))).sum()
                                      for k in range(len(seq_lens))]).sum()
                bw_prob = torch.exp(torch.dot(bw_seq_hiddens[j], bw_seq_feats[j])) / bw_denom
                bw_seq_loss += torch.log(bw_prob)

            fw_seq_loss = - fw_seq_loss / seq_len
            bw_seq_loss = - bw_seq_loss / seq_len

            fw_loss += fw_seq_loss
            bw_loss += bw_seq_loss

        return fw_loss/len(seq_lens), bw_loss/len(seq_lens)
        """

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

    def forward(self, desc1, desc2):
        """Forward function.

        Args:
            - desc1: (torch autograd Variable) descriptors of the first branch.
            - desc2: (torch autograd Variable) descriptors of the second branch.
            - labels: (torch autograd Variable) similarity labels (1 for similar items,
              0 for dissimilar).

        Returns:
            autograd.Variable with the stochastic bidirectional contrastive loss value computed as:
            loss = sum_f(sum_k(max(0, m - d(f, v) + d(f, v_k)) +
                   sum_v(sum_k(max(0, m - d(v, f) + d(v, f_k)),
            where sum_X denotes sumatory over X, m is the margin value, d is the distance metric,
            f and v are desc1 and desc2, v_k are non-matching desc2s for a given desc1, and f_k are
            non-matching desc1s for a given desc2.

        """
        loss = autograd.Variable(torch.Tensor([0]))
        if desc1.is_cuda:
            loss = loss.cuda()
        # same_dists = F.cosine_similarity(desc1, desc2)
        same_dists = paper_dist(desc1, desc2)
        for i, d1 in enumerate(desc1):
            idxs = [x for x in range(desc2.size()[0]) if x != i]
            # diff_dists = F.cosine_similarity(d1.repeat(desc2.size()[0] - 1, 1), desc2[idxs, :])
            diff_dists = paper_dist(d1.repeat(desc2.size()[0] - 1, 1), desc2[idxs, :])
            loss += torch.sum(torch.max(self.margin - same_dists[i].repeat(desc2.size()[0] -1,
                                                                           1) + diff_dists, 1)[0])

        for j, d2 in enumerate(desc2):
            idxs = [x for x in range(desc1.size()[0]) if x != j]
            # diff_dists = F.cosine_similarity(d2.repeat(desc1.size()[0] - 1, 1), desc1[idxs, :])
            diff_dists = paper_dist(d2.repeat(desc1.size()[0] - 1, 1), desc1[idxs, :])
            loss += torch.sum(torch.max(self.margin - same_dists[j].repeat(desc1.size()[0] -1,
                                                                           1) + diff_dists, 1)[0])

        return loss/desc1.size()[0]
