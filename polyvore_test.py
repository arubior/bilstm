import os
import sys
import time
import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets, models
from torch.utils.data import DataLoader
from datasets import PolyvoreDataset
from transforms import ImageTransforms, TextTransforms

writer = SummaryWriter() 

torch.manual_seed(1)

########################################################
# DATA LOADER
# ~~~~~~~~~~~

itr = {'train': ImageTransforms(227, 5, 224),
       'test': ImageTransforms(224)}

ttr = TextTransforms()

img_train_tf = lambda x: torchvision.transforms.ToTensor()(itr['train'].random_crop(
                     itr['train'].random_rotation(itr['train'].random_horizontal_flip(
                     itr['train'].resize(x)))))
img_test_val_tf = lambda x: torchvision.transforms.ToTensor()(itr['test'].resize(x))

txt_train_tf = lambda x: ttr.random_delete(ttr.normalize(x))
txt_test_val_tf = lambda x: ttr.normalize(x)

img_transforms = {'train': img_train_tf,
                  'test': img_test_val_tf,
                  'val': img_test_val_tf}

txt_transforms = {'train': txt_train_tf,
                  'test': txt_test_val_tf,
                  'val': txt_test_val_tf}

img_dir = 'datasets/polyvore/images'
json_dir = 'datasets/polyvore/label'
json_files = {'train': 'train_no_dup.json',
              'test': 'test_no_dup.json',
              'val': 'valid_no_dup.json'}

image_datasets = {x: PolyvoreDataset(os.path.join(json_dir, json_files[x]),
                                     img_dir,
                                     img_transform=img_transforms[x],
                                     txt_transform=txt_transforms[x])
                  for x in ['train', 'test', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                            batch_size=4, shuffle=True, num_workers=1)
               for x in ['train', 'test', 'val']}


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class FC_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim, fc_dim, hidden_dim):
        super(FC_BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc_dim = fc_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        """
        self.word_embeds = nn.Sequential(vocab_size, embedding_dim)
        self.word_embeds = nn.Sequential(
            nn.conv2d(n_channels, 224, 3),
            nn.maxpool2d(2),
            nn.conv2d(32, 64, 3),
            nn.maxpool2d(2))

        self.classifier = nn.sequential(
            nn.linear(h*w*64/4, 4096),
            nn.relu(inplace=true),
            nn.linear(4096, 1024),
            nn.relu(inplace=true),
            nn.linear(1024, n_classes),
            nn.softmax())
        self.fc = nn.Sequential(nn.Linear(embedding_dim, 20),
                                nn.Linear(20, fc_dim))
        """
        self.features = models.alexnet(pretrained=True).cuda()
        self.lstm = nn.LSTM(1000, hidden_dim // 2,
                            num_layers=1, bidirectional=True).cuda()

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).cuda()

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).cuda()

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)).cuda(),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)).cuda())

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.).cuda()
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas).cuda()

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        alex_feats = self.features(sentence)
        lstm_out, self.hidden = self.lstm(alex_feats.unsqueeze(1), self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0])).cuda()
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]).cuda(), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars).cuda()
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
FC_DIM = 10
HIDDEN_DIM = 4

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data['image'], data['tag']
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

if __name__ == '__main__':

    kimgs = torch.FloatTensor().cuda()
    ktags = []
    limgs = torch.FloatTensor().cuda()
    ltags = []

    for i, data in enumerate(dataloaders['train']):
        if i < 35:
            # get the inputs
            print(i)
            images, texts = data
            images = torch.stack(images)
            for im in images:
                import epdb; epdb.set_trace();
                out = torchvision.utils.make_grid(images)
                writer.add_image('Image', out, i) # Tensor
            """
            for idx, lab in enumerate(labels):
                if lab is 0:
                    kimgs = torch.cat((kimgs, torch.unsqueeze(inputs[idx], 0).cuda()), 0)
                    ktags.append(lab)
                else:
                    limgs = torch.cat((limgs, torch.unsqueeze(inputs[idx], 0).cuda()), 0)
                    ltags.append(lab)
            """
    writer.close()
    import epdb; epdb.set_trace();
    kl_data = [(kimgs, ktags), (limgs, ltags)]
    lenkls = min(len(kl_data[0][0]), len(kl_data[1][0]))
    ilen = 8
    kltraining_data = []
    for i in range(int(lenkls/ilen)):
        kltraining_data.append((kimgs[i*ilen:(i+1)*ilen], ktags[i*ilen:(i+1)*ilen]))
        kltraining_data.append((limgs[i*ilen:(i+1)*ilen], ltags[i*ilen:(i+1)*ilen]))

    tag_to_ix = {0: 0, 1: 1, START_TAG: 2, STOP_TAG: 3}

    model = FC_BiLSTM_CRF(tag_to_ix, EMBEDDING_DIM, FC_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.cuda()

    # Check predictions before training
    print("Predictions before training:")
    print(model.forward(autograd.Variable(kltraining_data[0][0].cuda())))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    numepochs = 5000
    tic = time.time()
    for epoch in range(numepochs):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in kltraining_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Variables of word indices.
            targets = torch.LongTensor([tag_to_ix[t] for t in tags]).cuda()

            # Step 3. Run our forward pass.
            # neg_log_likelihood = model.neg_log_likelihood(autograd.Variable(sentence), targets)
            neg_log_likelihood = model.neg_log_likelihood(autograd.Variable(sentence), targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            neg_log_likelihood.backward()
            optimizer.step()

            writer.add_scalar('data/loss', neg_log_likelihood.data[0], epoch) # data grouping by `slash` 
        sys.stdout.write("Epoch %i/%i took %f seconds\r" % (epoch, numepochs, time.time() - tic))
        sys.stdout.flush()

    # Check predictions after training
    print("Predictions after training:")
    print(model(autograd.Variable(kltraining_data[0][0])))
    writer.close()
