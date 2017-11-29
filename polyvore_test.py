import os
import sys
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets, models
from torch.utils.data import DataLoader
from datasets import PolyvoreDataset
from transforms import ImageTransforms, TextTransforms
from model import FC_BiLSTM_CRF

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
    import epdb; epdb.set_trace();

    kimgs = torch.FloatTensor().cuda()
    ktags = []
    limgs = torch.FloatTensor().cuda()
    ltags = []

    for i, data in enumerate(dataloaders['train']):
        if i < 35:
            # get the inputs
            inputs, labels = data['image'], data['tag']
            for idx, lab in enumerate(labels):
                if lab is 0:
                    kimgs = torch.cat((kimgs, torch.unsqueeze(inputs[idx], 0).cuda()), 0)
                    ktags.append(lab)
                else:
                    limgs = torch.cat((limgs, torch.unsqueeze(inputs[idx], 0).cuda()), 0)
                    ltags.append(lab)
    kl_data = [(kimgs, ktags), (limgs, ltags)]
    lenkls = min(len(kl_data[0][0]), len(kl_data[1][0]))
    ilen = 8
    kltraining_data = []
    for i in range(int(lenkls/ilen)):
        kltraining_data.append((kimgs[i*ilen:(i+1)*ilen], ktags[i*ilen:(i+1)*ilen]))
        kltraining_data.append((limgs[i*ilen:(i+1)*ilen], ltags[i*ilen:(i+1)*ilen]))

    tag_to_ix = {0: 0, 1: 1, START_TAG: 2, STOP_TAG: 3}

    features = models.alexnet(pretrained=True).cuda()

    model = BiLSTM(input_dim, EMBEDDING_DIM, FC_DIM, HIDDEN_DIM)
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
