import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from datasets import KiwisLlamasDataset
from utils import imshow
from tensorboardX import SummaryWriter

plt.ion() # interactive mode

writer = SummaryWriter() 

data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(256),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}


########################################################
# DATA LOADER
# ~~~~~~~~~~~

data_dir = 'dataset'

image_datasets = {x: KiwisLlamasDataset('%s.txt' % x,
                                         data_dir,
                                         transform=data_transforms[x])
                  for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                            batch_size=4, shuffle=True, num_workers=1)
                for x in ['train', 'val']}

class_names = {0: 'kiwi', 1: 'llama'}

use_gpu = torch.cuda.is_available()

# Observe data:
"""
for i_batch, sample_batched in enumerate(dataloaders['val']):
    print(i_batch, sample_batched['image'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        out = torchvision.utils.make_grid(sample_batched['image'])
        imshow(out, title=[class_names[x] for x in sample_batched['tag']])
        plt.show()
	break
"""


######################################################################
# Model definition
# ----------------

class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()


        model = models.alexnet(pretrained=True)
        import epdb; epdb.set_trace();
        self.features = nn.Sequential(
            nn.Conv2d(3, 224, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(224, 111, 3),
            nn.MaxPool2d(2))

        self.classifier = nn.Sequential(
            nn.Linear(111*54*54, 4096),
            nn.ReLU(inplace=true),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=true),
            nn.Linear(1024, n_classes),
            nn.SoftMax())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)



# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out






######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data['image'], data['tag']

                out = torchvision.utils.make_grid(inputs)
                writer.add_image('Image', out, epoch) # Tensor

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / float(dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            writer.add_scalar('data/loss', epoch_loss, epoch) # data grouping by `slash` 
            writer.add_scalar('data/accuracy', epoch_acc, epoch) # data grouping by `slash` 

            for name, param in model.named_parameters(): 
                 writer.add_histogram(name, param, epoch) 

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

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

    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #

    model_ft = models.resnet18(pretrained=True)
    model_ft = CNN()
    model_ft = ConvNet(2)

    import epdb; epdb.set_trace();
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft = model

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #
    
    import epdb; epdb.set_trace();
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=1)

    ######################################################################
    #

    visualize_model(model_ft)

    # if you want to show the input tensor, set requires_grad=True
    data = next(iter(dataloaders['val']))
    res = model_ft(Variable(data['image']))
    writer.add_graph(model_ft, res)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

