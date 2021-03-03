#! /usr/bin/env python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import time

import sys
import os
import glob

log_interval = 10
seed = 1
use_cuda = True
completed_batch =0
completed_test_batch =0


criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    global completed_batch
    train_loss = 0
    correct = 0
    total = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        completed_batch += 1
        
    print ('Train - batches : {}, average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        completed_batch, train_loss/(batch_idx+1), correct, total, 100.*correct/total))
    #print ("Timestamp %d, Iteration %d" % (int(round(time.time()*1000)),completed_batch))
    #print ("Timestamp %d, batchIdx %d" % (int(round(time.time()*1000)),batch_idx))


def test(model, device, test_loader, epoch):
    global completed_test_batch
    global completed_batch
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    completed_test_batch = completed_batch -  len(test_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = criterion(output, target)

            test_loss += loss.item() # sum up batch loss
            _, pred = output.max(1) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            completed_test_batch += 1

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    # Output test info for per epoch
    print('Test - batches: {}, average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
        completed_batch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




def getDatasets():
    train_data_dir = "/tmp/data"
    test_data_dir = "/tmp/data"
    
    transform_train = transforms.Compose([
        transforms.Resize(224),
        #transforms.RandomCrop(self.resolution, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return (torchvision.datasets.CIFAR10(root=train_data_dir, train=True, download=True, transform = transform_train),
            torchvision.datasets.CIFAR10(root=test_data_dir, train=False, download=True, transform = transform_test)
            )



def main(model_type, pretrained = False):

    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset, test_dataset = getDatasets()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, **kwargs)


    print('==> Building model..' + str(model_type))
    # create model from torchvision
    if pretrained:
        print("=> using pre-trained model '{}'".format(model_type))
        model = models.__dict__[model_type](pretrained=True)
    else:
        print("=> creating model '{}'".format(model_type))
        model = models.__dict__[model_type]()
        
    for param in model.parameters():
        param.requires_grad = True  # set False if you only want to train the last layer using pretrained model
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 10)

    #print(model) 
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    epochs = 1
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1, last_epoch=-1)
    
    # Output total iterations info for deep learning insights
    print("Total iterations: %s" % (len(train_loader) * epochs))

    for epoch in range(1, epochs+1):
        print("\nRunning epoch %s ..." % epoch)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()
       
        torch.save(model.state_dict(),  "/tmp/model_epoch_%d.pth"%(epoch))
        
    torch.save(model.state_dict(), "/tmp/model_epoch_final.pth")


if __name__ == '__main__':
    '''
    Supported Resnet models:
    * ResNet-18
    main("resnet18")
    * ResNet-34
    main("resnet34")
    * ResNet-50
    main("resnet50")
    * ResNet-101
    main("resnet101")
    * ResNet-152
    main("resnet152")
    '''
    #main("resnet18", pretrained = True)
    main("resnet18")
