#! /usr/bin/env python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pth_parameter_mgr
import torchvision
from torchvision import datasets, transforms
from monitor_pth import PytorchMonitor
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
        
    # Output train info for per epoch
    print ('Train - batches : {}, average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        completed_batch, train_loss/(batch_idx+1), correct, total, 100.*correct/total))
    print ("Iteration " + str(completed_batch) + ": tag train_accuracy, simple_value " + str(correct*1.0/total))
    print ("Iteration " + str(completed_batch) + ": tag train_loss, simple_value " + str(train_loss*1.0/(batch_idx+1)))


#def test(model, device, test_loader, epoch, pytorch_monitor):
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
    print ("Iteration " + str(completed_batch) + ": tag test_accuracy, simple_value " + str(test_acc/100.0))
    print ("Iteration " + str(completed_batch) + ": tag test_loss, simple_value " + str(test_loss))


def getDatasets():
    train_data_dir = pth_parameter_mgr.getTrainData(False)
    test_data_dir = pth_parameter_mgr.getTestData(False)
    
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

    return (torchvision.datasets.CIFAR10(root=train_data_dir, train=True, download=False, transform = transform_train),
            torchvision.datasets.CIFAR10(root=test_data_dir, train=False, download=False, transform = transform_test)
            )


def getWeightFile(args):
    initial_weight_dir_str = args.weights.strip()

    if not initial_weight_dir_str:
        return ""

    if not os.path.exists(initial_weight_dir_str):
        return ""

    input_weight_dir = os.path.expanduser(initial_weight_dir_str)
    allfiles = glob.iglob(input_weight_dir + '/*.*')
    weightfiles = [wt_f for wt_f in allfiles if wt_f.endswith(".pth")]

    weightfile = ""

    for wtfile in weightfiles:
        if wtfile.startswith("model_epoch_final"):
            return wtfile

        weightfile = wtfile

    return weightfile


def main(model_type, pretrained = False):
    parser = argparse.ArgumentParser(description='PyTorch Resnet Cifar10 Example')
    parser.add_argument('--train_dir', type=str, default='', help='input the path of model checkpoint file path')
    parser.add_argument('--weights', type=str, default='', help='input the path of initial weight file')

    args, unknown = parser.parse_known_args()

    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset, test_dataset = getDatasets()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=pth_parameter_mgr.getTrainBatchSize(), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=pth_parameter_mgr.getTestBatchSize(), shuffle=True, **kwargs)


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

    weightfile = getWeightFile(args)
    if weightfile:
        print ("Initial weight file is " + weightfile)
        model.load_state_dict(torch.load(weightfile, map_location=lambda storage, loc: storage))

    #print(model) 
    model.to(device)
    optimizer = pth_parameter_mgr.getOptimizer(model)
    if isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError('The specified optimizer LBFGS (Limited-memory BFGS) is not supported currently!')

    epochs = pth_parameter_mgr.getEpoch()
    scheduler = pth_parameter_mgr.getLearningRate(optimizer)
    isSchedulerReduceLROnPlateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    # Output total iterations info for deep learning insights
    print ("epochs: %s " % epochs)
    print ("len(train_loader): %s " % len(train_loader) )
    print("Total iterations: %s" % (len(train_loader) * epochs))

    for epoch in range(1, epochs+1):
        print("\nRunning epoch %s ..." % epoch)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        if scheduler and not isSchedulerReduceLROnPlateau:
            scheduler.step() 
            
        '''
        # TODO: need special handling for scheduler "ReduceLROnPlateau", since validation result is required by the scheduler step function       
        if scheduler and isSchedulerReduceLROnPlateau:
            val_loss = validate(...)
            # Note that step should be called after validate()
            scheduler.step(val_loss)
        '''
        
        torch.save(model.state_dict(), args.train_dir + "/model_epoch_%d.pth"%(epoch))
        
    torch.save(model.state_dict(), args.train_dir + "/model_epoch_final.pth")


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
