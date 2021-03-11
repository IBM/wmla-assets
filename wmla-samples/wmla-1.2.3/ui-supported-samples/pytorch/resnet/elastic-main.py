#! /usr/bin/env python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
#from monitor_pth import PytorchMonitor
import torchvision.models as models

import sys
import os

from fabric_model import FabricModel

log_interval = 10
seed = 1
use_cuda = True
completed_batch =0
completed_test_batch =0
# Number of workers
engines_number=16
train_data_dir = '/tmp'
test_data_dir = '/tmp'


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

'''
def train(model, device, train_loader, optimizer, epoch, pytorch_monitor):
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
        
        # Logging/output test info for each batch
        pytorch_monitor.log_train(epoch, batch_idx, len(data), target, loss, output)
        print ("batches :" + str(completed_batch), train_loss/(batch_idx+1))
        print ("Iteration " + str(completed_batch) + ": tag train_accuracy, simple_value " + str(correct*1.0/total))
        print ("Iteration " + str(completed_batch) + ": tag train_loss, simple_value " + str(train_loss*1.0/(batch_idx+1)))
        print ("Timestamp %d, Iteration %d" % (int(round(time.time()*1000)),completed_batch))
        print ("Timestamp %d, batchIdx %d" % (int(round(time.time()*1000)),batch_idx))


def test(model, device, test_loader, epoch, pytorch_monitor):
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
            
            # Logging/output test info for each batch
            pytorch_monitor.log_test(epoch, epoch, batch_idx, len(data), target, test_loss, output)
            print ("Iteration " + str(completed_test_batch) + ": tag test_accuracy, simple_value " + str(correct*1.0/total))
            print ("Iteration " + str(completed_test_batch) + ": tag test_loss, simple_value " + str(test_loss*1.0/(batch_idx+1)))

    test_loss /= len(test_loader.dataset)
    
    # Output test info for per epoch
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

'''


def getDatasets():
    global train_data_dir, test_data_dir    
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

'''
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
'''

def main(model_type, pretrained = False):
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train_dir', type=str, default='', help='input the path of model checkpoint file path')
    parser.add_argument('--weights', type=str, default='', help='input the path of initial weight file')

    args, unknown = parser.parse_known_args()
    
    print("train dir: %s"%args.train_dir)
    path=args.train_dir + "/../"
    sys.path.insert(0,path)
    import pth_parameter_mgr
    global train_data_dir, test_data_dir
    train_data_dir = pth_parameter_mgr.getTrainData(False)
    test_data_dir = pth_parameter_mgr.getTestData(False)
  
    torch.manual_seed(seed)
    
    '''
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset, test_dataset = getDatasets()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=pth_parameter_mgr.getTrainBatchSize(), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=pth_parameter_mgr.getTestBatchSize(), shuffle=True, **kwargs)
    '''

    print('==> Building model..' + str(model_type))
    # create model from torchvision
    if pretrained:
        print("=> using pre-trained model '{}'".format(model_type))
        model = models.__dict__[model_type](pretrained=True)
    else:
        print("=> creating model '{}'".format(model_type))
        model = models.__dict__[model_type]()
        
        '''
        weightfile = getWeightFile(args)
        if weightfile:
            print ("Initial weight file is " + weightfile)
            model.load_state_dict(torch.load(weightfile, map_location=lambda storage, loc: storage))
        '''

    #model.to(device)
    
    optimizer = pth_parameter_mgr.getOptimizer(model)
    epochs = pth_parameter_mgr.getEpoch()
    edt_m = FabricModel(model, getDatasets, F.cross_entropy, optimizer)
    edt_m.train(epochs, pth_parameter_mgr.getTrainBatchSize(), engines_number) 


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
    main("resnet18")
