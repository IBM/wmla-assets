#! /usr/bin/env python

from __future__ import print_function
import argparse

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

import sys
import traceback
import random
import argparse
import numpy as np
import os
import glob
import pth_parameter_mgr
import inference_helper

import importlib

from PIL import Image
import requests
import io

use_cuda = True

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

#data_transforms = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224),
#    transforms.ToTensor()
#])
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


#LABELS_URL = 'http://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

#labels = {int(key):value for (key, value)
#          in requests.get(LABELS_URL).json().items()}


def inference(args):
    model = models.__dict__["resnet18"]() 
    model.fc = nn.Linear(512, 10) 
    dictFilePath = args.model + "/model_epoch_final.pth"
    model.load_state_dict(torch.load(dictFilePath, map_location=lambda storage, loc: storage))
    model.eval()

    outputFile = os.path.join(args.output_dir, args.output_file)
    
    input_dir = os.path.expanduser(args.input_dir)
    imagefiles = glob.iglob(input_dir + '/*.*')
    imagefiles = [im_f for im_f in imagefiles
                if im_f.endswith(".jpg") or im_f.endswith(".jpeg") or im_f.endswith(".png")]
    
    prediction = np.ndarray([0, 5], np.float32)
    imagenames = np.ndarray([0], np.int32)
    labeldatas = np.ndarray([0, 5], np.int32)
    
    for imgFile in imagefiles:
       print (imgFile)
       
       with torch.no_grad():
           out = model(image_loader(data_transforms, imgFile))
       _, predicted = torch.max(out.data, 1)
       
       out = torch.nn.functional.softmax(out[0], dim=0)
       print("predictions: ")
       print(out)
       prob, label = torch.topk(out, 5)
       print("top5 probability: ")
       print (prob)
       print("top5 labels: ")
       print (label)

       ls = label.data.numpy()
       #print (ls)
       #for i in ls[0]:
       #    print(labels[i])
       
       #print (prob.data.numpy())
       #print (label.data.numpy())
       
       prediction = np.append(prediction, [prob.data.numpy()], 0)
       imagenames = np.append(imagenames, [imgFile], 0)
       labeldatas = np.append(labeldatas, [label.data.numpy()], 0)
            
       inference_helper.writeClassificationResult2(outputFile,
                                             imagenames, 
                                             prediction,
                                             labeldatas,
                                             prob_thresh = args.prob_thresh,
                                             label_file = args.label_file)
    

def getDatasets(data_dir):
    transform_data = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return datasets.CIFAR10(root=data_dir, train=False, download=False, transform = transform_data)
    
         
def validate(args):
    # create model
    model = models.__dict__["resnet18"]()
    model.fc = nn.Linear(512, 10)
    dictFilePath = args.model + "/model_epoch_final.pth"
    model.load_state_dict(torch.load(dictFilePath, map_location=lambda storage, loc: storage))
    #for p in model.parameters():
    #    print(p.data)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    val_dir = pth_parameter_mgr.getValData(False)
    val_data = getDatasets(val_dir);
    val_loader = torch.utils.data.DataLoader(val_data, 8, shuffle=True, **kwargs)    
    
    criterion = nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_size = 64

    # switch to evaluate mode
    model.eval()

    prediction = np.ndarray([0, 10], np.float32)
    imagenames = np.ndarray([0], np.int32)
    true_label = np.ndarray([0], np.int32)

    counter =0

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if i > 20 :
                break
            #print (i)

            #target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            
            loss = criterion(output, target)
            index = range(i * 64, (i+1)*64)
            pred = output.data
            print(pred)
            print(target.data)

            topk = (1, 10)
            maxk = max(topk)
            batch_size = target.size(0)


            predicted = output.data.squeeze().cpu().numpy()
            label = target.type(torch.LongTensor).cpu().numpy()

            prediction = np.append(prediction, predicted, 0)
            imagenames = np.append(imagenames, index, 0)
            true_label = np.append(true_label, target.data.numpy(), 0)


    inference_helper.writeClassificationResult(os.path.join(args.output_dir, args.output_file),
                                             imagenames, prediction, label_file = args.label_file, 
                                             ground_truth = true_label)

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res




                                         


                                         
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='', help='input the path of model checkpoint file')
    parser.add_argument('--input_dir', type=str, default='/tmp/', help='input the path of the input files')
    parser.add_argument('--output_dir', type=str, default='/tmp/', help='input the path of output file')
    parser.add_argument('--output_file', type=str, default='inference.json', help='input the name of the output file')
    parser.add_argument('--prob_thresh', type=float, default='0', help='input the probability threshold')
    parser.add_argument('--validate', type=str, default='', help='specify whether it is for validation')
    parser.add_argument('--label_file', type=str, default='', help='specify the label file')

    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # initialize the checkpoint class
    # Create Model
    torch.manual_seed(1)
    
    if args.validate == 'true':
        validate(args)
    else:
        inference(args);


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception as e:
        print(e)
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
                                         
