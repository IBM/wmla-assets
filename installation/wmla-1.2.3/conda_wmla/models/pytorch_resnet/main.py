################################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2021 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
################################################################################

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

data_dir = os.environ.get("DATA_DIR", "/tmp")
result_dir = os.environ.get("RESULT_DIR", "/tmp")
print("data_dir=%s, result_dir=%s" % (data_dir, result_dir))
os.makedirs(data_dir, exist_ok=True)

output_model_path = os.path.join(result_dir, "model", "trained_model.pt")

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_loader, optimizer, epoch):
    correct = 0
    trained_data_len = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        trained_data_len += len(data)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.6f}\tLoss: {:.6f}'.format(
            epoch, trained_data_len, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), correct * 1.0 / trained_data_len, loss.item()))

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = criterion(output, target)

            test_loss += loss.item()  # sum up batch loss
            _, pred = output.max(1)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

def get_datasets():
    transform_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(self.resolution, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return (torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train),
            torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
            )


def main(model_type, pretrained=False):
    parser = argparse.ArgumentParser(description='PyTorch Resnet Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    print(args)

    use_cuda = not args.no_cuda

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {}

    train_dataset, test_dataset = get_datasets()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if use_cuda:
        print("Let's use {} gpus".format(str(torch.cuda.device_count())))

    print('==> Building model..' + str(model_type))
    # create model from torchvision
    if pretrained:
        print("=> using pre-trained model '{}'".format(model_type))
        model = models.__dict__[model_type](pretrained=True)
    else:
        print("=> creating model '{}'".format(model_type))
        model = models.__dict__[model_type]()

    # multi-GPUs data
    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    if isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError('The specified optimizer LBFGS (Limited-memory BFGS) is not supported currently!')

    # Output total iterations info for deep learning insights
    #print("Total iterations: %s" % (len(train_loader) * args.epochs))

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    duration = (time.time() - start_time) / 60
    print("Train finished. Time cost: %.2f minutes" % duration)

    torch.save(model.state_dict(), output_model_path)
    print("Model saved in path: %s" % output_model_path)


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
