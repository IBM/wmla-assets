# based on commit d7749e6
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import argparse
import sys
import os

from fabric_model import FabricModel
from edtcallback import EDTLoggerCallback

data_dir = os.environ.get("DATA_DIR", "/tmp")
result_dir = os.environ.get("RESULT_DIR", "/tmp")
print("data_dir=%s, result_dir=%s" % (data_dir, result_dir))
os.makedirs(data_dir, exist_ok=True)

MAX_NUM_WORKERS=16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p =0.25, training=self.training)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.softmax(self.fc2(x), dim=1)
        return x


def getDatasets():
    return (datasets.MNIST(data_dir, train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
            datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
            )

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
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
    args, unknown = parser.parse_known_args()
    print(args)
    print(unknown)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95)

    edt_m = FabricModel(model, getDatasets, F.cross_entropy, optimizer, driver_logger=EDTLoggerCallback())
    edt_m.train(args.epochs, args.batch_size, MAX_NUM_WORKERS)

if __name__ == '__main__':
    main()