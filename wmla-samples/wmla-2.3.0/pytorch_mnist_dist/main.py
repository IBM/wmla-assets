################################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2020 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
################################################################################

from __future__ import division, print_function

import argparse

import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import time
import os

data_dir = os.environ.get("DATA_DIR", "/tmp")
result_dir = os.environ.get("RESULT_DIR", "/tmp")
print("data_dir=%s, result_dir=%s" % (data_dir, result_dir))
os.makedirs(data_dir, exist_ok=True)

output_model_path = os.path.join(result_dir, "model", "trained_model.pt")


class Average(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)


class Trainer(object):
    def __init__(self, net, optimizer, train_loader, test_loader, device):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc))

    def train(self):
        train_loss = Average()
        train_acc = Accuracy()

        self.net.train()

        for data, label in self.train_loader:
            data = data.to(self.device)
            label = label.to(self.device)

            output = self.net(data)
            loss = F.cross_entropy(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, label)

        return train_loss, train_acc

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()

        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = F.cross_entropy(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)

        return test_loss, test_acc


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def get_dataloader(root, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(
        root, train=True, transform=transform, download=True)
    sampler = DistributedSampler(train_set)

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler)

    test_loader = data.DataLoader(
        datasets.MNIST(root, train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=False)

    return train_loader, test_loader


def run(args):
    use_cuda = not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    if use_cuda:
        print("Using DistributedDataParallel")
        print("Let's use {} gpus per worker".format(str(torch.cuda.device_count())))
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(Net()).to(device)
        else:
            net = Net().to(device)

        net = nn.parallel.DistributedDataParallel(net)
    else:
        print("Using DistributedDataParallelCPU")
        net = Net().to(device)
        net = nn.parallel.DistributedDataParallelCPU(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_loader, test_loader = get_dataloader(args.root, args.batch_size)

    start_time = time.time()

    trainer = Trainer(net, optimizer, train_loader, test_loader, device)
    trainer.fit(args.epochs)

    duration = (time.time() - start_time) / 60
    print("Train finished. Time cost: %.2f minutes" % duration)

    if int(os.environ['RANK']) == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(net.state_dict(), output_model_path)
        print("Model saved in path: %s" % output_model_path)


def init_process(args):
    print('tcp://' + os.environ['MASTER_ADDR'] + ':' + os.environ['MASTER_PORT'])
    print("WORLD_SIZE=" + os.environ['WORLD_SIZE'] + ", RANK=" + os.environ['RANK'])
    # print(os.environ['GLOO_SOCKET_IFNAME'])

    distributed.init_process_group(
        backend=args.backend,
        # init_method=args.init_method,
        init_method='tcp://' + os.environ['MASTER_ADDR'] + ':' + os.environ['MASTER_PORT'],
        # rank=args.rank,
        rank=int(os.environ['RANK']),
        # world_size=args.world_size)
        world_size=int(os.environ['WORLD_SIZE']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='Name of the backend to use.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default=data_dir)
    parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--train_dir', type=str, default='')
    parser.add_argument('--train_checkpoint_dir', type=str, default='')
    parser.add_argument('--train_model_save_dir', type=str, default='')
    parser.add_argument('--train_log_dir', type=str, default='')
    args = parser.parse_args()
    print(args)

    init_process(args)
    run(args)


if __name__ == '__main__':
    main()
