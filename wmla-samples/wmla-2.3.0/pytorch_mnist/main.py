from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import sys
import os
from emetrics import EMetrics

data_dir = os.environ.get("DATA_DIR", "/tmp")
result_dir = os.environ.get("RESULT_DIR", "/tmp")
print("data_dir=%s, result_dir=%s" % (data_dir, result_dir))
os.makedirs(data_dir, exist_ok=True)

output_model_path = os.path.join(result_dir, "model")
os.makedirs(output_model_path, exist_ok=True)
output_model_pt = os.path.join(output_model_path, "trained_model.pt")
output_model_onnx = os.path.join(output_model_path, "trained_model.onnx")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        # input_size=x.size()
        # output_size=output.size()
        # print("\t\tForward: input size", x.size(), "output size", output.size())
        return output


def train(args, model, device, train_loader, optimizer, epoch, em):
    trained_data_len = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("\tTrain: input size", data.size(), "output_size", output.size())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        trained_data_len += len(data)
        if (batch_idx % args.log_interval == 0) or (batch_idx == len(train_loader) - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, trained_data_len, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            em.record(EMetrics.TEST_GROUP, batch_idx * len(data), {'loss': loss.item()})


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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

    args, unknown = parser.parse_known_args()
    print(sys.path)
    print("known arguments: ", args)
    print("unknown arguments", unknown)
    print("torch version: %s" % torch.__version__)

    use_cuda = not args.no_cuda

    if use_cuda:
        print("Let's use {} gpus".format(str(torch.cuda.device_count())))
        # for onnx
        torch.cuda.manual_seed(args.seed)
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        print("Let's use cpu")
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # multi-GPUs data
    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(Net()).to(device)
    else:
        model = Net().to(device)

    print("Model parameters are on cuda") if all(p.is_cuda for p in model.parameters()) else print("Model parameters are on cpu")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = time.time()
    with EMetrics.open() as em:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, em)
            test(args, model, device, test_loader)
    duration = (time.time() - start_time) / 60
    print("Train finished. Time cost: %.2f minutes" % duration)

    torch.save(model.state_dict(), output_model_pt)
    print("Model saved in path: %s" % output_model_pt)

    # export onnx model
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    if type(model) is nn.DataParallel:
        # torch.nn.DataParallel is not supported by ONNX exporter, please use 'attribute' module to unwrap model from torch.nn.DataParallel
        model = model.module
    torch.onnx.export(model, dummy_input, output_model_onnx, export_params=True)

    print("Onnx Model saved in path: %s" % output_model_onnx)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
