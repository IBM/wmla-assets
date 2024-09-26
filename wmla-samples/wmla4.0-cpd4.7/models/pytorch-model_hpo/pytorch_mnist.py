from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import datetime
import os
from emetrics import EMetrics
import DataParallel
import json

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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

root_folder=os.getenv("DATA_DIR")
output_model_folder = os.environ["RESULT_DIR"]
output_model_path = os.path.join(output_model_folder,"model")
output_model_path = os.path.join(output_model_path,"trained_model.pt")

import  os
print (os.getcwd())

try:
    with open("config.json", 'r') as f:
        json_obj = json.load(f)
    
    learning_rate = float(json_obj["learning_rate"])
    print('*****learning_rate=', learning_rate)
    #Add for test
    qq = float(json_obj["qq"])
    qq2 = int(json_obj["qq2"])
    qq3 = str(json_obj["qq3"])
    qq4 = int(json_obj["qq4"])
    epsilon = float(json_obj["epsilon"])
    print('*****qq=', qq)
    print('*****qq2=', qq2)
    print('*****qq3=', qq3)
    print('*****qq4=', qq4)
    print('*****epsilon=', epsilon)
except:
    print('failed to get learning rate')
    pass

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root_folder, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root_folder, train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#model = Net()
model = nn.DataParallel(Net())
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            em.record(EMetrics.TEST_GROUP,batch_idx*len(data),{'loss': loss.item()})
            

def test(epoch,em):
    model.eval()
    
    # HPO - start
    training_out =[]
    
    test_loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            #HPO
            i=i+1
            out = {'steps':i}
            out['loss'] = float(F.nll_loss(output, target).item() )
            print (out)
            training_out.append(out)

    test_loss /= len(test_loader.dataset)
    # accuracy = correct / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #em.record(EMetrics.TEST_GROUP,epoch,{'loss': test_loss, 'accuracy': accuracy})
    
    with open('{}/val_dict_list.json'.format(os.environ['RESULT_DIR']), 'w') as f:
        json.dump(training_out, f)
    # HPO - end 


with EMetrics.open() as em:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch,em)

torch.save(model.state_dict(),output_model_path)
