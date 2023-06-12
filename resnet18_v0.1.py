#!/usr/bin/env python
# ============================================================================== #
# Resnet-18 @ CIFAR-10
# ~ 85% TOP1 Accuracy
# 32x32 -> crop -> flip -> 10fc
# Powered by xiaolis@outlook.com 202305
# ============================================================================== #
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

datadir = './data'
totalepoch = 100
batchsize = 256

# ============================================================================== #
class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.encoder = torchvision.models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder.fc = nn.Linear(512, 10)
    def forward(self, x): return self.encoder(x)

def dataload():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    accuracy = 100.0 * total_correct / total_samples
    return accuracy

def train(model, trainloader, validloader, criterion, optimizer, device):
    model.train()
    trn_loss, val_acc = [], []
    for epoch in range(totalepoch):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, validloader, device)
        loss = running_loss / len(trainloader)
        print('[%d/%d] loss: %.3f, acc.: %.2f%%' % (epoch + 1, totalepoch, loss, acc))
        trn_loss.append(loss); val_acc.append(acc)
    with open('loss.txt', 'a') as f: f.write(','.join(map(str, trn_loss))); f.write('\n')
    with open('acc.txt', 'a') as f: f.write(','.join(map(str, val_acc))); f.write('\n')


# ============================================================================== #
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = dataload()
    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    train(model, trainloader, testloader, criterion, optimizer, device)
