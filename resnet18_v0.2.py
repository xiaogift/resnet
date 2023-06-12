#!/usr/bin/env python
# ============================================================================== #
# Resnet-18 @ CIFAR-10
# ~ 80% TOP1 Accuracy
# 32x32 -> crop -> color -> gray -> cutout -> noise -> blur -> rotation -> flip -> 10fc
# Powered by xiaolis@outlook.com 202305
# ============================================================================== #
import torch, random, time, os, PIL
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torchvision.datasets import CIFAR10, STL10
from torch.utils.data import DataLoader
from torchvision.models import resnet18

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATADIR = './data'

# ============================================================================== #
class Cutout(object):
    def __init__(self, length):
        self.length = length
    def __call__(self, image):
        width, height = image.size
        upper_left_x = random.randint(0, width - self.length)
        upper_left_y = random.randint(0, height - self.length)
        lower_right_x = upper_left_x + self.length
        lower_right_y = upper_left_y + self.length
        mask = PIL.Image.new('RGB', (self.length, self.length), (0, 0, 0))
        image.paste(mask, (upper_left_x, upper_left_y, lower_right_x, lower_right_y))
        return image

class AddNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        to_tensor = transforms.ToTensor()
        to_image = transforms.ToPILImage()  
        tensor_image = to_tensor(image)
        noise = torch.randn_like(tensor_image) * self.std + self.mean
        noisy_tensor_image = tensor_image + noise
        noisy_image = to_image(noisy_tensor_image)
        return noisy_image

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma=sigma
    def __call__(self, x):
        sigma=random.uniform(self.sigma[0], self.sigma[1])
        x=x.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return x

def augs_options(img_sz):
    options = [ transforms.RandomCrop(img_sz, padding=4),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([Cutout(4)], p=0.2),
                transforms.RandomApply([AddNoise(mean=0.0, std=0.1)], p=0.8),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomRotation((-15, 15)),
                transforms.RandomHorizontalFlip() ]
    return [[options[:i] + options[i+1:] for i in range(len(options))]+[options]]

class xDataLoader:
    def __init__(self, data_dir=DATADIR, num_workers=0):
        self.dir = data_dir
        self.nwk = num_workers

    def _set_cifar10(self):
        self.btz = 128
        self.img_sz = 32
        self.base = [ transforms.ToTensor(),
                      transforms.Normalize( mean = (0.49139968, 0.48215841, 0.44653091), 
                                            std = (0.24703223, 0.24348513, 0.26158784))]

    def _set_stl10(self):
        self.btz = 64
        self.img_sz = 96
        self.base = [ transforms.ToTensor(),
                      transforms.Normalize( mean = (0.4469, 0.4400, 0.4069), 
                                            std = (0.2603, 0.2566, 0.2713))]

    def _loader(self, dataset, batchsize, shf=True):
        return DataLoader( dataset=dataset, 
                           batch_size=batchsize,
                           shuffle=shf, 
                           pin_memory=True, 
                           drop_last=True,
                           num_workers=self.nwk )

    def get_cifar10(self, augs=[]):
        self._set_cifar10()
        trans = transforms.Compose(augs+self.base)
        train = CIFAR10(root=self.dir, train=True, transform=trans, download=True)
        valid = CIFAR10(root=self.dir, train=False, transform=transforms.Compose(self.base), download=True)
        return self._loader(train,self.btz), self._loader(valid,100,shf=False)

    def get_stl10(self, augs=[]):
        self._set_stl10()
        trans = transforms.Compose(augs+self.base)
        train=STL10(root=self.dir, split='train', transform=trans, download=True)
        valid=STL10(root=self.dir, split='test', transform=transforms.Compose(self.base), download=True)
        return self._loader(train,self.btz), self._loader(valid,100,shf=False)

# ============================================================================== #
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder.fc = nn.Linear(512, 10)
    def forward(self, x): return self.encoder(x)

# ============================================================================== #
def save_results(path, content):
    with open(path, 'a') as f:
        if os.stat(path).st_size == 0: f.write(','.join(map(str, content)))
        else: f.write('\n'); f.write(','.join(map(str, content)))

def evaluate(model, dataloader):
    model.eval()
    no_correct, total = 0, 0
    with torch.no_grad():
        for d in dataloader:
            imgs, labs = d[0].to(DEVICE), d[1].to(DEVICE)
            _, predict = model(imgs).max(1)
            total += labs.size(0)
            no_correct += (predict == labs).sum().item()
    return no_correct/total

def train(model, dataloaders, filename):
    model.train()
    trn_loss, val_acc = [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    for epoch in range(100):
        running_loss = 0.0
        for i, (img, lab) in enumerate(dataloaders[0]):
            inputs, labels = img.to(DEVICE), lab.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = "{:.4f}".format(evaluate(model, dataloaders[1]))
        loss = "{:.4f}".format((running_loss/len(dataloaders[0])))
        print(f'[{epoch+1}/100] Loss: {loss}, Accuracy.: {acc}')
        trn_loss.append(loss); val_acc.append(acc)
    save_results(f'Los.{filename}.txt', trn_loss)
    save_results(f'Acc.{filename}.txt', val_acc)

def ablation(img_sz):
    # for augs in augs_options(img_sz)[0]:
    for augs in augs_options(img_sz)[0][3:]:
        start_time = str(time.time())
        extra_augs = data.get_cifar10(augs)
        print(f'{start_time}\n{str(augs)}\n', '*'*30)
        train(ResNet18().to(DEVICE), extra_augs, start_time)

# ============================================================================== #
if __name__ == '__main__':
    data = xDataLoader()
    print('CIFAR-10 training ....'); ablation(32)
    print('STL-10 training ....'); ablation(96)
