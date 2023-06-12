#!/usr/bin/env python3
# ============================================================================== #
# ResNet-18(Pretrained) @ CIFAR-10
# Best Accuracy: 95.71%
# 32x32 -> 224x224 -> 10fc
# Powered by xiaolis@outlook.com
# ============================================================================== #
import torch, wandb
from torch import nn
from torch.optim import Adam, SGD
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10

DATA_DIR = './data'
MODEL_DIR = './simo.pth'
BATCH_SIZE = 128
NUM_WORKERS = 4

CONFIG ={ "architecture": "ResNet-18(Pre-trained)",
          "dataset": "CIFAR-10",
          "epochs": 100,
          "batch_size": BATCH_SIZE,
          "version": "0.4",
          "note": "32x32 -> crop -> flip -> 3x3 Conv1 -> 10fc" }

DEVICE = torch.device("cuda")
torch.manual_seed(42)

# ============================================================================== #
def get_data():
    basic = T.Compose([ T.Resize((224,224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=basic)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=basic)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader

def training(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, labs in train_loader:
        imgs = imgs.to(DEVICE)
        labs = labs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss/len(train_loader.dataset)

def evaluation(model, val_dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in val_dataloader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labs.size(0)
            correct += (predicted == labs).sum().item()
    return correct/total

class Simo:

    def __init__(self):
        self.wandb_run = wandb.init( project = "simo", name="resnet18_224", config = CONFIG) ###
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.accs = [0.0]
        self.transfer_learning()

    def _save_best(self):
        torch.save(self.model.state_dict(), MODEL_DIR)
        print(f'This model saved into {MODEL_DIR}')

    def _load_best(self):
        self.model.load_state_dict(torch.load(MODEL_DIR))
        print(f'Loaded model from {MODEL_DIR}')

    def _monitor(self, acc, loss):
        self.wandb_run.log({"acc": acc, "loss": loss}) ###
        if acc > max(self.accs): self._save_best()
        self.accs.append(acc)

    def transfer_learning(self):
        criterion = nn.CrossEntropyLoss()
        trn_data, val_data = get_data()
        self.model.to(DEVICE)
        print(f'Start retrain RetNet-18 on device {DEVICE} ...')

        for p in self.model.parameters(): p.requires_grad = False
        for p in self.model.fc.parameters(): p.requires_grad = True
        optimizer = Adam( self.model.parameters(), 
                          lr = 0.001, 
                          betas = (0.9, 0.999), 
                          weight_decay = 1e-4, 
                          eps = 1e-8 )

        for epoch in range(20):
            loss = training(self.model, trn_data, criterion, optimizer)
            acc = evaluation(self.model, val_data)
            print(f"Epoch {epoch+1}/20 - Loss:{loss:.4f} - Accuracy:{acc:.2%}")
            self._monitor(acc, loss)

        self._load_best()
        for param in self.model.parameters(): param.requires_grad = True
        optimizer = SGD( self.model.parameters(),
                         lr = 0.01, 
                         momentum = 0.9,
                         weight_decay = 1e-4 )

        for epoch in range(80):
            loss = training(self.model, trn_data, criterion, optimizer)
            acc = evaluation(self.model, val_data)
            print(f"Epoch {epoch+1}/80 - Loss:{loss:.4f} - Accuracy:{acc:.2%}")
            self._monitor(acc, loss)

        self.wandb_run.finish() ###

# ============================================================================== #
if __name__ == '__main__': Simo()
