import os 
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torchvision.io import read_image
import matplotlib.pyplot as plt



learning_rate = 1*(1e-3)
batch_size = 64
epochs = 5000

test_dataroot = "C:\\Users\\Frederik Trudslev\\Desktop\\Dataset\\shapes\\test"
train_dataroot = "C:\\Users\\Frederik Trudslev\\Desktop\\Dataset\\shapes\\train"

test_dataset =  torchvision.datasets.ImageFolder(root=test_dataroot,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Grayscale(),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.5), (0.5)),
                                           ]))

train_dataset =  torchvision.datasets.ImageFolder(root=train_dataroot,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Grayscale(),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.5), (0.5)),
                                           ]))



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200*200, 4),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)





def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




    

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Initialize the optimizer function 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


