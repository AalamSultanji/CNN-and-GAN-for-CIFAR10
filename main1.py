import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import os
import torchvision
import torchvision.transforms as transforms
import convolution_1


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Transforming and loading the CIFAR-10 dataset
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

#defining the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#loading the model
net=convolution_1.ConvNet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

#defining the training and testing algorithms
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_losses.append(train_loss/len(trainloader))
    train_accs.append(100.*correct/total)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_losses.append(test_loss/len(testloader))
    test_accs.append(100.*correct/total)


if __name__ == '__main__':
    for epoch in range(50):
        train(epoch)
        test(epoch)
    
    # Save results to files instead of plotting as there was some issue with matplotlib on the cluster
    print("Training complete!")
    print(f"Final training accuracy: {train_accs[-1]:.2f}%")
    print(f"Final test accuracy: {test_accs[-1]:.2f}%")
    
    # Save metrics to a text file
    with open('results.txt', 'w') as f:
        f.write("Training Results\n")
        f.write(f"Final training accuracy: {train_accs[-1]:.2f}%\n")
        f.write(f"Final test accuracy: {test_accs[-1]:.2f}%\n")
        f.write(f"Final training loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final test loss: {test_losses[-1]:.4f}\n")
