import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet,self).__init__()
        '''
        Architecture:
        2 convolutional layers, each followed by a max pooling layer and using RelU as activation functions.
        2 dense layers, with softmax in the output layer. 
        Total trainable layers=4.
        Input: 32x32x3 image, taken from CIFAR-10 dataset.
        Output: probabilities for each of the 10 classes.

        '''
        #first block: conv layer+maxpooling
        #input size=32x32x3, outputs size = 32x32x32
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1, bias=False)
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)

        #second block: conv layer+maxpooling
        #input size=32x32x32, output size=32x32x64
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1, bias=False)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)

        #fully connected layers
        self.flatten=nn.Flatten()
        self.fc=nn.Linear(in_features=8*8*64, out_features=num_classes)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        #first conv layer
        x=F.relu(self.conv1(x))
        x=self.pool1(x)
        
        #second conv layer
        x=F.relu(self.conv2(x))
        x=self.pool2(x)

        #fully connected layers
        x=self.flatten(x)
        x=self.fc(x)
        return x


        
