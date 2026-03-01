from convolution_1 import *
import torch

model=ConvNet()
print(model)

inputs=torch.randn(1,3,32,32)
outputs=model(inputs)
print(outputs.shape)
print(torch.sum(outputs, dim=1))
print(torch.max(outputs), torch.min(outputs))