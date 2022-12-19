import numpy as np
import torch.nn as nn
import torch

m = nn.Softmax(dim=0)
input = torch.randn(3)
print("input: ", input)
output = m(input)
print("output: ", output)