import numpy as np
import torch.nn as nn
import torch

tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(tensor)
print("Shape: ", tensor.shape)
print("tensor.data: ", tensor.data)
print("tensor.data.numpy(): ", tensor.data.numpy())
print("tensor.data.numpy().shape: ", tensor.data.numpy().shape)

print(np.array(tensor))
print("np.zeros(10): ", np.zeros(10))
print("np.zeros(10).shape: ", np.zeros(10).shape)
