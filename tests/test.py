import numpy as np
import torch.nn as nn
import torch
import torch.distributions as distr

tensor = torch.tensor([7,4,4,8], dtype=torch.float32, requires_grad=True)

print("tensor: ", tensor)

probs = nn.functional.softmax(tensor, dim=0)

print("probs",probs)

distribution = distr.Categorical(probs=probs)

print("distribution",distribution)

sample = distribution.sample()

print("sample",sample)

log_prob = distribution.log_prob(sample)

print("log_prob",log_prob)

entropy = distribution.entropy()

print("entropy",entropy)

