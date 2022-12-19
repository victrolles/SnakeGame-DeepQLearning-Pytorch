import numpy as np
from collections import namedtuple

def softmax(list):
    exps = np.exp(list)
    return exps / np.sum(exps)

list = [1, 1, 10, 10]
print(softmax(list))

for _ in range(100):
    print(np.random.choice([0, 1, 2, 3], p=softmax(list)))