import torch.multiprocessing as mp
from collections import namedtuple
import numpy as np
import time

Exp = namedtuple('Exp', ['state', 'action', 'reward'])

def reader(value):
    
    while True:
        print("value: ", value.value)
        time.sleep(1)

def writer(value):
    while True:
        value.value = np.random.randint(1, 10)
        time.sleep(1)



def main():
    value = mp.Value('i')
    
    sender_p = mp.Process(target=writer, args=(value,))

    receiver_p = mp.Process(target=reader, args=(value,))

    sender_p.start()
    receiver_p.start()

    sender_p.join()
    receiver_p.join()

if __name__ == '__main__':
    main()