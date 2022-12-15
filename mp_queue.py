import torch.multiprocessing as mp
from collections import namedtuple
import numpy as np
import time

Exp = namedtuple('Exp', ['state', 'action', 'reward'])

def add_queue(q):
    
    while True:
        # state = np.zeros(10)
        # action = np.random.randint(0, 4)
        # reward = np.random.randint(0, 10)
        # exp = Exp(state, action, reward)
        q.put(np.random.randint(0, 10))
        time.sleep(1)

def print_queue(q):
    elements = []
    while True:
        elements.append(q.get())

        print("elements: ", elements)

def main():
    q = mp.Queue(maxsize=10)
    
    sender_p1 = mp.Process(target=add_queue, args=(q,))
    sender_p2 = mp.Process(target=add_queue, args=(q,))

    receiver_p = mp.Process(target=print_queue, args=(q,))

    sender_p1.start()
    sender_p2.start()
    receiver_p.start()

    sender_p1.join()
    sender_p2.join()
    receiver_p.join()

if __name__ == '__main__':
    main()