import multiprocessing as mp

def addtolist(x):
    x[0] = 1
    x[2] = -2
    print(type(x))
    print(x[:])

if __name__ == '__main__':
    shared_list = mp.Array('i', 5)
    for i in range(5):
        shared_list[i] = -1
    p = mp.Process(target=addtolist, args=(shared_list,))
    p.start()
    p.join()
    print(shared_list[0])