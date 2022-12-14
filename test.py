import time
import multiprocessing as mp

def counter(num):
    count = 0
    while count < num:
        count+=1

def main():
    print("nbr of cpu:", mp.cpu_count())
    print("starting")
    start = time.perf_counter()


    p1 = mp.Process(target=counter, args=(125000000,))
    p2 = mp.Process(target=counter, args=(125000000,))
    p3 = mp.Process(target=counter, args=(125000000,))
    p4 = mp.Process(target=counter, args=(125000000,))
    p5 = mp.Process(target=counter, args=(125000000,))
    p6 = mp.Process(target=counter, args=(125000000,))
    p7 = mp.Process(target=counter, args=(125000000,))
    p8 = mp.Process(target=counter, args=(125000000,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    print("finished in:", time.perf_counter()-start, "seconds")

if __name__ == '__main__':
    main()