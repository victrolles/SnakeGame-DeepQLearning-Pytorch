import multiprocessing as mp
import time

def count(number):
    time.sleep(1)
    print('Starting process:', mp.current_process().name)
    print("number:", number)

def main():
    start = time.time()

    p1 = mp.Process(target=count, args=(125000000,))
    p2 = mp.Process(target=count, args=(125000000,))
    p3 = mp.Process(target=count, args=(125000000,))
    p4 = mp.Process(target=count, args=(125000000,))
    p5 = mp.Process(target=count, args=(125000000,))
    p6 = mp.Process(target=count, args=(125000000,))
    p7 = mp.Process(target=count, args=(125000000,))
    p8 = mp.Process(target=count, args=(125000000,))
    p9 = mp.Process(target=count, args=(125000000,))
    p10 = mp.Process(target=count, args=(125000000,))
    p11 = mp.Process(target=count, args=(125000000,))
    p12 = mp.Process(target=count, args=(125000000,))
    p13 = mp.Process(target=count, args=(125000000,))
    p14 = mp.Process(target=count, args=(125000000,))
    p15 = mp.Process(target=count, args=(125000000,))
    p16 = mp.Process(target=count, args=(125000000,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p15.start()
    p16.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    p15.join()
    p16.join()

    end = time.time()
    print(end-start)

if __name__ == '__main__':
    main()