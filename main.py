import time

import numpy as np


def funciton():
    a = np.random.randn(100000)
    b = np.random.randn(100000)

    begin = time.time()
    c = np.dot(a, b)
    ending = time.time()
    print("time", str((ending - begin) * 1000), "ms");


def broadcasting():
    A = np.array([
        [56.0, 0.0, 4.4, 68.0],
        [1.2, 104.0, 52.0, 8.0],
        [1.8, 135.0, 99.0, 0.9]
    ])
    print(A)
    #计算列和
    cal = A.sum(axis=0)
    print(cal)
    #reshape可以保证矩阵形状
    percentage = 100 * A / cal.reshape(1,4)
    print(percentage)

a = np.zeros(shape = (5,5))
print(a.shape)