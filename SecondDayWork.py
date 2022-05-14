import math

import numpy as np
import h5py
import matplotlib.pyplot as plt
# 引入pylab解决打印问题
import pylab as pl
from lr_utils import load_dataset

xTr, yTr, xTe, yTe, classes = load_dataset()
xTr_translation = xTr.reshape(xTr.shape[0], -1).T
xTe_translation = xTe.reshape(xTe.shape[0], -1).T
xTr_set = xTr_translation / 255
xTe_set = xTe_translation / 255


def printXTr(num):
    plt.imshow(xTr[num])
    pl.show()


def printYTr(num):
    # 打印出当前的训练标签值
    # 使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1 , 去掉维度为1
    # print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
    # 只有压缩后的值才能进行解码操作
    print("y=" + str(yTr[:, num]) + ", it's a " + classes[np.squeeze(yTr[:, num])].decode(
        "utf-8") + "' picture")


def showTr(num):
    printXTr(num)
    printYTr(num)


def showData():
    # yTr.shape ----  (1 , 209)
    # yTr.shape[0] ----- 1
    # yTr.shape[1] ----- 209
    numTr = yTr.shape[1]
    numTe = yTe.shape[1]
    # xTr.shapr ----- (209 , 64 , 64 , 3)  209张图片 64*64  RGB三通道
    # 209 * 64 * 64 * 3 = 2,568,192 个数据集
    numPx = xTr.shape[2]

    # Printinggggggggggggggggg
    print("我的训练图片有：" + str(numTr))
    print("我的测试图片有: " + str(numTe))
    print("我的图片像素： " + str(numPx) + "px * " + str(numPx) + "px")
    print("每张图片矩阵： ( " + str(numPx) + " , " + str(numPx) + " , " + " 3 )")
    print("训练图片  ：" + str(xTr.shape))
    print("训练结果  ：" + str(yTr.shape))
    print("测试图片  ：" + str(xTe.shape))
    print("测试结果  ：" + str(yTe.shape))


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def init(dimention):
    w = np.zeros(shape=(dimention, 1))
    b = 0
    assert (w.shape == (dimention, 1))
    assert (isinstance(b, int) or isinstance(b, float))
    return w, b


def propagate(w, b, X, Y):
    num = X.shape[1]
    # 正向
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = (-1 / num) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    # 反向
    dz = A - Y

    dw = (1 / num) * np.dot(X, dz.T)
    db = (1 / num) * np.sum(dz)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost


def proTest():
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    grads, cost = propagate(w, b, X, Y)
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("cost = " + str(cost))


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    # []列表
    # ()元组  不可以修改内部单个
    # {}字典
    global dw
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("迭代次数是：%i  当前误差为：%f" % (i, cost))
    params = {
            "w":w,
            "b":b
    }
    grads = {
            "dw":dw,
            "db":db
    }
    return params, grads, costs


def OpTest():
    print("====================测试optimize====================")
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))


def predict(w, b, X):
    m = X.shape[1]
    preY = np.zeros(shape = (1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) +b)

    for i in range(m):
        preY[0, i] = 1 if A[0, i] > 0.5 else 0
    assert (preY.shape == (1, m))
    return preY


def preTest():
    # 测试predict
    print("====================测试predict====================")
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    print("predictions = " + str(predict(w, b, X)))


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    #初始化
    w, b = init(X_train.shape[0])
    #训练
    parameters , grads , costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    #获得训练结果
    w, b = parameters["w"], parameters["b"]
    #结果进行预测
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    #打印
    #np.mean ---- 求取平均值
    print("训练集测试准确度： ", format(100 - np.mean(np.abs(Y_prediction_train - Y_train))*100), "%")
    print("测试集测试准确度： ", format(100 - np.mean(np.abs(Y_prediction_test - Y_test))*100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d


def modelTest():
    print("====================测试model====================")
    # 这里加载的是真实的数据，请参见上面的代码部分。
    d = model(xTr_set, yTr, xTe_set, yTe, num_iterations=2000, learning_rate=0.005, print_cost=True)
    # 绘制图
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


modelTest()