#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pynn.backend import software    # 后端实现，提供add_layer() ,multiply()
from pynn.layer import dense        # 全连接定义
from pynn.loss import cross_entropy_softmax_loss    # 交叉函数
from pynn.optimizer import RMSprop      # 优化器
from pynn.model import model            # 模型定义
import scipy.io                     # 读取.mat文件，，这里用来读取MNIST的.mat数据与标签
import numpy as np                  # 数组计算
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent / "pynn" / "dataset"      # 输入的mat文件都是扁平化后的结果，就是把8*8的图像拉成了64*1的格式
TRAIN_X_MAT = DATASET_DIR / "mnist_training_8x8_0d2.mat"
TRAIN_Y_MAT = DATASET_DIR / "mnist_train_labels_onehot.mat"
TEST_X_MAT  = DATASET_DIR / "mnist_test_8x8_0d2.mat"
TEST_Y_MAT  = DATASET_DIR / "mnist_test_labels_onehot.mat"
# train set 
mat_contents = scipy.io.loadmat(str(TRAIN_X_MAT))           # 读取.mat文件,返回字典
xs_train = mat_contents["data_0d2"][:, :]                   # 读取数据部分，这个param是二维的
xs_train = np.delete(xs_train, [0, 7, 56, 63], 0)           # 删除坏行，axis= 0 删除行， 0 7 56 63行

# train labels
mat_contents = scipy.io.loadmat(str(TRAIN_Y_MAT))           # 读取.mat文件,返回字典
ys_train = mat_contents["mnist_train_labels"][:, :]         # 读取标签部分，这个param是二维的

# test set
mat_contents = scipy.io.loadmat(str(TEST_X_MAT))            # 读取.mat文件,返回字典
xs_test = mat_contents["data_0d2"][:, :]                    # 读取数据部分，这个param是二维的
xs_test = np.delete(xs_test, [0, 7, 56, 63], 0)             # 删除坏行，axis= 0 删除行， 0 7 56 63行

# test labels
mat_contents = scipy.io.loadmat(str(TEST_Y_MAT))            # 读取.mat文件,返回字典
ys_test = mat_contents["mnist_test_labels"][:, :]           # 读取标签部分，这个param是二维的


#%%
# Model
m = model(software())   # Backend selection

# Add layers        每次输入一个batch大小的样本进入网络,输入的x_train 维度是(输入维度，样本数) (60,6000)
m.add(
    dense(40, input_dim=60, activation="relu", bias_config=[0, 0]),
    net_corner=[1, 1],  # 子阵列左上角位置，意思是该层映射到硬件的哪个位置，用在硬件后端
    # dp_rep=[1, 1, 1],
)
m.add(
    dense(10, activation="stable_softmax", bias_config=[0, 0]),     # 这里没写输入维度的原因是，在model.add里，当当前层不是第一层时，会把当前层的输入维度设置为前一层的输出维度
    net_corner=[1, 1],
    # dp_rep=[1, 1, 1],
)

# Auxilary
m.summary()     # 打印网络结构概览，遍历模型的每一层，逐层输出，层编号i,该层的input_dim和output_dim 该层权重矩阵的尺寸，累计参数量


#%%
no_epoches = 2  # No. of epoches
ys_test_hardware = []  # Initialize inference results   
accuracy = []  # Initialize accuracy results
# Fit

m.compile(cross_entropy_softmax_loss(), RMSprop(lr=0.01), save=True)  # 加入损失函数，优化器，save=True保存模型参数

for epo in range(no_epoches):    # 循环两个epoch

    # Train
    m.fit(xs_train, ys_train, batch_size=100, epochs=1)             # fit是model属性，训练模型，x_train的维度是(输入维度，样本数)，fit调用model中__fitloop,然后调用forwardpass，再调用call，完成dense层的前向传递

    # Inference
    ys_test_hardware.append(m.predict(xs_test, batch_size=100))     # predict是model属性，预测函数，batch_size每次预测样本数，x_test的维度是(输入维度，样本数)
    accuracy.append(m.evaluate(ys_test_hardware[epo], ys_test))     # evaluate是model属性，评估函数，计算预测结果与真实标签的准确率


#%%
m.v.plot()          # 显示训练过程中的损失值和准确率曲线,记得在view.py开头加上 matplotlib.use("AGG")
input('?')          # 防止程序运行结束后图像窗口关闭