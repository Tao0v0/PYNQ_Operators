#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from pynn.view import view
from tqdm import tqdm


class model:
    # The 1D list of layer objects
    layer_list = []

    # The loss obj
    loss = []

    # The optimizer obj
    optimizer = []
    def __init__(self, backend):
        '''
        inital, backend, and view

        Parameters
        ----------
        backend : TYPE backend
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.backend = backend
        self.v = view()

    def add(self, layer, net_corner=[0, 0]):
        '''
        Add a layer to the neural network model

        Parameters
        ----------
        layer : TYPE a layer object
            DESCRIPTION.
        net_corner : TYPE, subarray left upper corner
            DESCRIPTION. The default is [0, 0].

        Returns
        -------
        None.

        '''

        # In case it's not the first layer,
        # set its input_dim according to the last layer
        if self.layer_list != []:
            layer.input_dim = self.layer_list[-1].output_dim    # 设置当前层的输入维度为前一层的输出维度，当前层不是第一层时

        # RESET weight dimension (due to new input_dim)
        layer.set_weight_dim()          # 设置当前层的权重矩阵维度

        # Backend
        layer.backend = self.backend    # 设置当前层的后端为模型的后端

        # Add one layer in the backend
        self.backend.add_layer(layer.weight_dim, net_corner, len(self.layer_list))      # 在后端添加当前层，传入权重矩阵维度，子阵列左上角位置，当前层编号

        # Add the current layer to the list
        layer.nlayer = len(self.layer_list)
        self.layer_list.append(layer)


    def compile(self, loss, optimizer, save=True):
        '''
        Complie the model

        Parameters
        ----------
        loss : TYPE, a loss object
            DESCRIPTION.
        optimizer : TYPE, an optimizer object
            DESCRIPTION.
        save : TYPE, Boolen
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        '''

        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.backend = self.backend
        self.backend.initialize_weights(save)

    def fit(
        self, x_train, y_train, batch_size=10, epochs=1, shuffle=True
    ):
        '''
        Fit the model according to x_train and y_train

        Parameters
        ----------
        x_train : TYPE, 2D numpy array
            DESCRIPTION. Each column is a training sample
        y_train : TYPE 2D numpy array
            DESCRIPTION. Each column is a training label
        batch_size : TYPE, optional
            DESCRIPTION. The default is 10.
        epochs : TYPE, optional
            DESCRIPTION. The default is 1.
        shuffle : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        # Check sample and label dimension match
        if np.size(x_train, 1) != np.size(y_train, 1):              # 检查x和y的列数是否相等，np.size(x,1)返回x的列数,1是axis
            raise ValueError("Training sample number mismatch!")    

        # The No. of training samples
        n_sample = np.size(x_train, 1)                              # 返回x_train的列数，即样本数，x_train的shape是(输入维度，样本数)

        # Epoch loop
        for ep in tqdm(range(epochs)):                              # tqdm是 一个进度条库，用于显示循环的进度条 

            print(" Training on ep = %d.\n" % (ep))
            # @todo
            x_train_shuffled = x_train
            y_train_shuffled = y_train

            # Process batch data
            for b_start in tqdm(range(0, n_sample, batch_size)):    # 从0到n_sample，每次步进batch_size，b_start
                    # tqdm(range(...)) 和 range(...) 迭代出的数值完全一样，区别只是 tqdm(...) 会在终端/Notebook 里实时显示当前进度
                b_end = min(b_start + batch_size, n_sample)         # 计算当前batch的结束索引，防止越界

                print(" Training on samples %d to %d.\n" % (b_start, b_end))

                x_batch = x_train_shuffled[:, b_start : b_end]      # 切片取出当前batch的输入数据，所有行，b_start到b_end列
                y_batch = y_train_shuffled[:, b_start : b_end]      # 切片取出当前batch的标签数据，所有行，b_start到b_end列

                [loss_value, accuracy] = self.__fit_loop(x_batch, y_batch)  # 私有函数，训练一个batch，返回损失值和准确率
                self.v.save(loss_value, accuracy)                           # 保存损失值和准确率到view对象中
                print("Trainning batch accuracy = %f.\n" % (accuracy))      # 打印当前batch的准确率     

    def predict(self, x_test, batch_size=10):
        '''
        Inference

        Parameters
        ----------
        x_test : TYPE 2D numpy
            DESCRIPTION. Each column is a training sample.
        batch_size : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        y_test : TYPE
            DESCRIPTION.

        '''

        n_sample = np.size(x_test, 1) # np.size(x,1)返回x的列数,1是axis

        # Process batch data
        for b_start in tqdm(range(0, n_sample, batch_size)): # tqdm是 一个进度条库，用于显示循环的进度条,不加也行

            b_end = min(b_start + batch_size, n_sample)      # 以防越界，因为b_start + batch_size 可能大于 n_sample
            x_batch = x_test[:, b_start:b_end]
            if b_start == 0:
                y_test = self.__forwardpass(x_batch)        # 私有函数，前向传递，返回当前batch个样本的输出
            else:                                           # 否则按列拼接
                y_test = np.concatenate(
                    (y_test, self.__forwardpass(x_batch)), axis=1
                )

        return y_test


    def evaluate(self, ys, ys_truth):
        '''
        Inference accuracy

        Parameters
        ----------
        ys : TYPE, 2D numpy,
            DESCRIPTION.
        ys_truth : TYPE, 2D numpy,
            DESCRIPTION.

        Returns
        -------
        accuracy : TYPE
            DESCRIPTION.

        '''

        label_predict = np.argmax(ys, axis=0)
        label_truth = np.argmax(ys_truth, axis=0)
        accuracy = np.mean(label_predict == label_truth)

        return accuracy

    def summary(self):
        '''
        Print a summary of the network

        Returns
        -------
        None.

        '''

        print("------------------------------")
        total = 0
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            weights = layer.weight_dim
            print(
                "layer "
                + str(i)
                + " : input: "
                + str(layer.input_dim)
                + " output: "
                + str(layer.output_dim)
                + " weights: "
                + str(weights)
            )

            total = total + np.prod(weights)

            print("Total parameters:" + str(total))
            print("------------------------------")

    def __fit_loop(self, x_train, y_train):
        '''
        A private internal function to do fitting

        Parameters
        ----------
        x_train : TYPE 2D numpy array
            DESCRIPTION. Each column is a training sample.
        y_train : TYPE 2D numpy array
            DESCRIPTION. Each column is a training label.

        Returns
        -------
        loss_value : TYPE
            DESCRIPTION.
        accuracy : TYPE
            DESCRIPTION.

        '''
        ys = self.__forwardpass(x_train)
        dys, accuracy = self.loss.calc_delta(ys, y_train)
        loss_value = self.loss.calc_loss(ys, y_train)

        grads = self.__backwardpass(dys)
        self.optimizer.update(grads)
        return loss_value, accuracy

    def __forwardpass(self, x_train):
        '''
        Forward pass of a sequential model

        Parameters
        ----------
        x_train : TYPE 2D numpy array
            DESCRIPTION. Each column is a training sample.

        Returns
        -------
        y_ : TYPE
            DESCRIPTION.

        '''
        y_ = x_train

        for l in self.layer_list:
            y_ = l.call(y_)         # 调用layer.py中的call函数进行前向传递

        return y_

    def __backwardpass(self, dys):
        '''
        Backward pass of a sequential model

        Parameters
        ----------
        dys : TYPE, 1D numpy array.
            DESCRIPTION. Delta of the last layer

        Returns
        -------
        grads : TYPE, 1D list. Each element a 2D numpy array,
            DESCRIPTION.  the gradients of a layer

        '''
        dys_ = dys
        grads = []

        for l in self.layer_list[::-1]:  # How about reverse(self.layer_list)
            gradient, dys_ = l.calc_gradients(dys_)
            grads.append(gradient)

        return grads[::-1]
