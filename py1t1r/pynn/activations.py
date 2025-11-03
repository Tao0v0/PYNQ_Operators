#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This class provides various activation functions
import numpy as np


class activations:
    # Get the activation function handel.
    @staticmethod
    def get(act_name, direction):
        # act_name: string input,  name of the activation function
        # direction: string input, either 'act' or 'deriv'
        # act: output, the function handle

        act = "d" + act_name if direction == "deriv" else act_name  # 根据direction决定是获取激活函数还是其导数函数
        return getattr(activations, act)        # getattr() 函数用于获取对象的属性，这里是获取activations类中的指定激活函数

    # Activation functions
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def dlinear(x):
        return 1

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1 - x ** 2

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def drelu(x):
        return x > 0

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    @staticmethod
    def stable_softmax(x):
        return np.exp(x - np.amax(x, axis=0)) / sum(np.exp(x - np.amax(x, axis=0)))
        
    def gelu(x, approximate=True):
        """
        GELU 激活（默认用 tanh 近似），只用于前向推理。
        exact 公式：0.5 * x * (1 + erf(x / sqrt(2)))
        """
        if approximate:
            # Hendrycks & Gimpel 的 tanh 近似，数值稳定且不依赖 erf
            c = np.sqrt(2.0/np.pi)
            return 0.5 * x * (1.0 + np.tanh(c*(x + 0.044715*(x**3))))
        else:
            # 精确形式（需要 np.erf，可用 numpy.special.erf）
            return 0.5 * x * (1.0 + np.erf(x / np.sqrt(2.0)))