#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:29:06 2019

@author: Zhongrui Wang, Ye Zhuo, Wenhao
"""
# Abstract class for different optimizers
from abc import ABC, abstractmethod
import numpy as np


class optimizer(ABC):
    @property
    @abstractmethod
    def backend(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, grads):
        pass


class RMSprop(optimizer):
    # Backend obj
    backend = []

    # History of previous dWs and gradient mean square
    dWs_pre = []
    grad_mean_sqr = []


    # RMSprop constructor function for RMSprop with momentum.
    def __init__(self, lr=0.001, momentum=0.0, decay=0.9, eps=1e-8):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.eps = eps


    # Input grads: cell array of gradients, each layer per cell
    def update(self, grads):

        # If it's the first time to use grad_mean_sqr
        if self.grad_mean_sqr == []:
            self.grad_mean_sqr = [np.zeros(np.shape(x)) for x in grads]

        # If it's the first time to use momentum
        if self.dWs_pre == []:
            self.dWs_pre = [np.zeros(np.shape(x)) for x in grads]

        # For all layers
        for j, (i, k) in enumerate(zip(grads, self.grad_mean_sqr)):

            # Gradient mean square (evolution)
            k = self.decay * k + (1 - self.decay) * np.square(i)

            # Add momentum
            self.dWs_pre[j] = (
                self.lr * np.divide(i, (np.power(k, 0.5) + self.eps))
                - self.momentum * self.dWs_pre[j]
            )

        # hardware call (although it's dWs_pre, it's not pre at this moment)
        self.backend.update(self.dWs_pre)
