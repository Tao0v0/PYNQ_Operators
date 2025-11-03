#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Abstract class for different losses
from abc import ABC, abstractmethod
import numpy as np


class loss(ABC):

    # TODO discuss
    @abstractmethod
    # Output-Label (a batch)
    def calc_delta(self, ys, y_train):
        pass

    @abstractmethod
    # Loss function (a batch)
    def calc_loss(self, ys, y_train):
        pass


class cross_entropy_softmax_loss(loss):

    # TODO discuss
    def __init__(self):
        pass

    def calc_delta(self, ys, ys_train):
        dys = ys_train - ys
        # Only the last output matters for some recurrent neural networks

        winner_ys = np.argmax(ys, axis=0)
        winner_ys_train = np.argmax(ys_train, axis=0)

        accuracy = np.mean(winner_ys - winner_ys_train == 0)
        return dys, accuracy

    def calc_loss(self, ys, ys_train):
        # ys: 2D numpy array, forward pass outputs.
        # ys_train: 2D numpy array, training labels.

        loss = 0
        for n in range(np.size(ys, 1)):
            for row in range(np.size(ys, 0)):
                loss = loss - ys_train[row, n] * np.log(ys[row, n])

        # Mean loss per sample
        loss = loss / np.size(ys, 1)
        return loss
