#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from backend import backend
from pynq import Overlay
from pynq import Xlnk

# CPU backend
class fpga_mlp(backend):


    # Weight (list)
    W = []

    # Instantiation
    def __init__(self):
        """


        Returns
        -------
        None.

        """
        self.overlay = Overlay(
            "/home/xilinx/pynq/overlays/vmm_mlp_100_40_10/vmm_mlp_100_40_10.bit"
        )
        self.dma1 = self.overlay.axi_dma_0
        self.dma2 = self.overlay.axi_dma_1
        # pass

    def add_layer(self, weight_dim, *args):
        """
        # Add a layer to the backend.

        Parameters
        ----------
        weight_dim : 2-element int list, rows & cols of the 2D matrix
            DESCRIPTION.
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Initial value (-1,1)
        self.W.append(2 * np.random.rand(*weight_dim) - 1)

        # Normalization (why sqrt(num of columns)?)
        self.W[-1] = self.W[-1] / np.sqrt(np.size(self.W[-1], 0))

    def initialize_weights(self, *args):
        """
        #    Initialize weight (to be compatiable with

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # Debug only Load weights
        # mat_contents = scipy.io.loadmat("weight1.mat")
        # self.W[0] = mat_contents["temp1"]

        # mat_contents = scipy.io.loadmat("weight2.mat")
        # self.W[1] = mat_contents["temp2"]
        pass

    def update(self, dW):
        """
        # Update weights

        Parameters
        ----------
        dW : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # dW (weight changes): nx1 list, each entity a numpy array

        for l in range(len(dW)):
            self.W[l] = self.W[l] + dW[l]
        pass

    def check_layer(self, layer):
        """
        # Validate if the layer is in range

        Parameters
        ----------
        layer : TYPE int, layer number
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if layer >= len(self.W):
            raise ValueError("Layer number error", "layer")

    def multiply(self, vec, layer):
        """
        # Forward pass VMM for dense, LSTM

        Parameters
        ----------
        vec : TYPE 2D numpy matrix, each column per sample
            DESCRIPTION.
        layer : TYPE int, layer number
            DESCRIPTION.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        # Check layer
        self.check_layer(layer)

        # VMM
        output = np.matmul(self.W[layer], vec)
        if layer == 0:
            dma = self.dma1
            # pass
        else:
            dma = self.dma2
            # pass

        # Prepare input/output buffers
        batch_dim = 100
        output_dim, input_dim = np.shape(self.W[layer])
        # output matrix

        input_size = batch_dim * input_dim + input_dim * output_dim
        output_size = batch_dim * output_dim

        xlnk = Xlnk()

        input_buffer = xlnk.cma_array(shape=(input_size,), dtype=np.float32)
        output_buffer = xlnk.cma_array(shape=(output_size,), dtype=np.float32)

        # Assign value to input buffer
        np.copyto(
            input_buffer,
            np.concatenate((vec.flatten(), self.W[layer].flatten()), axis=0),
        )

        # Hardware communication
        dma.sendchannel.transfer(input_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(output_buffer)
        dma.recvchannel.wait()

        # Send
        output = np.copy(output_buffer)
        output = np.reshape(output, (output_dim, batch_dim))

        # Close buffer
        input_buffer.close()
        output_buffer.close()

        return output

    def multiply_reverse(self, vec, layer):
        """
        Backward pass VMM for dense, LSTM

        Parameters
        ----------
        vec : TYPE 2D numpy matrix, each column per sample
            DESCRIPTION.
        layer : TYPE int, layer number
            DESCRIPTION.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """

        # Check layer
        self.check_layer(layer)

        # VMM
        output = np.matmul(self.W[layer].transpose(), vec)

        return output
