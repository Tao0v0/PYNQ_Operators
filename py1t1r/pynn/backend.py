#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Abstract class for different backends
from abc import ABC, abstractmethod
import numpy as np


class backend(ABC):
    @abstractmethod
    def initialize_weights(self, *args):
        """
        #% Initialized the weights of the network


        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def update(self, dWs):
        """
        # Change weight (given only delta weight cell arrays)

        Parameters
        ----------
        dWs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def add_layer(self, weight_dim):
        """
        Add a layer (given weight dimension vector)

        Parameters
        ----------
        weight_dim : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    @abstractmethod
    def multiply(self, vec, layer):
        """
        Forward pass for dense/LSTM (vectors, and the layer ID)

        Parameters
        ----------
        vec : TYPE
            DESCRIPTION.
        layer : TYPE
            DESCRIPTION.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        pass

    @abstractmethod
    def multiply_reverse(self, vec, layer):
        """
        # Backward pass for dense/LSTM (vector, and the layer ID)

        Parameters
        ----------
        vec : TYPE
            DESCRIPTION.
        layer : TYPE
            DESCRIPTION.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        pass

    @abstractmethod
    def check_layer(self, layer):
        """
        # Check whether layer ID in proper range

        Parameters
        ----------
        layer : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass


# CPU backend
class software(backend):
    def __init__(self):
        """
        # Instantiation

        Returns
        -------
        None.

        """

        # Weight (list)
        self.W = []

    # Add a layer to the backend.
    def add_layer(self, weight_dim, *args):
        """


        Parameters
        ----------
        weight_dim : TYPE
            DESCRIPTION.
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # weight_dim: 2-element int list, rows & cols of the 2D matrix

        # Initial value (-1,1)
        self.W.append(2 * np.random.rand(*weight_dim) - 1) # weight_dim 是二维矩阵的形状， random.rand将随机生成一个指定形状的矩阵

        # Normalization (why sqrt(num of columns)?)
        self.W[-1] = self.W[-1] / np.sqrt(np.size(self.W[-1], 0))

    # Initialize weight (to be compatiable with
    def initialize_weights(self, *args):
        """


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

    # Update weights
    def update(self, dW):
        """


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

    # Validate if the layer is in range
    def check_layer(self, layer):
        """


        Parameters
        ----------
        layer : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # layer: int, layer number

        if layer >= len(self.W):
            raise ValueError("Layer number error", "layer")

    # Forward pass VMM for dense, LSTM
    def multiply(self, vec, layer):
        """


        Parameters
        ----------
        vec : TYPE
            DESCRIPTION.
        layer : TYPE
            DESCRIPTION.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        # vec: 2D numpy matrix, each column per sample
        # layer: int, layer number

        # Check layer
        self.check_layer(layer)

        # VMM
        output = np.matmul(self.W[layer], vec)

        return output

    # Backward pass VMM for dense, LSTM
    def multiply_reverse(self, vec, layer):
        """


        Parameters
        ----------
        vec : TYPE
            DESCRIPTION.
        layer : TYPE
            DESCRIPTION.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        # vec: 2D numpy matrix, each column per sample
        # layer: int, layer number

        # Check layer
        self.check_layer(layer)

        # VMM
        output = np.matmul(self.W[layer].transpose(), vec)

        return output


# Crossbar backend
class xbar(backend):
    def __init__(self, base, software_aided=False):
        """
        Instantiation

        Parameters
        ----------
        base : TYPE memristor object (real array or simu array)
            DESCRIPTION. The default is "fast".

        Returns
        -------
        None.

        """

        self.base = base

        # no. of the physical layer (otherwise 0)
        self.phys_layer_num = []
        
        # SET and gate voltage 19/63*VDD = 1.51V
        self.Vg_max = 19
        self.Vg_min = 6  # Min SET gate voltage 6/63*VDD = 0.48V
        self.Vg0 = 13  # Initial SET gate voltage 13/63*VDD = 1.03V

        #RESET and gate voltage
        self.V_reset = 25  # RESET alone RESET voltage 25/63*VDD = 1.98V

        # Read parameters (range for VMM [-V_read, V_read])
        self.V_read = 31

        # Mapping (Conductance to weight, Gate voltage to conductance)
        self.ratio_G_W = 100e-6  # Early test 100 to 250e-6
        # Delta_V_gate / Delta_conductancd ~1/98e-6
        self.ratio_Vg_G = 13 / 100e-6

        # the array conductance for multiply reverse
        # update the value after weight update
        self.array_G = []

        # History
        self.V_gate_last = []  # tuned parameter determines the final G

        # Using software-aided backend
        self.software_aided=software_aided

    def add_layer(self, weight_dim, net_corner, layer_original):
        """
        Add a physical layer to the backend.

        Parameters
        ----------
        weight_dim : TYPE 2-element list
            DESCRIPTION.
        net_corner : TYPE
            DESCRIPTION.
        layer_original : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if any(weight_dim) == False:
            # Software layer
            self.phys_layer_num.append(np.nan)
        else:
            # Physical layer
            self.phys_layer_num.append(len(self.base.subs))

            # Physical size accouting verticial differential pair
            phys_size = weight_dim[::-1]
            phys_size[0] = phys_size[0] * 2

            # Physically allocate the layer
            self.base.add_sub(net_corner, phys_size)

    def initialize_weights(self, save=True):
        """
        Initialize weights

        Parameters
        ----------
        save : TYPE, Bool
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """

        # Initial SET voltages (Vg0)
        self.V_gate_last = [np.zeros(x.net_size) + self.Vg0 for x in self.base.subs]

        # Update the initial conductance
        # RESET pulse
        self.base.update_subs(self.V_reset, 0)
        # SET pulse
        self.base.update_subs(0, self.V_gate_last)

        # Initialize saving variables
        if save:
            self.save = True
            self.G_hist = []  # G history
            self.V_reset_hist = []  # Programming history
            self.Vg_set_hist = []
            self.V_hist = []  # VMM history
            self.I_hist = []
        else:
            self.save = False

        # Read and store the weight
        self.read_conductance()

    # Update weights (conductance)
    def update(self, dWs_original):

        # Remove dW (0x0 dimension) of "virtual" layers
        dWs = [
            dW__ for (x, dW__) in zip(self.phys_layer_num, dWs_original) if ~np.isnan(x)
        ]
        for dW in dWs:
            assert not np.any(np.isnan(dW))

        # Check number of dWs is same with number of physical layers
        if len(dWs) != len(self.base.subs):
            raise ValueError("Wrong number of weight gradient matrices")

        # Initialize variables
        Vr = []
        Vg = []

        for layer in range(len(dWs)):

            # Transpose
            dW = dWs[layer].T

            # Gradient scaling and to voltage conversion
            dV_temp = self.ratio_Vg_G * self.ratio_G_W * dW

            # Vertical differential pair
            dV = np.empty([x * y for x, y in zip(np.shape(dV_temp), [2, 1])])
            dV[0::2, :] = dV_temp
            dV[1::2, :] = -dV_temp

            # RESET if dV is negative
            Vr.append(self.V_reset * (dV < 0))

            # SET gate voltage
            temp_Vg = self.V_gate_last[layer] + dV
            # Regulate the min and max SET gate voltage
            temp_Vg[temp_Vg > self.Vg_max] = self.Vg_max
            temp_Vg[temp_Vg < self.Vg_min] = self.Vg_min
            Vg.append(temp_Vg)

        # RESET pulse
        p1 = self.base.update_subs(Vr, 0)
        # SET pulse
        p2 = self.base.update_subs(0, Vg)
        # An alternative, but with slightly more noisey SET
        # p1 = self.base.update_subs(Vr, Vg_apply) # Combined SET/RESET

        # Save updated gate voltages
        self.V_gate_last = Vg

        # update the conductance for software backpropogation
        self.read_conductance()

        # Save pulse history if needed
        if self.save:
            self.V_reset_hist.append(p1[0])
            self.Vg_set_hist.append(p2[1])

    # READ_CONDUCTANCE read the conductances of all sub_arrays
    def read_conductance(self):

        # Conductance read
        [self.array_G, G_usable_array] = self.base.read_subs()

        if self.save:
            self.G_hist.append(G_usable_array)
        for array in G_usable_array:
            assert not np.any(np.isnan(array))

        return G_usable_array

    # Forward pass VMM for dense, LSTM
    def multiply(self, vec, layer_original):
        # vec (INPUT): 2D matrix, each column per sample
        # layer (INPUT): layer number

        # Check is the layer a valid layer and return physical layer no
        layer = self.check_layer(layer_original)

        # get the max value of each cols of vec
        vec_col_max = np.amax(np.abs(vec), axis=0)

        # Voltage scaling (input * scaling = voltage)
        voltage_scaling = np.where(
            vec_col_max > 0, self.V_read / vec_col_max, self.V_read / self.V_read,
        )

        assert not np.any(np.isnan(voltage_scaling))

        # Differential pair
        V_input = np.empty([x * y for x, y in zip(np.shape(vec), [2, 1])])
        temp = vec * voltage_scaling
        temp[temp > self.V_read] = self.V_read
        V_input[0::2, :] = temp
        V_input[1::2, :] = -V_input[0::2, :]

        assert not np.any(np.isnan(V_input))

        # Software aided or hardware
        if self.software_aided:
            I_output = np.dot(self.array_G[layer].T, V_input)
        else:
            I_output = self.base.subs[layer].read_current(V_input)

        # Scaling back (voltage and weight scaling)
        vmm_output = I_output / voltage_scaling / self.ratio_G_W

        assert not np.any(np.isnan(vmm_output))

        # Plot and save
        if self.save:
            self.V_hist.append(V_input)
            self.I_hist.append(I_output)

        return vmm_output

    # Backward pass VMM for dense, LSTM
    def multiply_reverse(self, vec, layer_original):
        # vec (INPUT) : each col per input vector
        # layer_original : original layer number

        # Check is the layer a valid layer and return physical layer no
        layer = self.check_layer(layer_original)

        # Retrieve last read G
        G = self.array_G[layer]

        # Reduce from the vertical differential pair to a single scalar
        w = G[0::2, :] - G[1::2, :]

        # Reverse multiplication
        # w is tranposed compared to upper level algorithrms
        output = np.dot(w, vec) / self.ratio_G_W
        return output

    # Validate if the layer is in range
    def check_layer(self, layer_original):
        # layer: int, layer number

        # Check the correspnding
        layer = self.phys_layer_num[layer_original]

        if layer >= len(self.base.subs):
            raise ValueError("Layer number error", "layer")

        return layer



# Crossbar backend for 16 bit
class xbar_16bit(backend):
    def __init__(self, base, software_aided=False):
        """
        Instantiation

        Parameters
        ----------
        base : TYPE memristor object (real array or simu array)
            DESCRIPTION. The default is "fast".

        Returns
        -------
        None.

        """

        self.base = base

        # no. of the physical layer (otherwise 0)
        self.phys_layer_num = []

        # SET and gate voltage 19/63*VDD = 1.51V
        self.V_set = 31
        self.Vg_min = 6  # Min SET gate voltage 6/63*VDD = 0.48V
        self.Vg_max = 19
        self.Vg0 = 13  # Initial SET gate voltage 13/63*VDD = 1.03V

        #RESET voltage
        self.V_reset = 25  # RESET alone RESET voltage 25/63*VDD = 1.98V
        self.V_gate_reset = 63

        # Read parameters (range for VMM [-V_read, V_read])
        self.V_read = 31

        # Mapping (Conductance to weight, Gate voltage to conductance)
        self.ratio_G_W = 100e-6  # Early test 100 to 250e-6
        # Delta_V_gate / Delta_conductancd ~1/98e-6
        self.ratio_Vg_G = 13 / 100e-6

        # the array conductance for multiply reverse
        # update the value after weight update
        self.array_G = []

        # History
        self.V_gate_last = []  # tuned parameter determines the final G

        # Using software-aided backend
        self.software_aided=software_aided

    def add_layer(self, weight_dim, net_corner, layer_original):
        """
        Add a physical layer to the backend.

        Parameters
        ----------
        weight_dim : TYPE 2-element list
            DESCRIPTION.
        net_corner : TYPE
            DESCRIPTION.
        layer_original : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if any(weight_dim) == False:
            # Software layer
            self.phys_layer_num.append(np.nan)
        else:
            # Physical layer
            self.phys_layer_num.append(len(self.base.subs))

            # Physical size accouting verticial differential pair
            phys_size = weight_dim[::-1]
            phys_size[0] = phys_size[0] * 2

            # Physically allocate the layer
            self.base.add_sub(net_corner, phys_size)

    def initialize_weights(self, save=True):
        """
        Initialize weights

        Parameters
        ----------
        save : TYPE, Bool
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """

        # Initial SET voltages (Vg0)
        self.V_gate_last = [np.zeros(x.net_size) + self.Vg0 for x in self.base.subs]

        # Update the initial conductance
        # RESET pulse
        self.base.update_subs(self.V_gate_reset, 0, self.V_reset)
        # SET pulse
        self.base.update_subs(self.V_gate_last, self.V_set, 0)

        # Initialize saving variables
        if save:
            self.save = True
            self.G_hist = []  # G history
            self.V_reset_hist = []  # Programming history
            self.Vg_set_hist = []
            self.V_hist = []  # VMM history
            self.I_hist = []
        else:
            self.save = False

        # Read and store the weight
        self.read_conductance()

    # Update weights (conductance)
    def update(self, dWs_original):

        # Remove dW (0x0 dimension) of "virtual" layers
        dWs = [
            dW__ for (x, dW__) in zip(self.phys_layer_num, dWs_original) if ~np.isnan(x)
        ]
        for dW in dWs:
            assert not np.any(np.isnan(dW))

        # Check number of dWs is same with number of physical layers
        if len(dWs) != len(self.base.subs):
            raise ValueError("Wrong number of weight gradient matrices")

        # Initialize variables
        Vr = []
        Vg = []

        for layer in range(len(dWs)):

            # Transpose
            dW = dWs[layer].T

            # Gradient scaling and to voltage conversion
            dV_temp = self.ratio_Vg_G * self.ratio_G_W * dW

            # Vertical differential pair
            dV = np.empty([x * y for x, y in zip(np.shape(dV_temp), [2, 1])])
            dV[0::2, :] = dV_temp
            dV[1::2, :] = -dV_temp

            # @todo
            # RESET if dV is negative (or <th_reset)
            Vr.append(self.V_reset * (dV < 0))

            # Update (p1, p2 are pulse history)
            # Regulate the min and max SET gate voltage
            temp_Vg = self.V_gate_last[layer] + dV
            temp_Vg[temp_Vg > self.Vg_max] = self.Vg_max
            temp_Vg[temp_Vg < self.Vg_min] = self.Vg_min
            Vg.append(temp_Vg)

        # RESET pulse
        p1 = self.base.update_subs(self.V_gate_reset, 0, Vr)
        # SET pulse
        p2 = self.base.update_subs(Vg, self.V_set, 0)
        # An alternative, but with slightly more noisey SET
        # p1 = self.base.update_subs(Vr, Vg_apply) # Combined SET/RESET

        # Save updated gate voltages
        self.V_gate_last = Vg

        # update the conductance for software backpropogation
        self.read_conductance()

        # Save pulse history if needed
        if self.save:
            self.V_reset_hist.append(p1[2])
            self.Vg_set_hist.append(p2[0])
            # self.V_set_hist.append(p2[1])

    # READ_CONDUCTANCE read the conductances of all sub_arrays
    def read_conductance(self):

        # Conductance read
        [self.array_G, G_usable_array] = self.base.read_subs()

        if self.save:
            self.G_hist.append(G_usable_array)
        for array in G_usable_array:
            assert not np.any(np.isnan(array))

        return G_usable_array

    # Forward pass VMM for dense, LSTM
    def multiply(self, vec, layer_original):
        # vec (INPUT): 2D matrix, each column per sample
        # layer (INPUT): layer number

        # Check is the layer a valid layer and return physical layer no
        layer = self.check_layer(layer_original)

        # get the max value of each cols of vec
        vec_col_max = np.amax(np.abs(vec), axis=0)

        # Voltage scaling (input * scaling = voltage)
        voltage_scaling = np.where(
            vec_col_max > 0, self.V_read / vec_col_max, self.V_read / self.V_read,
        )

        assert not np.any(np.isnan(voltage_scaling))

        # Repeat V_input (dp_rep vertical duplication)
        V_input = np.empty([x * y for x, y in zip(np.shape(vec), [2, 1])])
        V_input[0::2, :] = vec * voltage_scaling
        V_input[1::2, :] = -V_input[0::2, :]

        assert not np.any(np.isnan(V_input))

        # Software aided or hardware
        if self.software_aided:
            I_output = np.dot(self.array_G[layer].T, V_input)
        else:
            I_output = self.base.subs[layer].read_current(V_input)

        # Scaling back (voltage and weight scaling)
        vmm_output = I_output / voltage_scaling / self.ratio_G_W

        assert not np.any(np.isnan(vmm_output))

        # Plot and save
        if self.save:
            self.V_hist.append(V_input)
            self.I_hist.append(I_output)

        return vmm_output

    # Backward pass VMM for dense, LSTM
    def multiply_reverse(self, vec, layer_original):
        # vec (INPUT) : each col per input vector
        # layer_original : original layer number

        # Check is the layer a valid layer and return physical layer no
        layer = self.check_layer(layer_original)

        # Retrieve last read G
        G = self.array_G[layer]

        # Reduce from the vertical differential pair to a single scalar
        w = G[0::2, :] - G[1::2, :]

        # Reverse multiplication
        # w is tranposed compared to upper level algorithrms
        output = np.dot(w, vec) / self.ratio_G_W
        return output

    # Validate if the layer is in range
    def check_layer(self, layer_original):
        # layer: int, layer number

        # Check the correspnding
        layer = self.phys_layer_num[layer_original]

        if layer >= len(self.base.subs):
            raise ValueError("Layer number error", "layer")

        return layer

