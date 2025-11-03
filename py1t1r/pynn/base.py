#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import numpy as np


class memristor_array(ABC):
    #  A superclass for other array interfaces
    #  The purpose of this is just to be able to test identical training
    #  protocols on real and simulated arrays.
    #  Additionally, could be applied to simulations of various styles, or
    #  various types of memristors, etc
    @property
    # @abstractmethod
    # def net_size(self):
    #     raise NotImplementedError

    @abstractmethod
    def read_conductance(self):
        pass

    @abstractmethod
    def update_conductance(self, V_in, V_out, V_gate):
        pass

    @abstractmethod
    def read_current(self, V_in):
        pass


class multi_array(memristor_array):
    # Function : (1) bypass defect rows/cols (2) implement subarrays

    def __init__(self, array, row_list, col_list):
        # array: realarray or softarray objs
        # row_list/col_list: lists of good rows/cols

        self.array = array
        self.net_size_full = [array.NUM_WL, array.NUM_BL]  # Physical size

        self.row_list = row_list
        self.col_list = col_list
        self.net_size = [len(row_list), len(col_list)]  # Useable size

        self.subs = []

    def read_conductance(self):
        # output size = useful array size

        G_full, _ = self.array.read_conductance()  # Full array read
        
        return G_full[np.ix_(self.row_list, self.col_list)]

    def update_conductance(self, v_reset, Vg_set):
        # Vg_set / V_reset : np arrays, with size = size of the useable array

        V_reset_full = np.zeros(self.net_size_full)
        V_reset_full[np.ix_(self.row_list, self.col_list)] = v_reset

        Vg_set_full = np.zeros(self.net_size_full)
        Vg_set_full[np.ix_(self.row_list, self.col_list)] = Vg_set

        self.array.update_conductance(V_reset_full, Vg_set_full)

    def add_sub(self, net_corner, net_size):
        # Creates a rectangular subarray

        new_sub_array = sub_array(self, net_corner, net_size)
        self.subs.append(new_sub_array)

        # Check that they are nonoverlapping:
        check = np.zeros(self.net_size)
        for i in self.subs:
            check += i.mask
        if np.any(check > 1):
            raise ValueError("Overlapping arrays created")

    def read_subs(self):
        # Output G_subs, a list, each is a sub-array G matrix
        # Output G_useful_array, size = size of the useable array

        G_useable_array = self.read_conductance()
        G_subs = [np.reshape(G_useable_array[i.mask], i.net_size) for i in self.subs]

        return G_subs, G_useable_array

    def update_subs(self, V_WL, V_gate):
        # Input V_WL, V_gate: size = size of the useable array
        # Output V_WL_sum, V_gate_sum: size = size of the useable array
        # @todo need?
        V_WL = self.expand(V_WL)
        V_gate = self.expand(V_gate)

        V_WL_sum = np.zeros(self.net_size)
        V_gate_sum = np.zeros(self.net_size)

        for i in range(len(self.subs)):
            V_WL_sum = V_WL_sum + V_WL[i]
            V_gate_sum = V_gate_sum + V_gate[i]

        self.update_conductance(V_WL_sum, V_gate_sum)

        return V_WL_sum, V_gate_sum

    def expand(self, a):
        # B = OBJ.EXPAND(A) attempts to expand A to match each
        # subarray. A can be:
        #   A cell array of the same size as OBJ.SUBS
        #   A vector of the same size as OBJ.SUBS
        #   A scalar
        #   'GND'
        # B is always a cell array, with SIZE(B{I}) == OBJ.NET_SIZE for
        # all I and SIZE(B) == SIZE(OBJ.SUBS)
        b = []
        if isinstance(a, list) and len(a) == len(self.subs):
            for i in range(len(self.subs)):
                # Expand each entry
                b.append(self.subs[i].expand(a[i]))
        elif a == "GND" or np.size(a) == 1:
            for i in range(len(self.subs)):
                # Expand a
                b.append(self.subs[i].expand(a))
        else:
            raise ValueError("Not sure how to expand this input")
        return b

    def read_current(self, V_in):
        # Input Vin size: [no. of useful rows, no. of samples]
        # Output size: [no. of useful cols, no. of samples]

        V_in_full = np.zeros([self.net_size_full[0], np.size(V_in, 1)])
        V_in_full[self.row_list, :] = V_in
        # import pdb
        # pdb.set_trace()
        I_out_full, _, _ = self.array.read_current(V_in_full)

        return I_out_full[self.col_list, :]


class multi_array_16bit(memristor_array):
    # Function : (1) bypass defect rows/cols (2) implement subarrays

    def __init__(self, array, row_list, col_list):
        # array: realarray or softarray objs
        # row_list/col_list: lists of good rows/cols

        self.array = array
        self.net_size_full = [array.NUM_WL, array.NUM_BL]  # Physical size

        self.row_list = row_list
        self.col_list = col_list
        self.net_size = [len(row_list), len(col_list)]  # Useable size

        self.subs = []

    def read_conductance(self):
        # output size = useful array size
        self.array.readmode="slow"
        G_full, _ = self.array.read_conductance()  # Full array read
        self.array.readmode="fast"

        return G_full[np.ix_(self.row_list, self.col_list)]

    def update_conductance(self, Vg_set, v_set, v_reset):
        # Vg_set / V_reset : np arrays, with size = size of the useable array

        Vg_set_full = np.zeros(self.net_size_full)
        Vg_set_full[np.ix_(self.row_list, self.col_list)] = Vg_set

        V_reset_full = np.zeros(self.net_size_full)
        V_reset_full[np.ix_(self.row_list, self.col_list)] = v_reset

        V_set_full = np.zeros(self.net_size_full)
        V_set_full[np.ix_(self.row_list, self.col_list)] = v_set

        self.array.update_conductance(Vg_set_full, V_set_full, V_reset_full)

    def add_sub(self, net_corner, net_size):
        # Creates a rectangular subarray

        new_sub_array = sub_array(self, net_corner, net_size)
        self.subs.append(new_sub_array)

        # Check that they are nonoverlapping:
        check = np.zeros(self.net_size)
        for i in self.subs:
            check += i.mask
        if np.any(check > 1):
            raise ValueError("Overlapping arrays created")

    def read_subs(self):
        # Output G_subs, a list, each is a sub-array G matrix
        # Output G_useful_array, size = size of the useable array

        G_useable_array = self.read_conductance()
        G_subs = [np.reshape(G_useable_array[i.mask], i.net_size) for i in self.subs]

        return G_subs, G_useable_array

    def update_subs(self, V_gate, V_BL, V_WL):
        # Input V_WL, V_gate: size = size of the useable array
        # Output V_WL_sum, V_gate_sum: size = size of the useable array
        # @todo need?
        V_WL = self.expand(V_WL)
        V_BL = self.expand(V_BL)
        V_gate = self.expand(V_gate)

        V_WL_sum = np.zeros(self.net_size)
        V_BL_sum = np.zeros(self.net_size)
        V_gate_sum = np.zeros(self.net_size)

        for i in range(len(self.subs)):
            V_WL_sum = V_WL_sum + V_WL[i]
            V_BL_sum = V_BL_sum + V_BL[i]
            V_gate_sum = V_gate_sum + V_gate[i]

        self.update_conductance(V_gate_sum,V_BL_sum, V_WL_sum)

        return V_gate_sum, V_BL_sum, V_WL_sum

    def expand(self, a):
        # B = OBJ.EXPAND(A) attempts to expand A to match each
        # subarray. A can be:
        #   A cell array of the same size as OBJ.SUBS
        #   A vector of the same size as OBJ.SUBS
        #   A scalar
        #   'GND'
        # B is always a cell array, with SIZE(B{I}) == OBJ.NET_SIZE for
        # all I and SIZE(B) == SIZE(OBJ.SUBS)
        b = []
        if isinstance(a, list) and len(a) == len(self.subs):
            for i in range(len(self.subs)):
                # Expand each entry
                b.append(self.subs[i].expand(a[i]))
        elif a == "GND" or np.size(a) == 1:
            for i in range(len(self.subs)):
                # Expand a
                b.append(self.subs[i].expand(a))
        else:
            raise ValueError("Not sure how to expand this input")
        return b

    def read_current(self, V_in):
        # Input Vin size: [no. of useful rows, no. of samples]
        # Output size: [no. of useful cols, no. of samples]

        V_in_full = np.zeros([self.net_size_full[0], np.size(V_in, 1)])
        V_in_full[self.row_list, :] = V_in
        # import pdb
        # pdb.set_trace()
        I_out_full, _, _ = self.array.read_current(V_in_full)

        return I_out_full[self.col_list, :]


class sub_array(memristor_array):
    # Create and operate of subarrays

    def __init__(self, base, net_corner, net_size):
        # Create a sub-array

        array_size = base.net_size
        for x, y, z in zip(net_size, net_corner, array_size):
            if x + y > z:
                raise ValueError("Network exceeds array bounds")

        self.net_size = net_size
        self.net_corner = net_corner

        # Create a logical mask showing which elements are part of the
        self.mask = np.full(array_size, False)
        self.mask[
            net_corner[0] : net_corner[0] + net_size[0],
            net_corner[1] : net_corner[1] + net_size[1],
        ] = True

        # Save the base array
        self.array = base

    def read_current(self, V_in):
        # Input V_in size: [no. rows of the sub, no. of samples]
        # Output: [no. cols of the sub, no. of samples]

        expanded_V_in = np.zeros([self.array.net_size[0], np.size(V_in, 1)])
        expanded_V_in[
            self.net_corner[0] : self.net_corner[0] + self.net_size[0], :
        ] = V_in
        # Returns columns
        I_out_useable_array = self.array.read_current(
            expanded_V_in
        )
        return I_out_useable_array[
            self.net_corner[1] : self.net_corner[1] + self.net_size[1], :
        ]

    def expand(self, a, x=0):
        # B = OBJ.EXPAND(A) zero-pads array A so that it lines up right
        # with the object matrix.
        # B = OBJ.EXPAND(A,X), where X is a scalar, pads with X
        # instead.
        # This is a utility designed for use inside of other methods on
        # this class.

        # First part: Expand a to net_size
        if np.shape(a) == tuple(self.net_size):
            b = a
        elif a == "GND":
            b = np.zeros(self.net_size)
        elif np.shape(a) == ():  # if a scalar
            b = np.ones(self.net_size) * a
        elif np.shape(a, 0) == self.net_size[0] and np.shape(a, 1) == 1:
            b = np.tile(a, (1, self.net_size[1]))
        elif np.shape(a)[1] == self.net_size[1] and np.shape(a)[0] == 1:
            b = np.tile(a, (self.net_size[0], 1))
        else:
            raise ValueError("Not sure how to expand this input")

        # Second part: Pad b to array_size
        c = np.zeros(self.array.net_size) + x
        c[self.mask] = b.flatten()
        return c

    def read_conductance(self):
        # Output G_sub : size = size of the sub-array
        # Not in use for NNs

        G_useable_array = self.array.read_conductance()
        if np.shape(G_useable_array) != tuple(self.net_size):
            G_sub = np.reshape(G_useable_array[self.mask], tuple(self.net_size))
        return G_sub

    ##

    def update_conductance(self, Vg_set, V_reset):
        # Input Vg_set, V_reset : size = size of the sub-array
        # Not in use for NNs

        self.array.update_conductance(self.expand(Vg_set), self.expand(V_reset))