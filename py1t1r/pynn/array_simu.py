#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:56:36 2020

@author: yezhuo
"""
from pynn.base import memristor_array

import numpy as np

class array_simu(memristor_array):
    #  Simulated subarray
    #  Assumes 0 wire resistance, all devices identical, 100#  yield, 0 noise
    #  Construction syntax: self = array_simu1(G0, UPDATE_FUN, GMIN, GMAX)

    def __init__(self, g0, update_fun, Vg_RESET=63, V_SET=32, gmin=0, gmax=np.inf):
        """
          self = array_simu1(G0, UPDATE_FUN, GMIN, GMAX)
          creates a simulated memristor array.
          G0 is the initial weight matrix, and determines the size of
          the array. G0 can also be a cell array holding:
            {'random' SIZE MIN MAX}     Generates uniformly distributed
                                        random weights between MIN and
                                        MAX
            {'fixed' SIZE VALUE}        Initializes all weights to
                                        exactly VALUE
          UPDATE_FUN gives delta-G as a def of G, V_in, and V_trans
          GMIN defaults to 0
          GMAX defaults to Inf
          The pulse width is fixed for now, but number of pulses may
          eventually be something we can tune.
          """

        self.Vg_RESET = Vg_RESET  # 0-63
        self.V_SET = V_SET  # 0-63
        self.gmin = gmin
        self.gmax = gmax
        self.conductances = self.generate_g0(g0)
        self.NUM_WL, self.NUM_BL = np.shape(self.conductances)
        self.update_fun = update_fun

    def generate_g0(self, g0):
        """
        generate initial weights, random or fixed
        """
        if type(g0) == list:
            if g0[0].lower() == "random":
                #  Second entry is size, third is minval, 4th is maxval.
                return np.random.uniform(low=g0[2], high=g0[3], size=g0[1])
            elif g0[0].lower() == "fixed":

                #  Second entry is size, third is value
                return np.zeros(g0[1]) + g0[2]
            else:
                raise ValueError("Check 1st item of g0.")
        raise ValueError("Check g0.")

    def read_conductance(self):
        """
        Noise-free reading.
        """
        dout = np.ones_like(self.conductances)  # Fake DOUT
        return self.conductances, dout

    def read_current(self, V_in):
        #  Optional arguments can be entered but are ignored
        return np.dot(self.conductances.T, V_in), None, None

    def update_conductance(self, V_WL, V_gate):
        """
        self.UPDATE_CONDUCTANCE(V_IN, V_OUT, V_TRANS)
        V_in and V_trans are vectors, or scalars
        V_out is fixed grounded right now.
        """

        # Expand inputs
        V_WL = self.expand(V_WL)
        V_gate = self.expand(V_gate)

        #  Get conductance:
        g = self.conductances

        # RESET first
        if np.any(V_gate):
            g = g + self.update_fun(g, self.V_SET, V_gate)

        # SET second
        if np.any(V_WL):
            g = g + self.update_fun(g, -V_WL, self.Vg_RESET)

        #  Enforce the maxima:
        g[g < self.gmin] = self.gmin
        g[g > self.gmax] = self.gmax

        #  Save result:
        self.conductances = g

    def expand(self, value):
        """
        Expand return array to same size of (NUM_WL, NUM_BL) with value (input)
        """
        if np.shape(value) == ():
            # value is a number
            return np.ones(self.NUM_WL, self.NUM_BL) * value
        elif np.shape(value) == tuple([self.NUM_WL, self.NUM_BL]):
            # sizes match
            return value

        raise ValueError("Check the expand input")

class array_simu_16bit(memristor_array):
    #  Simulated subarray
    #  Assumes 0 wire resistance, all devices identical, 100#  yield, 0 noise
    #  Construction syntax: self = array_simu1(G0, UPDATE_FUN, GMIN, GMAX)

    def __init__(self, g0, update_fun, Vg_RESET=63, V_SET=32, gmin=0, gmax=np.inf):
        """
          self = array_simu1(G0, UPDATE_FUN, GMIN, GMAX)
          creates a simulated memristor array.
          G0 is the initial weight matrix, and determines the size of
          the array. G0 can also be a cell array holding:
            {'random' SIZE MIN MAX}     Generates uniformly distributed
                                        random weights between MIN and
                                        MAX
            {'fixed' SIZE VALUE}        Initializes all weights to
                                        exactly VALUE
          UPDATE_FUN gives delta-G as a def of G, V_in, and V_trans
          GMIN defaults to 0
          GMAX defaults to Inf
          The pulse width is fixed for now, but number of pulses may
          eventually be something we can tune.
          """

        self.Vg_RESET = Vg_RESET  # 0-63
        self.V_SET = V_SET  # 0-63
        self.gmin = gmin
        self.gmax = gmax
        self.conductances = self.generate_g0(g0)
        self.NUM_WL, self.NUM_BL = np.shape(self.conductances)
        self.update_fun = update_fun

    def generate_g0(self, g0):
        """
        generate initial weights, random or fixed
        """
        if type(g0) == list:
            if g0[0].lower() == "random":
                #  Second entry is size, third is minval, 4th is maxval.
                return np.random.uniform(low=g0[2], high=g0[3], size=g0[1])
            elif g0[0].lower() == "fixed":

                #  Second entry is size, third is value
                return np.zeros(g0[1]) + g0[2]
            else:
                raise ValueError("Check 1st item of g0.")
        raise ValueError("Check g0.")

    def read_conductance(self):
        """
        Noise-free reading.
        """
        dout = np.ones_like(self.conductances)  # Fake DOUT
        return self.conductances, dout

    def read_current(self, V_in):
        #  Optional arguments can be entered but are ignored
        return np.dot(self.conductances.T, V_in), None, None

    def update_conductance(self, V_gate, V_BL, V_WL):
        """
        self.UPDATE_CONDUCTANCE(V_IN, V_OUT, V_TRANS)
        V_in and V_trans are vectors, or scalars
        V_out is fixed grounded right now.
        """

        # Expand inputs
        V_WL = self.expand(V_WL)
        V_gate = self.expand(V_gate)

        #  Get conductance:
        g = self.conductances

        # RESET first
        if np.any(V_gate):
            g = g + self.update_fun(g, self.V_SET, V_gate)

        # SET second
        if np.any(V_WL):
            g = g + self.update_fun(g, -V_WL, self.Vg_RESET)

        #  Enforce the maxima:
        g[g < self.gmin] = self.gmin
        g[g > self.gmax] = self.gmax

        #  Save result:
        self.conductances = g

    def expand(self, value):
        """
        Expand return array to same size of (NUM_WL, NUM_BL) with value (input)
        """
        if np.shape(value) == ():
            # value is a number
            return np.ones(self.NUM_WL, self.NUM_BL) * value
        elif np.shape(value) == tuple([self.NUM_WL, self.NUM_BL]):
            # sizes match
            return value

        raise ValueError("Check the expand input")

