#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pynn.base import memristor_array
from pynq import Overlay
from pynq import Xlnk
import numpy as np
from helpers import dout2v
import time

import pdb ###########################################################

class array_real(memristor_array):

    # Real subarray

    def __init__(
        self,
        overlay_addr,
        NUM_WL=128,
        NUM_BL=64,
        VDD=5,
        setreset_srref1=32,
        setreset_srref2=25,
        setreset_pulse_width=127,
        vmmread_refh=39,
        vmmread_refl=31,
        vmmread_srref0=63,
        read_srref2=39,
        vmmread_tiagain=0,
        v_read_vmm=31,
        readmode="fast",
        vmmode = "vmmsplit"
    ):
        """
          Creates a real memristor array.
        """
        # Global parameters
        self.NUM_WL = NUM_WL
        self.NUM_BL = NUM_BL
        self.VDD = VDD  # in V

        # Global AXI-Lite parameters
        self.mode_sel = 0  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.setreset_srref1 = setreset_srref1  # V_BL for set, 6-bit, [0, 63]
        self.setreset_srref2 = setreset_srref2  # V_WL for reset, 6-bit, [0, 63] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # for both set and reset, 8-bit, [0, 255]
        self.setreset_pulse_width = setreset_pulse_width
        self.vmmread_refh = vmmread_refh  # REF_H for vmm, 6-bit, [0, 63]
        # REF_L for vmm and read, 6-bit, [0, 63]
        self.vmmread_refl = vmmread_refl
        self.vmmread_srref0 = vmmread_srref0  # V_gate for vmm and read, [0, 63]
        self.read_srref2 = read_srref2  # V_WL for read, [0, 63]
        self.vmmread_tiagain = vmmread_tiagain  # TIA gain, 2-bit, [0, 3]

        # Non AXI-Lite parameters
        self.v_read_vmm = v_read_vmm  # VMM read voltage, 6-bit, [0, 63]
        self.readmode = readmode
        self.vmmode = vmmode

        # Load overlay
        self.overlay = Overlay(overlay_addr)
        self.dma = self.overlay.axi_dma_0
        self.top_ip = self.overlay.top_main_0

        # Initialize AXIS IO buffers
        xlnk = Xlnk()
        self.input_buffer_setreset = xlnk.cma_array(
            shape=(self.NUM_WL * self.NUM_BL,), dtype=np.uint8
        )
        self.output_buffer_setreset = self.input_buffer_setreset
        self.input_buffer_vmm = xlnk.cma_array(
            shape=(NUM_WL * (NUM_WL + 1),), dtype=np.uint8
        )
        self.output_buffer_vmm = xlnk.cma_array(
            shape=(NUM_BL * (NUM_WL + 1),), dtype=np.uint8
        )
        self.output_buffer_read = xlnk.cma_array(
            shape=(self.NUM_WL * self.NUM_BL,), dtype=np.uint8
        )

        # Write AXI-Lite global parameters
        self.top_ip.write(0x00, self.mode_sel)
        self.top_ip.write(0x04, self.setreset_srref1)
        self.top_ip.write(0x08, self.setreset_srref2)
        self.top_ip.write(0x0C, self.setreset_pulse_width)
        self.top_ip.write(0x10, self.vmmread_refh)
        self.top_ip.write(0x14, self.vmmread_refl)
        self.top_ip.write(0x18, self.vmmread_srref0)
        self.top_ip.write(0x1C, self.read_srref2)
        self.top_ip.write(0x20, self.vmmread_tiagain)

    def read_conductance(self):
        """
        Read top module
        """
        if self.readmode == "slow":
            # Slow read
            #temp2 = self.vmmread_tiagain
            #self.vmmread_tiagain = 1 ####################################################
            #self.top_ip.write(0x20, self.vmmread_tiagain) ####################################################
            #self.top_ip.write(0x18, 53)

            i_tia, dout_map = self.read_conductance_single()
            
            #self.vmmread_tiagain = temp2
            #self.top_ip.write(0x20, self.vmmread_tiagain) ####################################################
            #self.top_ip.write(0x18, self.vmmread_srref0)

            v_read = dout2v(self.read_srref2 - self.vmmread_refl, self.VDD)

            print('mean dout read =', np.mean(dout_map))
        
        elif self.readmode == "fast":
            # Fast read, with diagonal input
            v_diagonal = np.diag(np.ones(self.NUM_WL, dtype=np.uint8) * self.v_read_vmm)

            temp1 = self.vmmode
            #temp2 = self.vmmread_tiagain
            self.vmmode = "vmmraw"
            #self.vmmread_tiagain = 1 ####################################################
            #self.top_ip.write(0x20, self.vmmread_tiagain) ####################################################
            #self.top_ip.write(0x18, 53) ####################################################
            i_tia, _, dout_map = self.read_current(v_diagonal)
            self.vmmode = temp1
            #self.vmmread_tiagain = temp2 ####################################################
            #self.top_ip.write(0x20, self.vmmread_tiagain) ####################################################
            #self.top_ip.write(0x18, self.vmmread_srref0) ####################################################

            v_read = dout2v(
                self.v_read_vmm / 63 * (self.vmmread_refh - self.vmmread_refl), self.VDD,
            )
        else:
            raise ValueError("Wrong read mode.")

        # Array conductance in S
        g = i_tia / v_read

        # Force negative g_map zero, if vmm read
        if self.readmode == "fast":
            g[g < 0] = 0

        return g, dout_map

    def read_conductance_single(self):
        """
        Single read core module
        Return: TIA currents (in A) matrix
        """
        # Single read module enable
        self.top_ip.write(0x00, 3)

        # Start receive data
        self.dma.recvchannel.transfer(self.output_buffer_read)
        self.dma.recvchannel.wait()
        # DOUT to currents
        dout_map = np.copy(self.output_buffer_read).reshape((self.NUM_WL, self.NUM_BL))
        r_tia = 1000 * (10 ** self.vmmread_tiagain)  # TIA resistor in Ohm
        # TIA out in V (wrt virtual ground)
        v_tia_out = ((63 - dout_map) / 63) * dout2v(self.vmmread_refl, self.VDD)
        i_tia = v_tia_out / r_tia

        # RESET FPGA
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 0)
        time.sleep(0.01)

        return i_tia, dout_map

    def read_current(self, V_in):
        """
        VMM module
        V_in: 2D array, each column is an input vector.
        3 MODES
        vmmode "vmmsingle": inputs are [-31, 31]. To be shifted to [0, 62]. Each column is a vector (size 128), total 129 columns.
        vmmode "vmmsplit" : inputs are [-31, 31]. To be shifted to [0, 62].
        vmmode "vmmraw": inputs are [0 63].
        """

        # Check V_in dimension, must be 128 x 128
        if V_in.shape != (self.NUM_WL, self.NUM_WL):
            raise ValueError("VMM input dimension mismatch with pre-defined buffer.")

        # Add the compensation vector (the vector of all 0)
        V_in_comp = np.concatenate((V_in, np.zeros((self.NUM_WL, 1))), axis=1)
        # For VMM, shift V_in from [-31, 31] to [0, 62]
        if self.vmmode in ["vmmsingle", "vmmsplit"]:
            V_in_comp = V_in_comp + 31
            if np.any(V_in_comp < 0) or np.any(V_in_comp > 62):
                raise ValueError("Shifted VMM input should be within -31 to 31.")
        elif self.vmmode == "vmmraw":
            if np.any(V_in_comp < 0) or np.any(V_in_comp > 63):
                raise ValueError("Raw VMM input should be within 0-63.")
        else:
            raise ValueError("Wrong VMM mode.")

        # For VMM with splits
        if self.vmmode == "vmmsplit":
            for i in range(8):
                V_in_slice = np.zeros((self.NUM_WL, self.NUM_WL + 1))
                V_in_slice[i::8, :] = V_in_comp[i::8, :]
                if i == 0:
                    dout_map_comp, dout_map = self.read_current_single(V_in_slice)
                else:
                    temp, dout_map = self.read_current_single(V_in_slice)
                    dout_map_comp = dout_map_comp - (63 - temp)
        else:  # "vmmsingle" or "vmmraw"
            dout_map_comp, dout_map = self.read_current_single(V_in_comp)

        # DOUT to currents
        r_tia = 1000 * (10 ** self.vmmread_tiagain)  # TIA resistor in Ohm
        # TIA out in V (wrt virtual ground)
        v_tia_out = ((63 - dout_map_comp) / 63) * dout2v(self.vmmread_refl, self.VDD)
        i_tia = v_tia_out / r_tia

        # Current scaling (from V nominal: [-31, 31] to V real: [-31/63*(refh-refl)/63*VDD, 31/63*(refh-refl)/63*VDD]) to)
        if self.vmmode in ["vmmsingle", "vmmsplit"]:
            scaling = dout2v(self.vmmread_refh - self.vmmread_refl, self.VDD) / 63
            i_tia = i_tia / scaling
            # Each column is a VMM result vector, for upper level codes
            i_tia = i_tia.T

        return i_tia, dout_map_comp, dout_map

    def read_current_single(self, V_in):

        # VMM module enable
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 2)

        # Flatten, and copy to buffer
        self.input_buffer_vmm[:] = np.copy(V_in.flatten("F"))

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_vmm)
        self.dma.recvchannel.transfer(self.output_buffer_vmm)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Reshape DOUT, calculate compensated DOUT
        dout_map = np.copy(self.output_buffer_vmm).reshape((self.NUM_WL + 1, self.NUM_BL))
        dout_map_comp = 63 + np.int32(dout_map[0 : self.NUM_WL, :]) - dout_map[-1, :]

        print(f'vmm mode = {self.vmmode} gain = {self.vmmread_tiagain}') ####################################################
        print('mean dout rest =', np.mean(dout_map[0 : self.NUM_WL, :]), 'mean dout last =', np.mean(dout_map[-1, :]))

        # RESET FPGA
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 0)
        time.sleep(0.01)

        return dout_map_comp, dout_map

    def update_conductance(self, V_WL, V_gate):
        """
        vmmode | V_WL | V_gate
        00   | 0    | 0      | skip
        01   | 0    | >0     | SET
        02   | >0   | 0      | RESET
        03   | >0   | >0     | RESET + SET (slightly poor than RESET followed by SET)

        Parameters
        ----------
        V_WL : TYPE 2D array
            DESCRIPTION.  WL voltage
        V_gate : TYPE 2D array
            DESCRIPTION.  Gate voltage

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Expand inputs
        V_WL = self.expand(V_WL)
        V_gate = self.expand(V_gate)

        # Compute set_reset_mode matrix
        index_SET = V_gate > 0
        index_RESET = V_WL > 0
        index_RESET_only = np.bitwise_and(index_RESET, ~index_SET)
        set_reset_mode = (index_RESET << 7) + (index_SET << 6)

        # Combine set_reset_mode and 6-bit voltages
        D_in = (
            set_reset_mode
            + np.where(index_RESET_only, V_WL, 0)
            + np.where(index_SET, V_gate, 0)
        )
        self.input_buffer_setreset[:] = D_in.flatten("F")

        # SET/RESET module enable
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 1)

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_setreset)
        self.dma.recvchannel.transfer(self.output_buffer_setreset)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Output validation
        if not (np.array_equal(self.input_buffer_setreset, self.output_buffer_setreset)):
            raise ValueError("SET/RESET AXIS communication error.")

        # RESET FPGA
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 0)
        time.sleep(0.01)

    def expand(self, value):
        """
        Expand return array to same size of (NUM_WL, NUM_BL) with value (input)
        """
        if np.shape(value) == ():
            # value is a number
            return np.ones((self.NUM_WL, self.NUM_BL)) * value
        elif np.shape(value) == (self.NUM_WL, self.NUM_BL):
            # sizes match
            return value

        raise ValueError("Check the expand input")

class array_real10(memristor_array):

    # Real subarray

    def __init__(
        self,
        overlay_addr,
        NUM_WL=128,
        NUM_BL=64,
        BATCH_SIZE_VMM=129,
        BATCH_SIZE_VMM2=129,
        NUM_BL_BLK=4,  # VMM2 partition
        NUM_WL_BLK=8,  # VMM2 partition
        VDD=5,
        setreset_srref1=32,
        setreset_srref2=25,
        setreset_pulse_width=127,
        vmmread_refh=39,
        vmmread_refl=31,
        vmmread_srref0=63,
        read_srref2=39,
        vmmread_tiagain=0,
        v_read_vmm=31,
        readmode="fast",
        vmmode = "vmm1"
    ):
        """
        Creates a real memristor array.
        """
        # Global parameters
        self.NUM_WL = NUM_WL  # No. of word lines
        self.NUM_BL = NUM_BL  # No. of bit lines
        self.VDD = VDD  # VDD in V
        self.BATCH_SIZE_VMM = BATCH_SIZE_VMM  # VMM batch size
        self.BATCH_SIZE_VMM2 = BATCH_SIZE_VMM2  # VMM2 batch size
        self.NUM_BL_BLK = NUM_BL_BLK  # VMM2 bitline wise blocks
        self.NUM_WL_BLK = NUM_WL_BLK  # VMM2 wordline wise blocks

        # Global AXI-Lite parameters
        self.mode_sel = 0  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.setreset_srref1 = setreset_srref1  # V_BL for set, 6-bit, [0, 63]
        self.setreset_srref2 = setreset_srref2  # V_WL for reset, 6-bit, [0, 63]
        # for both set and reset, 8-bit, [0, 255]
        self.setreset_pulse_width = setreset_pulse_width
        self.vmmread_refh = vmmread_refh  # REF_H for vmm, 6-bit, [0, 63]
        # REF_L for vmm and read, 6-bit, [0, 63]
        self.vmmread_refl = vmmread_refl
        self.vmmread_srref0 = vmmread_srref0  # V_gate for vmm and read, [0, 63]
        self.read_srref2 = read_srref2  # V_WL for read, [0, 63]
        self.vmmread_tiagain = vmmread_tiagain  # TIA gain, 2-bit, [0, 3]

        # Non AXI-Lite parameters
        self.v_read_vmm = v_read_vmm  # VMM read voltage, 6-bit, [0, 63]
        self.readmode = readmode
        self.vmmode=vmmode

        # Load overlay
        self.overlay = Overlay(overlay_addr)
        self.dma = self.overlay.axi_dma_0
        self.top_ip = self.overlay.top_main_0

        # Initialize AXIS IO buffers
        xlnk = Xlnk()
        self.input_buffer_setreset = xlnk.cma_array(
            shape=(self.NUM_WL * self.NUM_BL,), dtype=np.uint8
        )
        self.output_buffer_setreset = self.input_buffer_setreset
        self.input_buffer_vmm = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM * self.NUM_WL,), dtype=np.uint8
        )
        self.output_buffer_vmm = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM * self.NUM_BL,), dtype=np.uint8
        )
        self.output_buffer_read = xlnk.cma_array(
            shape=(self.NUM_WL * self.NUM_BL,), dtype=np.uint8
        )
        self.input_buffer_vmm2 = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM2 * self.NUM_WL,), dtype=np.uint8
        )
        self.output_buffer_vmm2 = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM2 * self.NUM_WL_BLK * self.NUM_BL,), dtype=np.uint8,
        )

        # Write AXI-Lite global parameters
        self.top_ip.write(0x00, self.mode_sel)
        self.top_ip.write(0x04, self.setreset_srref1)
        self.top_ip.write(0x08, self.setreset_srref2)
        self.top_ip.write(0x0C, self.setreset_pulse_width)
        self.top_ip.write(0x10, self.vmmread_refh)
        self.top_ip.write(0x14, self.vmmread_refl)
        self.top_ip.write(0x18, self.vmmread_srref0)
        self.top_ip.write(0x1C, self.read_srref2)
        self.top_ip.write(0x20, self.vmmread_tiagain)

    def read_conductance(self):
        """
        Read top module
        """
        if self.readmode == "slow":
            # Slow read
            i_tia, dout_map = self.read_conductance_single()
            v_read = dout2v(self.read_srref2 - self.vmmread_refl, self.VDD)
        elif self.readmode == "fast":
            # Fast read, with diagonal input
            v_diagonal = np.diag(np.ones(self.NUM_WL, dtype=np.uint8) * self.v_read_vmm)

            temp = self.vmmode
            self.vmmode = "vmm1raw"
            i_tia, _, dout_map = self.read_current(v_diagonal)
            self.vmmode = temp

            v_read = dout2v(
                self.v_read_vmm / 63 * (self.vmmread_refh - self.vmmread_refl), self.VDD,
            )
        else:
            raise ValueError("Wrong read mode.")

        # Array conductance in S
        g = i_tia / v_read

        # Force negative g_map zero, if vmm read
        if self.readmode == "fast":
            g[g < 0] = 0

        return g, dout_map

    def read_conductance_single(self):
        """
        Single read core module
        Return: TIA currents (in A) matrix
        """
        # Single read module enable
        self.top_ip.write(0x00, 3)

        # Start receive data
        self.dma.recvchannel.transfer(self.output_buffer_read)
        self.dma.recvchannel.wait()

        # DOUT to currents
        dout_map = np.copy(self.output_buffer_read).reshape((self.NUM_WL, self.NUM_BL))
        r_tia = 1000 * (10 ** self.vmmread_tiagain)  # TIA resistor in Ohm
        v_tia_out = ((63 - dout_map) / 63) * dout2v(
            self.vmmread_refl, self.VDD
        )  # TIA out in V (wrt virtual ground)
        i_tia = v_tia_out / r_tia

        # RESET FPGA
        self.top_ip.write(0x00, 0)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        time.sleep(0.01)

        return i_tia, dout_map

    def read_current(self, V_in):
        """
        VMM module
        V_in: 2D array, each column is an input vector.
        4 MODES
        vmmode "vmm1": inputs are [-31V, 31V]. To be shifted to [0V, 62V]. I_out are scaled as if the V_in are [-31V, 31V].
        vmmode "vmm2" : inputs are [-31V, 31V]. To be shifted to [0V, 62V].I_out are scaled as if the V_in are [-31V, 31V].
        vmmode "vmm1raw": inputs are [0 63], corresponding to [V_REFL, V_REFH].
        vmmode "vmm2raw": inputs are [0 63], corresponding to [V_REFL, V_REFH].
        """

        # Check V_in dimension
        if self.vmmode in ["vmm1", "vmm1raw"]:
            batch_size = self.BATCH_SIZE_VMM  # VMM (no split) batch size
        elif self.vmmode in ["vmm2", "vmm2raw"]:
            batch_size = self.BATCH_SIZE_VMM2  # VMM2 split batch size
        else:
            raise ValueError("Wrong VMM mode.")

        if V_in.shape != (batch_size - 1, self.NUM_WL):  # -1 for comp vec
            raise ValueError("VMM input dimension mismatch with pre-defined buffer.")

        # Add the compensation vector (the vector of all 0)
        V_in_comp = np.concatenate((V_in, np.zeros((self.NUM_WL, 1))), axis=1)

        # Shift V_in from [-31, 31] to [0, 62], check V_in range
        if self.vmmode in ["vmm1", "vmm2"]:
            V_in_comp = V_in_comp + 31
            if np.any(V_in_comp < 0) or np.any(V_in_comp > 62):
                raise ValueError("Shifted VMM input must be [-31, 31]")
        else:  # Case "vmm1raw" or "vmm2raw"
            if np.any(V_in_comp < 0) or np.any(V_in_comp > 63):
                raise ValueError("Raw VMM input must be [0, 63]")

        # Hardware call
        if self.vmmode in ["vmm1", "vmm1raw"]:
            dout_map_comp, dout_map = self.read_current_single(V_in_comp)
        else:  # "vmm2" or "vmm2raw"
            dout_map_comp, dout_map = self.read_current2_single(V_in_comp)

        # DOUT_comp to currents
        r_tia = 1000 * (10 ** self.vmmread_tiagain)  # TIA resistor in Ohm
        v_tia_out = ((63 - dout_map_comp) / 63) * dout2v(
            self.vmmread_refl, self.VDD
        )  # TIA out in V (wrt virtual ground)
        i_tia = v_tia_out / r_tia

        # Current scaling (from V nominal: [-31, 31] to V real: [-31/63*(refh-refl)/63*VDD, 31/63*(refh-refl)/63*VDD]) to)
        # Not apply to "vmm1raw" or "vmm2raw"
        if self.vmmode in ["vmm1", "vmm2"]:
            scaling = dout2v(self.vmmread_refh - self.vmmread_refl, self.VDD) / 63
            i_tia = i_tia / scaling
            # Each column is a VMM result vector, for upper level codes
            i_tia = i_tia.T

        return i_tia, dout_map_comp, dout_map

    def read_current_single(self, V_in):

        # VMM module enable
        self.top_ip.write(0x00, 2)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2

        # Flatten, and copy to buffer
        self.input_buffer_vmm[:] = np.copy(V_in.flatten("F"))

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_vmm)
        self.dma.recvchannel.transfer(self.output_buffer_vmm)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Reshape DOUT, calculate compensated DOUT
        dout_map = np.copy(self.output_buffer_vmm).reshape(
            (self.BATCH_SIZE_VMM, self.NUM_BL)
        )
        dout_map_comp = (
            63 + np.int32(dout_map[: self.BATCH_SIZE_VMM - 1, :]) - dout_map[-1, :]
        )

        # RESET FPGA
        self.top_ip.write(0x00, 0)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2
        time.sleep(0.01)

        return dout_map_comp, dout_map  # dout_map is 2D

    def read_current2_single(self, V_in):

        # VMM2 module enable
        self.top_ip.write(0x00, 4)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2

        # Flatten, and copy to buffer
        self.input_buffer_vmm2[:] = np.copy(V_in.flatten("F"))

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_vmm2)
        self.dma.recvchannel.transfer(self.output_buffer_vmm2)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Reshape DOUT, calculate compensated DOUT
        dout_map = np.copy(self.output_buffer_vmm2).reshape(
            ((self.BATCH_SIZE_VMM2, self.NUM_WL_BLK, self.NUM_BL))
        )

        delta_dout_map_3d = 63 - np.int32(dout_map)
        dout_map_2d = 63 - np.sum(delta_dout_map_3d, axis=1)

        dout_map_comp = (
            63
            + np.int32(dout_map_2d[: self.BATCH_SIZE_VMM2 - 1, :])
            - np.tile(dout_map_2d[-1, :], (self.BATCH_SIZE_VMM2 - 1, 1))
        )

        # RESET FPGA
        self.top_ip.write(0x00, 0)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2
        time.sleep(0.01)

        return dout_map_comp, dout_map  # dout_map is 3D.

    def update_conductance(self, V_WL, V_gate):
        """
        vmmode | V_WL | V_gate
        00   | 0    | 0      | skip
        01   | 0    | >0     | SET
        02   | >0   | 0      | RESET
        03   | >0   | >0     | RESET + SET (slightly poor than RESET followed by SET)

        Parameters
        ----------
        V_WL : TYPE 2D array
            DESCRIPTION.  WL voltage
        V_gate : TYPE 2D array
            DESCRIPTION.  Gate voltage

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Expand inputs
        V_WL = self.expand(V_WL)
        V_gate = self.expand(V_gate)

        # Compute set_reset_mode matrix
        index_SET = V_gate > 0
        index_RESET = V_WL > 0
        index_RESET_only = np.bitwise_and(index_RESET, ~index_SET)
        set_reset_mode = (index_RESET << 7) + (index_SET << 6)

        # Combine set_reset_mode and 6-bit voltages
        D_in = (
            set_reset_mode
            + np.where(index_RESET_only, V_WL, 0)
            + np.where(index_SET, V_gate, 0)
        )
        self.input_buffer_setreset[:] = D_in.flatten("F")

        # SET/RESET module enable
        self.top_ip.write(0x00, 1)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_setreset)
        self.dma.recvchannel.transfer(self.output_buffer_setreset)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Output validation
        if not (np.array_equal(self.input_buffer_setreset, self.output_buffer_setreset)):
            raise ValueError("SET/RESET AXIS communication error.")

        # RESET FPGA
        self.top_ip.write(0x00, 0)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        time.sleep(0.01)

    def expand(self, value):
        """
        Expand return array to same size of (NUM_WL, NUM_BL) with value (input)
        """
        if np.shape(value) == ():
            # value is a number
            return np.ones((self.NUM_WL, self.NUM_BL)) * value
        elif np.shape(value) == (self.NUM_WL, self.NUM_BL):
            # sizes match
            return value

        raise ValueError("Check the expand input")

class array_real10_16bit(memristor_array):

    # Real subarray

    def __init__(
        self,
        overlay_addr,
        NUM_WL=128,
        NUM_BL=64,
        BATCH_SIZE_VMM=129,
        BATCH_SIZE_VMM2=129,
        NUM_BL_BLK=4,  # VMM2 partition
        NUM_WL_BLK=8,  # VMM2 partition
        VDD=5,
        setreset_pulse_width=127,
        vmmread_refh=39,
        vmmread_refl=31,
        vmmread_srref0=63,
        read_srref2=39,
        vmmread_tiagain=0,
        v_read_vmm=31,
        readmode="fast",
        vmmode="vmm1raw"

    ):
        """
        Creates a real memristor array.
        """
        # Global parameters
        self.NUM_WL = NUM_WL  # No. of word lines
        self.NUM_BL = NUM_BL  # No. of bit lines
        self.VDD = VDD  # VDD in V
        self.BATCH_SIZE_VMM = BATCH_SIZE_VMM  # VMM batch size
        self.BATCH_SIZE_VMM2 = BATCH_SIZE_VMM2  # VMM2 batch size
        self.NUM_BL_BLK = NUM_BL_BLK  # VMM2 bitline wise blocks
        self.NUM_WL_BLK = NUM_WL_BLK  # VMM2 wordline wise blocks

        # Global AXI-Lite parameters
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.mode_sel = 0
        # for both set and reset, 8-bit, [0, 255]
        self.setreset_pulse_width = setreset_pulse_width
        # REF_H for vmm, 6-bit, [0, 63]
        self.vmmread_refh = vmmread_refh
        # REF_L for vmm and read, 6-bit, [0, 63]
        self.vmmread_refl = vmmread_refl
        # V_gate for vmm and read, [0, 63]
        self.vmmread_srref0 = vmmread_srref0
        # V_WL for read, [0, 63]
        self.read_srref2 = read_srref2
        # TIA gain, 2-bit, [0, 3]
        self.vmmread_tiagain = vmmread_tiagain

        # Non AXI-Lite parameters
        # VMM read voltage, 6-bit, [0, 63]
        self.v_read_vmm = v_read_vmm
        self.readmode = readmode
        self.vmmode=vmmode

        # Load overlay
        self.overlay = Overlay(overlay_addr)
        self.dma = self.overlay.axi_dma_0
        self.top_ip = self.overlay.top_main_0

        # Initialize AXIS IO buffers
        xlnk = Xlnk()
        self.input_buffer_setreset = xlnk.cma_array(
            shape=(self.NUM_WL * self.NUM_BL,), dtype=np.uint16
        )
        self.output_buffer_setreset = self.input_buffer_setreset
        self.input_buffer_vmm = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM * self.NUM_WL,), dtype=np.uint16
        )
        self.output_buffer_vmm = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM * self.NUM_BL,), dtype=np.uint16
        )
        self.output_buffer_read = xlnk.cma_array(
            shape=(self.NUM_WL * self.NUM_BL,), dtype=np.uint16
        )
        self.input_buffer_vmm2 = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM2 * self.NUM_BL_BLK * self.NUM_WL,), dtype=np.uint16
        )
        self.output_buffer_vmm2 = xlnk.cma_array(
            shape=(self.BATCH_SIZE_VMM2 * self.NUM_WL_BLK * self.NUM_BL,),
            dtype=np.uint16,
        )

        # Write AXI-Lite global parameters
        self.top_ip.write(0x00, self.mode_sel)
        self.top_ip.write(0x04, self.setreset_pulse_width)
        self.top_ip.write(0x08, self.vmmread_refh)
        self.top_ip.write(0x0C, self.vmmread_refl)
        self.top_ip.write(0x10, self.vmmread_srref0)
        self.top_ip.write(0x14, self.read_srref2)
        self.top_ip.write(0x18, self.vmmread_tiagain)

    def read_conductance(self):
        """
        Read top module
        """
        if self.readmode == "slow":
            # Slow read
            i_tia, dout_map = self.read_conductance_single()
            v_read = dout2v(self.read_srref2 - self.vmmread_refl, self.VDD)
        elif self.readmode == "fast":
            # Fast read, with diagonal input
            v_diagonal = np.diag(np.ones(self.NUM_WL, dtype=np.uint16) * self.v_read_vmm)

            temp = self.vmmode
            self.vmmode = "vmm1raw"
            i_tia, _, dout_map = self.read_current(v_diagonal)
            self.vmmode = temp
            
            v_read = dout2v(
                self.v_read_vmm / 63 * (self.vmmread_refh - self.vmmread_refl), self.VDD,
            )
        else:
            raise ValueError("Wrong read mode.")

        # Array conductance in S
        g = i_tia / v_read

        # Force negative g_map zero, if vmm read
        if self.readmode == "fast":
            g[g < 0] = 0

        return g, dout_map

    def read_conductance_single(self):
        """
        Single read core module
        Return: TIA currents (in A) matrix
        """
        # Single read module enable
        self.top_ip.write(0x00, 3)

        # Start receive data
        self.dma.recvchannel.transfer(self.output_buffer_read)
        self.dma.recvchannel.wait()

        # DOUT to currents
        dout_map = np.copy(self.output_buffer_read).reshape((self.NUM_WL, self.NUM_BL))
        # TIA resistor in Ohm
        r_tia = 1000 * (10 ** self.vmmread_tiagain)
        # TIA out in V (wrt virtual ground)
        v_tia_out = ((63 - dout_map) / 63) * dout2v(self.vmmread_refl, self.VDD)
        i_tia = v_tia_out / r_tia

        # RESET FPGA
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 0)
        time.sleep(0.01)

        return i_tia, dout_map

    def read_current(self, V_in):
        """
        VMM module
        V_in: 2D array, each column is an input vector.
        4 MODES
        vmmode "vmm1": inputs are [-31V, 31V]. To be shifted to [0V, 62V]. I_out are scaled as if the V_in are [-31V, 31V].
        vmmode "vmm2" : inputs are [-31V, 31V]. To be shifted to [0V, 62V]. I_out are scaled as if the V_in are [-31V, 31V].
        vmmode "vmm1raw": inputs are [0 63], corresponding to [V_REFL, V_REFH].
        vmmode "vmm2raw": inputs are [0 63], corresponding to [V_REFL, V_REFH].
        """

        # Check V_in dimension
        if self.vmmode in ["vmm1", "vmm1raw"]:
            batch_size = self.BATCH_SIZE_VMM  # VMM (no split) batch size
        elif self.vmmode in ["vmm2", "vmm2raw"]:
            batch_size = self.BATCH_SIZE_VMM2  # VMM2 split batch size
        else:
            raise ValueError("Wrong VMM mode.")
        # -1 for comp vec
        if V_in.shape != (batch_size - 1, self.NUM_WL):
            raise ValueError("VMM input dimension mismatch with pre-defined buffer.")

        # Add the compensation vector (the vector of all 0)
        V_in_comp = np.concatenate((V_in, np.zeros((self.NUM_WL, 1))), axis=1)

        # Shift V_in from [-31, 31] to [0, 62], check V_in range
        if self.vmmode in ["vmm1", "vmm2"]:
            V_in_comp = V_in_comp + 31
            if np.any(V_in_comp < 0) or np.any(V_in_comp > 62):
                raise ValueError("Shifted VMM input must be [-31, 31]")
        else:  # Case "vmm1raw" or "vmm2raw"
            if np.any(V_in_comp < 0) or np.any(V_in_comp > 63):
                raise ValueError("Raw VMM input must be [0, 63]")

        # Hardware call
        if self.vmmode in ["vmm1", "vmm1raw"]:
            dout_map_comp, dout_map = self.read_current_single(V_in_comp)
        else:  # "vmm2" or "vmm2raw"
            dout_map_comp, dout_map = self.read_current2_single(V_in_comp)

        # DOUT_comp to currents
        # TIA resistor in Ohm
        r_tia = 1000 * (10 ** self.vmmread_tiagain)
        # TIA out in V (wrt virtual ground)
        v_tia_out = ((63 - dout_map_comp) / 63) * dout2v(self.vmmread_refl, self.VDD)
        i_tia = v_tia_out / r_tia

        # Current scaling (from V nominal: [-31, 31] to V real: [-31/63*(refh-refl)/63*VDD, 31/63*(refh-refl)/63*VDD]) to)
        # Not apply to "vmm1raw" or "vmm2raw"
        if self.vmmode in ["vmm1", "vmm2"]:
            scaling = dout2v(self.vmmread_refh - self.vmmread_refl, self.VDD) / 63
            i_tia = i_tia / scaling
            # Each column is a VMM result vector, for upper level codes
            i_tia = i_tia.T

        return i_tia, dout_map_comp, dout_map

    def read_current_single(self, V_in):

        # VMM module enable
        self.top_ip.write(0x00, 2)  # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2

        # Flatten, and copy to buffer
        self.input_buffer_vmm[:] = np.copy(V_in.flatten("F"))

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_vmm)
        self.dma.recvchannel.transfer(self.output_buffer_vmm)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Reshape DOUT, calculate compensated DOUT
        dout_map = np.copy(self.output_buffer_vmm).reshape(
            (self.BATCH_SIZE_VMM, self.NUM_BL)
        )
        dout_map_comp = (
            63 + np.int32(dout_map[: self.BATCH_SIZE_VMM - 1, :]) - dout_map[-1, :]
        )

        # RESET FPGA
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2
        self.top_ip.write(0x00, 0)
        time.sleep(0.01)

        return dout_map_comp, dout_map  # dout_map is 2D

    def read_current2_single(self, V_in):

        # VMM2 module enable
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2
        self.top_ip.write(0x00, 4)

        # Copy input matrix (each column a vector) horizontally
        V_in = np.tile(V_in, (1, self.NUM_BL_BLK))

        # Split input matrix (each column a vector) vertically
        V_in_wlblk_list = np.vsplit(V_in, self.NUM_WL_BLK)

        # Put the vertical splits horizontally and flatten
        V_in = np.hstack(V_in_wlblk_list)
        self.input_buffer_vmm2[:] = np.copy(V_in.flatten('F'))

        print('V_in shape = ', np.shape(V_in))

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_vmm2)
        self.dma.recvchannel.transfer(self.output_buffer_vmm2)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Reshape DOUT, calculate compensated DOUT
        dout_map = np.copy(self.output_buffer_vmm2).reshape(
            ((self.NUM_WL_BLK, self.NUM_BL_BLK, self.BATCH_SIZE_VMM2, self.NUM_BL // self.NUM_BL_BLK))
        )

        delta_dout_map_4d = 63 - np.int32(dout_map)
        dout_map_3d = 63 - np.sum(delta_dout_map_4d, axis=0)
        dout_map_3d = np.swapaxes(dout_map_3d, 0, 1)
        dout_map_2d = np.reshape(dout_map_3d, (self.BATCH_SIZE_VMM2, self.NUM_BL))

        dout_map_comp = (
            63 + np.int32(dout_map_2d[: self.BATCH_SIZE_VMM2 - 1, :]) - dout_map_2d[-1, :]
        )

        # RESET FPGA
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ, 4: VMM2
        self.top_ip.write(0x00, 0)
        time.sleep(0.01)

        return dout_map_comp, dout_map  # dout_map is 3D.

    def update_conductance(self, V_gate, V_BL, V_WL):
        """
        Mode | V_WL | V_gate
        00   | 0    | 0      | skip
        01   | 0    | >0     | SET
        02   | >0   | 0      | RESET

        Parameters
        ----------
        V_gate : TYPE 2D array
            DESCRIPTION.  Gate voltage
        V_BL : TYPE 2D array
            DESCRIPTION.  BL voltage
        V_WL : TYPE 2D array
            DESCRIPTION.  WL voltage
        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Expand inputs
        V_gate = np.int16(self.expand(V_gate))
        V_BL = np.int16(self.expand(V_BL))
        V_WL = np.int16(self.expand(V_WL))

        # Compute set_reset_mode matrix
        index_SET = V_BL > 0
        index_RESET = V_WL > 0

        # Check if any device both SET and RESET
        assert not np.any(np.bitwise_and(index_RESET, index_SET))

        # Prepare SET/RESET
        set_reset_mode = (index_RESET << 15) + (index_SET << 14)

        # Combine set_reset_mode and 6-bit voltages
        D_in = (
            set_reset_mode
            + (V_gate << 8)
            + np.where(index_RESET, V_WL, 0)
            + np.where(index_SET, V_BL, 0)
        )
        self.input_buffer_setreset[:] = D_in.flatten("F")

        # SET/RESET module enable
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 1)

        # AXIS Communication
        self.dma.sendchannel.transfer(self.input_buffer_setreset)
        self.dma.recvchannel.transfer(self.output_buffer_setreset)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Output validation
        if not (np.array_equal(self.input_buffer_setreset, self.output_buffer_setreset)):
            raise ValueError("SET/RESET AXIS communication error.")

        # RESET FPGA
        # mode_sel[1:0], 1: SETRESET, 2: VMM, 3: READ
        self.top_ip.write(0x00, 0)
        time.sleep(0.01)

    def expand(self, value):
        """
        Expand return array to same size of (NUM_WL, NUM_BL) with value (input)
        """
        if np.shape(value) == ():
            # value is a number
            return np.ones((self.NUM_WL, self.NUM_BL)) * value
        elif np.shape(value) == (self.NUM_WL, self.NUM_BL):
            # sizes match
            return value

        raise ValueError("Check the expand input")
