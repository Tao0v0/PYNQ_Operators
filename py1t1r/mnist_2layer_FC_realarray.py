#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pynn.model import model
from pynn.layer import dense
from pynn.loss import cross_entropy_softmax_loss
from pynn.optimizer import RMSprop

from pynn.backend import xbar #from pynn.backend import xbar_16bit as xbar
from pynn.base import multi_array #from pynn.base import multi_array_16bit as multi_array
from pynn.array_real import array_real #from pynn.array_real import array_real10_16bit as array_real

import numpy as np
import scipy.io
from helpers import path, NN_data_save
from tqdm import tqdm

# import good rows/cols
data = np.load(path.array_pick_8_5_fast) #data = np.load(path.array_pick_18_1_fast)
good_rows = data["good_rows"]
good_cols = data["good_cols"]

# SRREF0 < 63 is to suppress VMM current with transistor resistance
base = multi_array(array_real(path.bit, vmmread_refh = 35, v_read_vmm = 63, vmmread_srref0 = 49, vmmread_tiagain = 0, readmode='fast', vmmode = 'vmmsingle'), good_rows, good_cols)
m = model(xbar(base))
m.backend.ratio_G_W = 100e-6
m.backend.ratio_Vg_G = 10 / 300e-6

# Add layers
m.add(
    dense(40, input_dim=60, activation="relu", bias_config=[0, 0]), net_corner=[0, 0]
)
m.add(
    dense(10, activation="stable_softmax", bias_config=[0, 0]), net_corner=[0, 40]
)

# Auxilary
m.summary()

no_epoches = 1  # No. of epoches
ys_test_hardware = []  # Initialize inference results
accuracy = []  # Initialize accuracy results

# Fit
m.compile(cross_entropy_softmax_loss(), RMSprop(lr=0.01), save=True)

print("Begin of loaidng data....")

# train set
mat_contents = scipy.io.loadmat(path.dataset_training)
xs_train = mat_contents["data_0d2"][:, 0:5120]
xs_train = np.delete(xs_train, [0, 7, 56, 63], 0)

# train labels
mat_contents = scipy.io.loadmat(path.dataset_training_labels)
ys_train = mat_contents["mnist_train_labels"][:, 0:5120]

# test set
mat_contents = scipy.io.loadmat(path.dataset_test)
xs_test = mat_contents["data_0d2"][:, :]
xs_test = np.delete(xs_test, [0, 7, 56, 63], 0)

# test labels
mat_contents = scipy.io.loadmat(path.dataset_test_labels)
ys_test = mat_contents["mnist_test_labels"][:, :]

print("End of loaidng data....")

for epo in tqdm(range(no_epoches)):
    # Train
    m.fit(xs_train, ys_train, batch_size=128, epochs=1)

    # Inference
    #ys_test_hardware.append(m.predict(xs_test, batch_size=128))
    #accuracy.append(m.evaluate(ys_test_hardware[epo], ys_test))

import pickle
del m.backend.base.array.input_buffer_setreset
del m.backend.base.array.output_buffer_setreset
del m.backend.base.array.input_buffer_vmm
del m.backend.base.array.output_buffer_vmm
del m.backend.base.array.output_buffer_read
del m.backend.base.array.overlay
del m.backend.base.array.dma
del m.backend.base.array.top_ip
# pickle.dump(m, open(path.tests_folder + 'apr27_npu18-5_NN/NN_refh35_vreadvmm63_srref049_modefast.pickle', 'wb'))

NN_data_save(
    path=path.tests_folder + 'apr27_npu18-5_NN_realarray/',
    split_n=4,
    loss=m.v.loss_plot,
    accuracy=m.v.accuracy_plot,
    G_hist=m.backend.G_hist,
    V_reset_hist=m.backend.V_reset_hist,
    Vg_set_hist=m.backend.Vg_set_hist,
    V_hist=m.backend.V_hist,
    I_hist=m.backend.I_hist,
    ys_test_hardware=ys_test_hardware,
    ys_test_accuracy=accuracy,
)

# pdb.set_trace()

input("Ending......")