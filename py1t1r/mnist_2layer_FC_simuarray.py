#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pynn.model import model
from pynn.layer import dense
from pynn.loss import cross_entropy_softmax_loss
from pynn.optimizer import RMSprop

#from pynn.backend import xbar 
from pynn.backend import xbar_16bit as xbar
#from pynn.base import multi_array
from pynn.base import multi_array_16bit as multi_array
#from pynn.array_simu import array_simu
from pynn.array_simu import array_simu_16bit as array_simu

import numpy as np
import scipy.io
from helpers import path, NN_data_save
from tqdm import tqdm

# Simu array initialize
def update_fun(G, V, Vt):
    noise = 0.001
    return (
        -G * (V != 0)
        + (V > 0) * np.maximum(200e-6 / 13 * (Vt - 6), G)
        + noise * np.random.randn(*np.shape(G)) * 400e-6
    )

if __name__ == '__main__':

    # import good rows/cols
    good_rows = np.arange(128)
    good_cols = np.arange(64)

    g0 = ["random", [128, 64], 50e-6, 100e-6]

    # SRREF0 < 63 is to suppress VMM current with transistor resistance
    base = multi_array(array_simu(g0, update_fun, gmin = 0.0, gmax = 2e-4), good_rows, good_cols)
    m = model(xbar(base))
    m.backend.ratio_G_W = 100e-6
    m.backend.ratio_Vg_G = 13 / 200e-6

    # Add layers
    m.add(
        dense(40, input_dim=60, activation="relu", bias_config=[0, 0]), net_corner=[0, 0]
    )
    m.add(
        dense(10, activation="stable_softmax", bias_config=[0, 0]), net_corner=[0, 40]
    )

    # Auxilary
    m.summary()

    no_epoches = 2  # No. of epoches
    ys_test_hardware = []  # Initialize inference results
    accuracy = []  # Initialize accuracy results

    # Fit
    m.compile(cross_entropy_softmax_loss(), RMSprop(lr=0.01), save=True)

    print("Begin of loaidng data....")

    # train set
    mat_contents = scipy.io.loadmat(path.dataset_training)
    xs_train = mat_contents["data_0d2"][:, :]
    xs_train = np.delete(xs_train, [0, 7, 56, 63], 0)

    # train labels
    mat_contents = scipy.io.loadmat(path.dataset_training_labels)
    ys_train = mat_contents["mnist_train_labels"][:, :]

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
        ys_test_hardware.append(m.predict(xs_test, batch_size=100))
        accuracy.append(m.evaluate(ys_test_hardware[epo], ys_test))

    import pickle
    pickle.dump(m, open(path.tests_folder + 'apr28_simuNN/NN_refh35_vreadvmm63_srref049_modefast.pickle', 'wb'))

    input("Ending......")