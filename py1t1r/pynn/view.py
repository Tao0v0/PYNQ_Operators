#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib

# matplotlib.use("AGG")  表示无界面后端，不会弹出任何图形窗口，适用于服务器或没有显示器的环境
import matplotlib.pyplot as plt


class view:
    def __init__(self):
        self.DEBUG = 0
        self.DRAW_PLOT = 1
        self.len = 0

        self.loss_plot = []  # Loss
        self.accuracy_plot = []  # Accuracy

        return

    def display(self, text):
        if self.DEBUG:
            print(text)

    def save(self, loss, accuracy):
        #  Loss save
        self.loss_plot.append(loss)

        #  Accuracy save
        self.accuracy_plot.append(accuracy)

    def plot(self):

        self.fig1, self.ax1 = plt.subplots(1, 2)
        self.fig1.suptitle("In-batch Performance")

        self.ax1[0].plot(self.loss_plot)
        self.ax1[0].set_xlabel("No. of minibatches")
        self.ax1[0].set_ylabel("Loss per sample")
        self.ax1[1].plot(self.accuracy_plot)
        self.ax1[1].set_xlabel("No. of minibatches")
        self.ax1[1].set_ylabel("In-minibatch accuracy")
        self.fig1.tight_layout()
        self.fig1.show()

        # self.fig1.canvas.draw()
        # self.fig1.canvas.flush_events()

    def plotG_Vg_Vreset(self, G, Vreset, Vg):

        self.fig2, self.ax2 = plt.subplots(1, 3)  # G, Vg, V
        self.fig2.suptitle("G, Vreset and Vg")
        # self.fig2.show()

        # G matrix plot
        self.ax2[0].imshow(G)
        self.ax2[0].set_title("G matrix")

        # RESET voltage matrix plot
        self.ax2[1].imshow(Vreset)
        self.ax2[1].set_title("Reset voltages")

        # Gate voltage matrix plot
        self.ax2[2].imshow(Vg)
        self.ax2[2].set_title("Gate voltages")
        self.fig2.tight_layout()

        # self.fig2.canvas.draw()
        # self.fig2.canvas.flush_events()
