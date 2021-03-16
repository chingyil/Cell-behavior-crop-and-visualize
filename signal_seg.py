from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import csv
import pandas as pd
import numpy as np
import cv2
from itertools import product

# Finding the smallest index which is >= the input
def get_idx_from_x(ts, t, eps=5e-3):
    assert len(ts) > 0
    if type(ts) is list:
        return sum(np.array(ts) < t - eps)
    elif type(ts) is np.ndarray:
        return min(np.searchsorted(ts, t - eps), len(ts) - 1)
    raise ValueError("Unsupport type %s" % type(ts))

def get_y_from_t(ts, t, ys):
    tidx = sum(ts < t)
    return ys[tidx]

class SnaptoCursor(object):
    def __init__(self, ax0, ax1, t, y, lopt_fused, slopes, cdiff):
        self.ax0 = ax0
        self.ax1 = ax1
        self.t = t
        self.y = y
        self.lopt_fused = lopt_fused
        self.slopes = slopes
        self.cdiff = cdiff
        self.highlighted = [False] * len(lopt_fused)

    def on_click(self, event):

        if event.inaxes == self.ax0:
            print(event.xdata, event.ydata, event.inaxes)
            idx_fused = get_idx_from_x(self.lopt_fused, event.xdata) - 1

        elif event.inaxes == self.ax1:
            eventx, eventy = event.xdata, event.ydata
            diff = np.abs(self.slopes - eventx) + np.abs(self.cdiff - eventy)
            idx_fused = np.argmin(diff)

        need_highlighted = not self.highlighted[idx_fused]

        # Dehighlight other points and segments
        for idx_hl in [i for i in range(len(self.highlighted)) if self.highlighted[i]]:
            t0 = self.lopt_fused[idx_hl]
            t1 = self.lopt_fused[idx_hl+1]
            idx0 = get_idx_from_x(self.t, t0)
            idx1 = get_idx_from_x(self.t, t1)
            y0, y1 = self.y[idx0], self.y[idx1]

            slope, cdiff = self.slopes[idx_hl], self.cdiff[idx_hl]
            self.ax0.plot((t0, t1), (y0, y1), color='green')
            self.ax1.scatter(slope, cdiff, color='blue')
            self.highlighted[idx_hl] = False

        if need_highlighted:
            color = 'yellow'
            slope, cdiff = self.slopes[idx_fused], self.cdiff[idx_fused]
            t0 = self.lopt_fused[idx_fused]
            t1 = self.lopt_fused[idx_fused+1]
            idx0 = get_idx_from_x(self.t, t0)
            idx1 = get_idx_from_x(self.t, t1)
            y0, y1 = self.y[idx0], self.y[idx1]
    
            self.ax0.plot((t0, t1), (y0, y1), color=color)
            self.ax1.scatter(slope, cdiff, color=color)
            self.highlighted[idx_fused] = True

        plt.draw()
