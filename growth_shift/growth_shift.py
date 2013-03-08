#!/usr/bin/env python

import numpy as np
from scipy.signal import medfilt, gaussian
from scipy.ndimage import convolve1d


def load_file(path, header_rows=6, delim="\t"):

    data = np.genfromtxt(path, skip_header=header_rows,
                         delimiter=delim)[1:, 1:]

    return data


def numeric_derivative(Y, kernel=np.array([1, 0, -1]),
                       median_window=11, gauss_window=5):

    dY = convolve1d(Y, kernel, axis=0, mode='nearest')
    dYm = medfilt(dY, median_window)
    """
    out = np.zeros(dY.shape)
    for i in range(dY.size - mean_window):
        out[i] = dY[i: i + mean_window].mean()
    """

    out = convolve1d(dYm, gaussian(gauss_window * 2, std=gauss_window / 2.0),
                     axis=0, mode='nearest')
    return out


def get_d1(Y):

    return numeric_derivative(Y, kernel=np.array([1, 0, -1]),
                              median_window=11, gauss_window=5)


def get_d2(Y):

    return numeric_derivative(Y, kernel=np.array([-1, 2, -1]),
                              median_window=11, gauss_window=5)


def mode_shifts(dY, kernel_size=6):

    kernel = np.array([1] * kernel_size)
    kernel[kernel_size / 2:] = -1

    ddY = convolve1d(dY, kernel, axis=0, mode='nearest')
    ddYmodes = 2 * ((ddY > 0) - 0.5)
    ddYmodeShifts = convolve1d(ddYmodes, np.array([1, -1]), axis=0, mode='nearest')
    ddYshiftPos = np.where(ddYmodeShifts)
    return ddYshiftPos, ddYmodes[ddYshiftPos] > 0
