#!/usr/bin/env python

import numpy as np
from scipy.signal import medfilt, gaussian
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt


def load_file(path, header_rows=7, delim="\t", left_rows=1, has_time=True):

    data = np.genfromtxt(path, skip_header=header_rows,
                         delimiter=delim)

    if has_time:
        T = data[1:, 0]
    else:
        T = np.arange(data.shape[0] - 1)

    return data[1:, left_rows:], T


def numeric_derivative(Y, kernel,
                       median_window=7, gauss_window=5):

    dY = convolve1d(Y, kernel, axis=0, mode='nearest')
    dYm = medfilt(dY, median_window)
    g = gaussian(gauss_window * 2, std=gauss_window / 2.0)
    g /= g.sum()
    out = convolve1d(dYm, g, axis=0, mode='nearest')
    return out


def get_d1(Y):

    return numeric_derivative(Y, kernel=np.array([1, -1]))


def get_d2(Y):

    return numeric_derivative(Y, kernel=np.array([1, -2, 1]))


def mode_shifts(Y, kernel_size=6):

    kernel = np.array([1] * kernel_size)
    kernel[kernel_size / 2:] = -1

    ddY = get_d2(Y)

    #ddY = convolve1d(dY, kernel, axis=0, mode='nearest')
    ddYmodes = 2 * ((ddY > 0) - 0.5)
    ddYmodeShifts = np.abs(convolve1d(ddYmodes, np.array([1, -1]), axis=0, mode='nearest')) == 2
    ddYshiftPos = np.where(ddYmodeShifts)
    return ddYshiftPos[0], ddY[ddYshiftPos]


def get_shift_phenotype(Y, T, threshold=0.95, threshold2=0.05, visual=False):

    shiftPos, shiftDirection = mode_shifts(Y)
    sUp = shiftPos[shiftDirection > 0]
    sDown = shiftPos[(shiftDirection > 0) == False]
    dY = get_d1(Y)

    val = 0

    for t1 in sUp:

        if dY[t1] > threshold2 and (sDown > t1).any():

            t2 = sDown[sDown > t1][0]

            if (sUp > t2).any():

                t3 = sUp[sUp > t2][0]

                if (max((Y[t1], Y[t3])) * threshold > Y[t2] and
                        dY[t3] > threshold2):
                    val = T[t3] - T[t1]
                    break

    if visual:

        plt.plot(T, Y, 'b-')
        plt.plot(T[sUp], Y[sUp], 'g*')
        plt.plot(T[sDown], Y[sDown], 'ro')
        plt.text(T.max() * 0.05, Y.max() * 0.9, "Shift size: {0:.2f}".format(val))

    return val


def write_all_to_file(data, T, out_path, curve_start_index=1, **kwargs):

    fh = open(out_path, 'w')

    fh.write("{0}\t{1}\n".format('Curve', 'Shift'))

    for i in range(data.shape[1]):

        s = get_shift_phenotype(data[:, i], T, **kwargs)

        fh.write("{0}\t{1}\n".format(i + curve_start_index, s))

    fh.close()


"""
def get_shift_phenotype2(shiftPos, Y, shiftDirection):

    dCmp = np.greater
    sFind = np.argmax

    old_i = 0
    segments = []
    curSegStart = 1
    spikes = []
    dY = get_d2(Y)

    for i, sD in enumerate(shiftDirection):

        if dCmp(sD, 0):

            segments.append(shiftPos[i])
            spikes.append(sFind(dY[old_i: i]) + curSegStart)
            curSegStart = segments[-1]

            if dCmp == np.greater:
                dCmp = np.less
                sFind = np.argmin
            else:
                dCmp = np.greater
                sFind = np.argmax

            old_i = i

    return segments, spikes
"""
