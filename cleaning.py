# Copyright 2016 Michael D. Nunez

# This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

# # Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 03/26/16      Michael Nunez                            Original code
# 03/29/16      Michael Nunez                    Reorder inputs, add stopband

import numpy as np
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

# Butterworth and filtfilt wrapper


def butterfilt(data, sr, passband=(1.0, 50.0), stopband=(59., 61.),
               order=(5, 5), plotfreqz=False, plotlim=100):
    """Wrapper for Butterworth filter of sample*channel*trial EEG data

    Inputs:
    data - sample*channel*trial EEG data
    sr - sample rate (samps/sec)

    Optional Inputs:
    passband - low and high cutoffs for Butterworth passband
    stopband - low and high cuttoffs for Butterworth stopband
             Note that this should be used as a Notch filter
             For countries w/ 60 Hz power lines (Americas etc.) use (59., 61.)
             For countries w/ 50 Hz power lines (everywhere else) use (49., 51.)
    order - Order of both Butterworth filters: (passband, stopband)
    plotfreqz - Flag for plotting frequency responses of both filters
    plotlim - Upper limit of frequencies to plot

    Outputs:
    filtdata - Filtered sample*channel*trial EEG data
    """

    nyquist = .5 * float(sr)
    b, a = butter(order[0], [float(passband[0]) / nyquist,
                             float(passband[1]) / nyquist], btype='bandpass')
    filtdata = filtfilt(b, a, data, axis=0)

    if plotfreqz:
        w, h = freqz(b, a)
        plt.plot((nyquist / np.pi) * w, abs(h))
        plt.setp(plt.gca(),XLim=[0,plotlim],YLim=[0,1.1])
        plt.plot([0, nyquist], [np.sqrt(0.5), np.sqrt(0.5)], '--')
        plt.title(
            'Butterworth Passband Frequency Response, Order = %d' % order[1])

    if not not stopband:
        B, A = butter(order[1], [float(stopband[0]) / nyquist,
                      float(stopband[1]) / nyquist],
                      btype='bandstop')
        filtdata = filtfilt(B, A, data, axis=0)
        if plotfreqz:
            W, H = freqz(B, A)
            plt.figure()
            plt.plot((nyquist / np.pi) * W, abs(H))
            plt.setp(plt.gca(),XLim=[0,plotlim],YLim=[0,1.1])
            plt.plot([0, nyquist], [np.sqrt(0.5), np.sqrt(0.5)], '--')
            plt.title(
                'Butterworth Stopband Frequency Response, Order = %d'
                % order[1])

    return filtdata
