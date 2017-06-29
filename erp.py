# Copyright (C) 2017 Michael D. Nunez
#
# License: BSD (3-clause)

# # Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 03/30/16      Michael Nunez                            Original code
# 05/31/17      Michael Nunez                      Addition of epochsubset
# 06/29/17      Michael Nunez                      Addition of finddeflection

import numpy as np

# Subtract mean to baseline each epoch of EEG data


def baseline(data, wind=range(0, 50)):
    """Simple function to baseline EEG data for ERP calculations

    Inputs:
    data - sample*channel*trial or sample*trial*channel EEG data
    wind - subtracting the mean of this window to re-center EEG data

    Outputs:
    recentered - Re-centered EEG data
    """

    baselines = np.squeeze(np.mean(data[wind, :, :], axis=0))
    recentered = data - np.tile(baselines, (data.shape[0], 1, 1))

    return recentered


def epochsubset(data, newindex, lockindex=None):
    """Reepochs each epoch of EEG data by timelocking each epoch
    "i" to "newindex[i]" with maximum window size available

    Inputs:  
    data - sample*channel*trial EEG data
    newindex - Vector of length "trial"

    Optional inputs:
    lockindex - Sample in which newdata is timelocked
                Default: nanmin(newindex)

    Outputs:  
    newdata - Re-timelocked EEG data          
    lockindex - Sample in which newdata is timelocked
    badtrials - Index of trials where newindex contained nans
    """

    if lockindex is None:
        lockindex = int(np.nanmin(newindex))

    windsize = (np.shape(data)[0] - int(np.nanmax(newindex))) + int(lockindex)

    newdata = np.zeros(
        (int(windsize), int(np.shape(data)[1]), int(np.shape(data)[2])))

    for t in range(0, np.size(newindex)):
        if np.isfinite(newindex[t]):
            begin_index = int(newindex[t]) - lockindex
            end_index = windsize + begin_index
            newdata[:, :, t] = data[begin_index: end_index, :, t]

    badtrials = np.where(np.isnan(newindex))

    return newdata, lockindex, badtrials


def finddeflection(erp, cutoff, baseline, erpwindow):
    '''
    Converted from Ramesh Srinivasan's MATLAB function
    to find onset times of certain ERPs
    '''
    maxtime = np.shape(erp)[0] - np.size(baseline)
    onset = np.zeros((np.shape(erp)[1]))
    for j in np.arange(0, np.shape(erp)[1]):
        test = erp[:, j] < cutoff[j]
        testtime = np.where(test)[0]
        testval = np.zeros((np.size(test)))
        if np.size(testtime) > 0:
            for k in np.arange(0, np.size(testtime)):
                if testtime[k] < maxtime:
                    testval[testtime[k]] = np.sum(
                        test[testtime[k] + baseline])

            # Find where the deflection remains for the lenght of the baseline
            testbaseline = np.where(testval == np.size(baseline))[0]
            if np.size(testbaseline) > 0:
                onset[j] = erpwindow[testbaseline[0]]
            else:
                onset[j] = np.nan
        else:
            onset[j] = np.nan
    return onset
