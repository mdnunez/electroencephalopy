# Copyright (C) 2016 Michael D. Nunez
#
# License: BSD (3-clause)

# # Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 03/30/16      Michael Nunez                            Original code

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
