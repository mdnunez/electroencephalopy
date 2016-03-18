## Copyright 2016 Michael D. Nunez

#This program is free software: you can redistribute it and/or modify
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

## Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 03/16/16      Michael Nunez                            Original code

from scipy.signal import butter, filtfilt

##Butterworth and filtfilt wrapper 
def butterfilt(data,sr,lowcut=1.0,highcut=50.0,order=1):
	"""Wrapper for Butterworth filter of sample*channel*trial EEG data

	Useage:  
	filtdata = butterfilt(data,sr,lowcut=1,highcut=50,order=5)

	Inputs: 
	data - sample*channel*trial EEG data
	sr - sample rate (samps/sec)

	Optional Inputs: 
	lowcut - lower bandpass cutoff, default: 1. (Hz)
	highcut - high bandpass cutoff, default: 50. (Hz)
	order - Order of Butterworth filter

	Outputs:  
	filtdata - Filtered sample*channel*trial EEG data
	"""

	nyqist = .5*float(sr)
	b,a = butter(order,[float(lowcut)/nyqist, float(highcut)/nyqist],btype='bandpass')
	filtdata = filtfilt(b,a,data,axis=0)

	return filtdata
