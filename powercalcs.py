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

##Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

##Power calculation and plot
def eegpower(data,sr,freqs=[0., 50.],dB=False,plot=False,**kwargs):
	"""Calculates & plots power spectrum of sample*channel*trial EEG data
	
	Useage:  
	xfreqs, outpower, fourier = eegPower(data,sr,freqs=[],dB=False,plot=False,**kwargs)
	
	Inputs: 
	data - sample*channel*trial EEG data
	sr - sample rate
	
	Optional Inputs: 
	freqs - lower and upper boundary of frequencies to plot, default: [0., 50.] (Hz)
	dB - Units in standardized dB instead of standardized power
	plot - Produces plot
	varargin - Any "plt.plot" inputs after the first two, i.e., plt.plot(X,Y,**kwargs);
	
	Outputs:  
	xfreqs - Frequencies plotted (x-axis)
	outpower - Power values plotted (y-axis)
	fourier - Fouier coefficients
	"""
	
	##Find frequency interval	
	
	nyquist = (2./5.)*sr
	if freqs[1] > nyquist:
	    print 'User defined maximum frequency %0.3f is larger than the Nyquist frequency %0.3f! \n' % (freqs[1],nyquist)
	    print 'Using Nyquist frequnecy as maximum \n'
	    freqs[1] = nyquist
	
	#Recalculate if minimum is smaller than Nyquist sampling rate
	nsr = sr/np.shape(data)[0] #Nyquist sampling rate (Hz)
	if (freqs[0] < nsr) & (freqs[0] != 0):
		print 'User defined minimum frequency %0.3f is smaller than Nyquist sampling rate %0.3f! \n' % (freqs[0],nsr)
		print 'Using Nyquist sampling rate as minimum frequency \n'
		freqs[0] = nsr;
	
	##Calculate Power
	
	fourier = np.fft.fft(data,axis=0)/np.shape(data)[0]
	plotfreqs = np.arange(0.,freqs[1],nsr)
	minindex = np.argmin(abs(freqs[0] - plotfreqs))
	maxindex = np.shape(plotfreqs)[0]

	#Power in standardized units (\muV^2/Hz)
	power = np.mean(np.square(np.abs(fourier)),axis=2)*(2./nsr)
	if dB:
		power = 10*np.log10(power)

	xfreqs = plotfreqs[minindex:maxindex]
	outpower = power[minindex:maxindex,:]
	fourier = fourier[minindex:maxindex,:,:]

	if plot:
	    	plt.plot(xfreqs,outpower,**kwargs);
	    	plt.xlabel('Frequency (Hz)');
	    	if dB:
			plt.ylabel('Standardized Log Power (10*log_{10}(muV^2/Hz); dB)');
	    	else:
			plt.ylabel('Standardized Power (muV^2/Hz)');
	    	plt.title('EEG Power Spectrum');

	return xfreqs, outpower, fourier


