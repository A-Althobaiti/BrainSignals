import os
import errno
from os.path import isdir as dir_exist
from os.path import isfile as file_exist
from os import getcwd as pwd
from os import mkdir
from time import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample
from scipy.stats import zscore

from .utilities import fancy_print, ccf, normalize_channels
from .speed_up import filtfilt
from .epoch import Epoch
from .plotly_plot.plotly_plot import plotly_plot

'''
A statistical Python toolbox to explore brain signals

TODO:
Stream plot per band or data
interactive xcorr for the whole dataset
'''

class BrainSignals(object):
	'A toolbox to explore brain signals connectivity'

	version = '0.1'

	def __init__(self, dataset, name, sampling_rate=1000, num_channels=32, epoch_length=1):
		'''
		dataset: holds the actual brain signals, either as list or a numpy array, where columns are channels and rows are observations
		names: specifies the name of the dataset
		sampling_rate: is pre-defined as 1000; however, it must be overridden -specified- if the sampling rate is different
		num_channels: is pre-defined as 32; however, it must be overriden -specified- if the number of channels in the dataset is different
		epoch_length: is an arbitrary segmentatoin of the data, and the default value is 1 second
		'''
		self.dataset = dataset
		self.name = name
		self.sampling_rate = sampling_rate
		self.num_channels = num_channels
		self.epoch_length = epoch_length
		self.validate_input()
		self.epochs = [Epoch(self, epoch_id) for epoch_id in range(1, int(self.dataset.shape[0] / (self.sampling_rate * self.epoch_length)) + 1)]
		self.xcorr_dict = {}
		self.lags_dict = {}

	def __getitem__(self, index): 
		'gives access to epochs where indexing starts from 1 to length of epochs.'
		if index < 1 or index > len(self.epochs):
			raise ValueError('Index should be between 1 and {}.'.format(len(self.epochs)))
		return self.epochs[index-1]

	def __len__(self):
		'returns the number of segmentations, aka. epochs'
		return len(self.epochs)

	def __repr__(self):
		return 'BrainSignals(dataset={}, name={}, sampling_rate={}, num_channels={}, epoch_length={})'.format(self.dataset, self.name, self.sampling_rate, self.num_channels, self.epoch_length)

	def __str__(self):
		return 'Brain Signals object\n--------------------\n{}\nDataset dimensions: {}\nSampling rate: {} Hertz\nNumber of channels: {} channels\nEpoch length: {} seconds'.format(self.name, self.dataset.shape, self.sampling_rate, self.num_channels, self.epoch_length)

	def __getattr__(self, key):
		if key in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
			if key in self.__dict__:
				return self.__dict__[key]
			else:
				fancy_print('The {} band is not extracted.\nRun the extract method with \'{}\' as an argument.'.format(key, key))
				return None
		raise KeyError('{} is not a valid attribute'.format(key))

	def validate_input(self):
		'checks if the dimensions of the data comply with the number of channels and sampling rate'
		nrow, ncol = self.dataset.shape[0], self.dataset.shape[1]
		if ncol != self.num_channels:
			raise ValueError('Number of columns must match number of channels')
		if nrow % self.sampling_rate != 0:
			raise ValueError('Number of rows must be a multiple of sampling rate')

	def extract(self, band):
		'''
		Extracts a frequency band of the channels for each epoch.

		Parameters
		----------
		band : takes one or more of the following frequency bands
				+---------+--------------------+
				| 'delta' | 0.5 Hz to 4 Hz     |
				+---------+--------------------+
				| 'theta' | 4 Hz to 8 Hz       |
				+---------+--------------------+
				| 'alpha' | 8 Hz to 12 Hz      |
				+---------+--------------------+
				| 'beta'  | 12 Hz to 30 Hz     |
				+---------+--------------------+
				| 'gamma' | 30 Hz to 100 Hz    |
				+---------+--------------------+
				|  'all'  | extracts all bands |
				+---------+--------------------+
				the ranges of the bands are pre-defined and
				cannot be modeified. Unless the frequency range
				exceeds the sampling rate of the dataset, then
				the upper limit will be replaced by the sampling
				rate - 1.
		Returns
		----------
		None. The resulting computations will be stored in the object
		and accessible through calling '{object_name}.{band_name}'.
		'''
		bands = ['delta','theta','alpha','beta','gamma']
		frequency_range = {'delta':[0.5,4],'theta':[4,8],'alpha':[8,12],'beta':[12,30],'gamma':[30,100]}
		
		band_set = []

		start = time()

		if not band == 'all':
			band_set = [element for element in bands if element in band] if band is not None else []
		else:
			band_set = bands
			
		if band_set == []:
			raise ValueError('Acceptable band names are: \'delta\', \'theta\', \'alpha\', \'beta\', and \'gamma\'.')

		for band in band_set:
			low, high = frequency_range[band]
			if low >= self.sampling_rate:
				raise ValueError(errno.EINVAL, os.strerror(errno.EINVAL), '{} low frequency {} cannot be greater than the sampling rate {}.'.format(band, low, self.sampling_rate))
			if high >= self.sampling_rate:
				fancy_print('[Warning] {} high frequency {} is greater than the sampling rate {}, and it will be replaced with {}.'.format(band, low, self.sampling_rate, self.sampling_rate/2))
				high = self.sampling_rate/2
			self.__dict__[band] = np.zeros(self.dataset.shape, dtype=np.float32)
			# DO NOT MODIFY #
			# THIS IS MEANT FOR OPTIMIZATION, MUST NOT BE MODIFIED #
			b, a = butter(N=3, Wn=(2 * low / self.sampling_rate, 2 * high / self.sampling_rate), btype='bandpass', analog=False, output='ba')
			first_row = -a[1:] / (1.0 * a[0])
			n = a.size
			c = np.zeros((n - 1, n - 1), dtype=first_row.dtype)
			c[0] = first_row
			c[list(range(1, n - 1)), list(range(0, n - 2))] = 1
			IminusA = np.eye(max(len(a), len(b)) - 1) - c.T
			B = b[1:] - a[1:] * b[0]
			zi = np.linalg.solve(IminusA, B)
			#########################################################
			for epoch in self.epochs:
				for channel in epoch.channels:
					row_lower_bound = (epoch.epoch_id - 1) * self.sampling_rate
					row_upper_bound = epoch.epoch_id * self.sampling_rate
					column = channel.channel_id-1
					self.__dict__[band][row_lower_bound:row_upper_bound, column] = filtfilt(b, a, channel.dataset, zi)

		end = time()
		total_time = end - start
		minutes = int(total_time / 60)
		seconds = int(total_time % 60)
		ms = int(total_time * 1000 % 1000)
		fancy_print('Finished extracting {}.\nExtraction time: {} minute(s), {} second(s), and {} millisecond(s)'.format(band_set, minutes, seconds, ms))

	def xcorr(self, first_band, second_band=None, maxlags=26):
		'''
		Performs a cross-correlation between pairs of channels 
		over all epochs for a given oscillatory band(s).

		Parameters
		----------
		first_band: specify the name or names for the bands to be
					cross-correlated. It could be a string or a list.
					Allowed parameters are:
					'delta','theta','alpha','beta','gamma', or 'all'

		second_band: (optional) specify the name or names for the bands
					 to perform a dual-frequency cross-correlation. It
					 could be a string or a list.
					 Allowed parameters are:
					 'delta','theta','alpha','beta','gamma', or 'all'

		maxlags: specify the number of maximum lags to be considered.
				 It must be an integer.

		Returns
		-------
		None. The resulting computations will be stored in the object
		and accessible through calling:
		'{object_name}.display_xcorr({first_band_name}, {second_band_name})'
		where the second_band_name is optional. It is intended for dual-
		frequency cross-correlation.

		Notes
		-----
		if .xcorr is called with a first_band only, it will perform regular
		cross-correlation on that band. Likewise, if the second_band is the
		same as the first_band, then a regular cross-correlation on the band
		will be performed.

		Warning
		-------
		The percesion of this computation has been decreased for both memory
		and computation optimization. Therefore, a value of zero, 0, might
		be represented as 0.00000016e-16.
		'''
		start = time()
		bands = ['delta','theta','alpha','beta','gamma']
		lags = np.arange(-maxlags, maxlags+1)
		first_band_set, second_band_set = [], []

		if not first_band == 'all':
			first_band_set = [element for element in bands if element in first_band] if first_band is not None else []
		else:
			first_band_set = bands
		
		if not second_band == 'all':
			second_band_set = [element for element in bands if element in second_band] if second_band is not None else []
		else:
			second_band_set = bands
		
		if first_band_set == []:
			raise ValueError('Acceptable band names are: delta, theta, alpha, beta, and gamma.')

		dual_bands = []

		if second_band_set == []:
			dual_bands = ['{}-{}'.format(band, band) for band in first_band_set]
		else:
			dual_bands = ['{}-{}'.format(band_1, band_2) for band_1 in first_band_set for band_2 in second_band_set]

		for dual_band in dual_bands:
			band_1, band_2 = dual_band.split('-')
			xcorr_values = np.zeros((len(self), self.num_channels, self.num_channels), dtype=np.float32)
			lags_values = np.zeros((len(self), self.num_channels, self.num_channels), dtype=np.float32)
			for epoch_index in range(len(self)):
				for channel_1_index in range(self.num_channels):
					for channel_2_index in range(channel_1_index, self.num_channels):
						row_lower_bound = epoch_index * self.sampling_rate
						row_upper_bound = (epoch_index + 1) * self.sampling_rate
						first_channel = None
						second_channel = None
						try:
							first_channel = self.__dict__[band_1][row_lower_bound:row_upper_bound, channel_1_index]
							second_channel = self.__dict__[band_2][row_lower_bound:row_upper_bound, channel_2_index]
						except KeyError:
							fancy_print('The band ({} and {}) must be extracted first. Run the \'extract\' method.'.format(band_1, band_2))
							return None
						ac = ccf(first_channel, second_channel, maxlags)
						ac = ac ** 2
						max_index = np.argmax(ac)
						xcorr_values[epoch_index, channel_1_index, channel_2_index] = ac[max_index]
						xcorr_values[epoch_index, channel_2_index, channel_1_index] = ac[max_index]
						lags_values[epoch_index, channel_1_index, channel_2_index] = lags[max_index]
						lags_values[epoch_index, channel_2_index, channel_1_index] = -lags[max_index]
			self.xcorr_dict[dual_band] = xcorr_values
			self.lags_dict[dual_band] = lags_values
		
		end = time()
		total_time = end - start
		minutes = int(total_time / 60)
		seconds = int(total_time % 60)
		ms = int(total_time * 1000 % 1000)
		
		fancy_print('Finished cross-correlation of {}.\nCross-correlation time: {} minute(s), {} second(s), and {} millisecond(s)'.format(dual_bands, minutes, seconds, ms))

	def boxplot_channels(self):
		normalized_channels = normalize_channels(self.dataset)
		absolute_normalized_channels = np.abs(normalized_channels)
		#ignored 
		#absolute_normalized_channels = normalized_channels

		
		plotly_plot.boxplot(y=absolute_normalized_channels.mean(axis=0),
							name='Channels',
							title='{}'.format(self.name))

	def boxplot_epochs(self):
		resampled_dataset = resample(self.dataset, len(self.epochs), axis=0)
		z_scores = zscore(resampled_dataset, axis=0, ddof=0)

		plotly_plot.boxplot(y=z_scores.mean(axis=1),
							name='Epoch',
							title='{}'.format(self.name))

	def save(self, path='./Brain Signals/', overwrite=False):
		'''
		saves the current object to disk at Brain Signals directory, which is located at the
		current working directory if not specified; in other words, './Brain Signals/'.
		Each object is saved into its own directory with the name specified upon initialization
		of Brain Signals object.
		'''

		def save_metadata():
			start = time()
			list_of_metadata = ['name','sampling_rate','num_channels','epoch_length']
			with open('{}{}/metadata.dat'.format(path, self.name),'w') as metadata:
				for element in list_of_metadata:
					metadata.write('{}={}\n'.format(element, self.__dict__[element]))
			end = time()
			return end - start

		def save_dataset():
			start = time()
			np.savez_compressed('{}{}/dataset'.format(path, self.name), dataset=self.dataset)
			end = time()
			return end - start

		def save_bands():
			start = time()
			bands_dict = {}
			bands = ['delta','theta','alpha','beta','gamma']
			for band in bands:
				if band in self.__dict__:
					bands_dict[band] = self.__dict__[band]
			if len(bands_dict) > 0:
				np.savez_compressed('{}{}/bands'.format(path, self.name), **bands_dict)
			end = time()
			return end - start

		def save_xcorr():
			start = time()
			if len(self.xcorr_dict) > 0:
				np.savez_compressed('{}{}/xcorr'.format(path, self.name), **self.xcorr_dict)
			end = time()
			return end - start

		def save_lags():
			start = time()
			if len(self.xcorr_dict) > 0:
				np.savez_compressed('{}{}/lags'.format(path, self.name), **self.lags_dict)
			end = time()
			return end - start

		saving_time = 0

		if not dir_exist('{}{}'.format(path, self.name)):
			mkdir('{}{}'.format(path, self.name))
		else:
			if not overwrite:
				fancy_print('A BrainSignals object with the name: {} already exists,'.format(self.name))
				fancy_print('at {}{}'.format(path, self.name))
				ans = None
				while ans != 'Y' and ans != 'N':
					fancy_print('Would you like to overwrite it?')
					ans = input('[Y|N]: ')
				if ans == 'N':
					return

		saving_time += save_metadata()
		saving_time += save_dataset()
		saving_time += save_bands()
		saving_time += save_xcorr()
		saving_time += save_lags()
		minutes = int(saving_time / 60)
		seconds = int(saving_time % 60)
		ms = int(saving_time * 1000 % 1000)
		fancy_print('Finished saving BrainSignals object.\nSaving time: {} minute(s), {} second(s), and {} millisecond(s)'.format(minutes, seconds, ms))
