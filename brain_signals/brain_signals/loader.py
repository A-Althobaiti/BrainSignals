import errno
import os
from os.path import isdir as dir_exist
from os.path import isfile as file_exist
from time import time

import numpy as np
import h5py
from scipy.io import loadmat

from .brain_signals import BrainSignals
from .utilities import fancy_print, printList, array_reshape_by_column

class brain_signals_loader(object):
	def __init__(self, path, mapping=None):
		self.path = path
		self.mapping = 'r+' if mapping else None
		self.name = None
		self.dataset = None
		self.sampling_rate = None
		self.num_channels = None
		self.epoch_length = None
		self.loading_time = 0
		self.load_metadata()
		self.load_dataset()
		self.brain_signals_object = BrainSignals(self.dataset, self.name, self.sampling_rate, self.num_channels, self.epoch_length)
		self.load_bands()
		self.load_xcorr()
		self.load_lags()
		self.minutes = int(self.loading_time / 60)
		self.seconds = int(self.loading_time % 60)
		self.ms = int(self.loading_time * 1000 % 1000)
		fancy_print('Finished loading a BrainSignals object.\nLoading time: {} minute(s), {} second(s), and {} millisecond(s)'.format(self.minutes, self.seconds, self.ms))

	def load_metadata(self):
		start = time()
		if not file_exist('{}/metadata.dat'.format(self.path)):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), '{}/metadata.dat'.format(self.path))
		with open('{}/metadata.dat'.format(self.path),'r') as metadata:
			for line in metadata:
				parameter, value = line.split('=')
				if parameter == 'name':
					self.__dict__[parameter] = value[:-1]
				else:
					self.__dict__[parameter] = int(value)
		end = time()
		self.loading_time += end - start

	def load_dataset(self):
		start = time()
		if not file_exist('{}/dataset.npz'.format(self.path)):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), '{}/dataset.npz'.format(self.path))
		self.dataset = np.load('{}/dataset.npz'.format(self.path), mmap_mode=self.mapping)
		self.dataset = self.dataset['dataset']
		end = time()
		self.loading_time += end - start

	def load_bands(self):
		start = time()
		if file_exist('{}/bands.npz'.format(self.path)):
			bands_dict = np.load('{}/bands.npz'.format(self.path), mmap_mode=self.mapping)
			available_keys = [k for k in bands_dict.keys() if k in ['delta','theta','alpha','beta','gamma']]
			for key in available_keys:
				self.brain_signals_object.__dict__[key] = bands_dict[key]
		end = time()
		self.loading_time += end - start

	def load_xcorr(self):
		start = time()
		if file_exist('{}/xcorr.npz'.format(self.path)):
			xcorr_dict = np.load('{}/xcorr.npz'.format(self.path), mmap_mode=self.mapping)
			for key in [k for k in xcorr_dict.keys()]:
				self.brain_signals_object.xcorr_dict[key] = xcorr_dict[key]
		end = time()
		self.loading_time += end - start

	def load_lags(self):
		start = time()
		if file_exist('{}/lags.npz'.format(self.path)):
			lags_dict = np.load('{}/lags.npz'.format(self.path), mmap_mode=self.mapping)
			for key in [k for k in lags_dict.keys()]:
				self.brain_signals_object.lags_dict[key] = lags_dict[key]
		end = time()
		self.loading_time += end - start

	def create_BrainSignals(self):
		return self.brain_signals_object

class matlab_loader(object):
	def __init__(self, path, dataset_name=None, variable_name=None, num_channels=None, sampling_rate=None, epoch_length=1):
		self.path = path
		self.name = dataset_name
		self.variable_name = variable_name
		self.channels = num_channels
		self.matlab_object = None
		self.is_h5py = False
		self.dataset = None
		self.sampling_rate = sampling_rate
		self.epoch_length = epoch_length
		self.loading_time = 0
		self.check_path()
		self.load_matlab_object()
		self.get_dataset()
		self.check_dimensions()
		self.check_sampling_rate()
		self.generate_dataset_name()
		self.minutes = int(self.loading_time / 60)
		self.seconds = int(self.loading_time % 60)
		self.ms = int(self.loading_time * 1000 % 1000)
		fancy_print('Finished loading a matlab file.\nLoading time: {} minute(s), {} second(s), and {} millisecond(s)'.format(self.minutes, self.seconds, self.ms))

	def check_path(self):
		if not file_exist(self.path):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

	def create_BrainSignals(self):
		return BrainSignals(self.dataset, self.name, self.sampling_rate, self.channels, self.epoch_length)

	def load_matlab_object(self):
		start = time()
		try:
			self.matlab_object = loadmat(file_name=self.path, verify_compressed_data_integrity=True)
		except NotImplementedError:
			self.is_h5py = True
			self.matlab_object = h5py.File(self.path)
		except Exception as e:
			print(e)
			raise e
		end = time()
		self.loading_time += end - start

	def get_dataset(self):
		start = time()
		variables_list = [key for key in self.matlab_object.keys() if key not in ['__globals__','__header__','__version__']]
		if self.variable_name in variables_list:
			self.dataset = np.array(self.matlab_object[self.variable_name], dtype=np.float32)
			end = time()
			self.loading_time += end - start
		else:
			if not self.variable_name is None:
				fancy_print('The variable name \"{}\" does not exist in the matlab file.'.format(self.variable_name))
			if len(variables_list) < 1:
				raise KeyError(errno.ENOKEY, os.strerror(errno.ENOKEY), 'there are no variables in the matlab file!')
			ans = 0
			while ans < 1 or ans > len(variables_list):
				fancy_print('The variables available in this dataset are:')
				printList(variables_list)
				fancy_print('Please enter the number that corresponds to the variable name:')
				ans = int(input())
			self.variable_name = variables_list[ans-1]
			start = time()
			self.dataset = np.array(self.matlab_object[self.variable_name], dtype=np.float32)
			end = time()
			self.loading_time += end - start
		
	def check_dimensions(self):	
		if not self.channels in self.dataset.shape:
			if not self.channels is None:
				fancy_print('The number of channels, {}, provided is not one of the dataset\'s dimensions'.format(self.channels))
			ans = 0
			while ans < 1 or ans > len(self.dataset.shape):
				fancy_print('The dimensions of this dataset are:')
				printList(self.dataset.shape)
				fancy_print('Please enter the number that corresponds to the number of channels:')
				ans = int(input())
			self.channels = self.dataset.shape[ans - 1]
		channels_index = self.dataset.shape.index(self.channels)
		if channels_index == 1 and len(self.dataset.shape) == 2:
			pass
		else:
			start = time()
			reshaped_dataset = array_reshape_by_column(self.dataset, channels_index)
			if self.dataset.shape[channels_index] != reshaped_dataset.shape[1]:
				self.dataset = array_reshape_by_column(reshaped_dataset, 1)
			else:
				self.dataset = reshaped_dataset
			end = time()
			self.loading_time += end - start

	def check_sampling_rate(self):
		if (self.sampling_rate is None) or (self.dataset.shape[0] % self.sampling_rate != 0):
			self.sampling_rate = float('nan')
			while self.dataset.shape[0] % self.sampling_rate != 0:
				fancy_print('[Warning] the number of observations in the dataset is not a multiple of the sampling rate.')
				fancy_print('Please specify the sampling rate: ')
				self.sampling_rate = int(input())

	def generate_dataset_name(self):
		if self.name is None:
			self.name = self.path.split('/')[-1].split('.')[0]
			self.name = '{}_{}_{}spe_{}ch_{}Hz'.format(self.name, self.variable_name, self.epoch_length, self.channels, self.sampling_rate)

def loader(path, dataset_name=None, variable_name=None, num_channels=None, sampling_rate=None, epoch_length=1, mapping=True):
	if '/' in path or ':' in path:
		if dir_exist('{}'.format(path)):
			return brain_signals_loader(path, mapping).create_BrainSignals()
		elif file_exist('{}'.format(path)):
			return matlab_loader(path, dataset_name, variable_name, num_channels, sampling_rate, epoch_length).create_BrainSignals()
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), '{}'.format(path))
	else:
		if dir_exist('./Brain Signals/{}'.format(path)):
			return brain_signals_loader('./Brain Signals/{}'.format(path), mapping).create_BrainSignals()
		elif file_exist('./{}'.format(path)):
			return matlab_loader('./{}'.format(path), dataset_name, variable_name, num_channels, sampling_rate, epoch_length).create_BrainSignals()
		elif file_exist('./{}.mat'.format(path)):
			return matlab_loader('./{}.mat'.format(path), dataset_name, variable_name, num_channels, sampling_rate, epoch_length).create_BrainSignals()
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), '{}'.format(path))

#	v73 = loader('F160406-lfp-1kHz-0001-6000.mat', None, 'lfp_data', 32, 1000, 1)
#   loader(, None, 'lfp_data', 32, 1000, 1)
#	v73.extract('all')
#	v73.compute_xcorr('all', 'all')
#	v73.save()
#	v1 = loader('F150410-lfp-5min-1kHz.mat', None, 'pre_pmcao', 32, 1000, 1)
#	v2 = loader('F160406-lfp-1kHz-0001-6000_lfp_data_1spe_32ch_1000Hz', mapping=True)