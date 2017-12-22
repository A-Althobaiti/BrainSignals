import sys
from time import sleep

import numpy as np

def fancy_print(string):
	if string[-1] != '\n':
		string = string + '\n'
	for char in string:
		sys.stdout.write('{}'.format(char))
		sys.stdout.flush()
		sleep(0.025)

def printList(lst):
	for i, val in enumerate(lst):
		print('{}) {}'.format(i+1, val))

def array_reshape_by_column(array, index_of_column):
	new_array = []
	for col in range(array.shape[index_of_column]):
		new_array.append(array[:,col].flatten())
	return np.array(new_array)

def normalize_channels(array2d):
	return np.divide(np.subtract(array2d, array2d.mean(axis=0)), array2d.std(axis=0))

def verify_list_str_within_bounds(low, high, list_or_str):
	new_list = []

	if isinstance(list_or_str, list):
		pass
	elif isinstance(list_or_str, str):
		list_or_str = list_or_str.split()
	else:
		raise ValueError('Expected str or list of number(s), got {}:{}'.format(
			type(list_or_str), list_or_str))

	for element in list_or_str:
		if int(element) >= low and int(element) <= high:
			new_list.append(int(element))
		else:
			raise ValueError('Expected values between {} and {}, got {}'.format(
				low, high, element))

	return new_list

def ccf(x, y, maxlags=26):
	Nx = len(x)
	c = np.correlate(x, y, mode=2)
	c/= np.sqrt(np.dot(x,x) * np.dot(y,y))
	c = c[Nx-1-maxlags:Nx+maxlags]
	return c

'''
This is another version of ccf with fft, more percesion.

import numpy as np
from scipy.fftpack import fft, fftshift, ifft, fftn

def ccf_shot(x, y, Nx, maxlags=26):
	return np.real(fftshift(ifft(np.multiply(x, np.conj(y)))))[Nx-1-maxlags:Nx+maxlags]

def ccf_driver(frame):
	for ch_1 in range(len(frame[0])):
		for ch_2 in range(ch_1, len(frame[0])):
			ccf(frame[:,ch_1], frame[:,ch_2])

def ccf_fft_driver(arg):
	Nx = len(arg[:,0])
	corr_length = Nx * 2 - 1
	#frame_1 = fftn(frame, shape=(corr_length,), axes=[0])
	for ch_1 in range(len(frame[0])):
		x = fft(arg[:,ch_1], corr_length)
		for ch_2 in range(ch_1, len(frame[0])):
			ccf_shot(x, fft(arg[:,ch_2], corr_length), Nx)
'''