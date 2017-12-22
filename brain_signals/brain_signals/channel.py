import matplotlib.pyplot as plt
from matplotlib import style

from .plotly_plot.plotly_plot import plotly_plot
'''
A statistical Python toolbox to explore brain signals
'''

class Channel():
	'Channels of Brain Signals\' epochs to help performing exploratory data analysis (EDA)'
	def __init__(self, epoch, channel_id):
		self.epoch = epoch
		self.dataset = Channel.get_channel(epoch, channel_id)
		self.name = epoch.name
		self.sampling_rate = epoch.sampling_rate
		self.num_channels = epoch.num_channels
		self.epoch_length = epoch.epoch_length
		self.channel_id = channel_id

	def __repr__(self): # when the user calls the object information of the channel is printed
		return self.__str__()

	def __str__(self): # when the user calls the object information of the channel is printed
		return 'Channel {}'.format(self.channel_id)

	def plot(self, bands=None):
		band_set = ['delta','theta','alpha','beta','gamma']
		if bands == 'all':
			specified_bands = band_set
		else:
			specified_bands = [element for element in band_set if element in bands] if bands is not None else []
		
		lines = [self.dataset]
		names = ['Raw Data']

		for band in specified_bands:
			arr = self.__getattr__(band)
			if arr is not None:
				lines.append(arr)
				names.append(band)
		
		title = '{} - {}'.format(self.epoch.__str__(), self.__str__())
		plotly_plot.line_chart(names, title, *lines)


	def __getattr__(self, key):
		if key in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
			try:
				return self.epoch.__getattr__(key)[:,self.channel_id-1]
			except TypeError:
				return None
	
	@staticmethod
	def get_channel(epoch, channel_id):
		'Given an Epoch object, it returns the data frame for the specified channel number'
		return epoch.dataset[:,channel_id-1]