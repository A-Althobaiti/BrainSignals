from .channel import Channel
from .plotly_plot.plotly_plot import plotly_plot
from .utilities import verify_list_str_within_bounds

'''
A statistical Python toolbox to explore brain signals

TODO:
xcorr plot
	if single band use interactive mode
	if dual band use matplotlib
'''

class Epoch():
	'Segmentation of Brain Signals to help performing exploratory data analysis (EDA)'
	def __init__(self, brain_signals, epoch_id):
		self.brain_signals = brain_signals
		self.dataset = Epoch.get_epoch(brain_signals, epoch_id)
		self.name = brain_signals.name
		self.sampling_rate = brain_signals.sampling_rate
		self.num_channels = brain_signals.num_channels
		self.epoch_length = brain_signals.epoch_length
		self.epoch_id = epoch_id
		self.channels = [Channel(self, channel_id) for channel_id in range(1, self.num_channels + 1)]
		self.xcorr_dict = {}
		self.lags_dict = {}

	def __getitem__(self, index): 
		'gives access to channels where indexing starts from 1 to length of channels.'
		if index < 1 or index > len(self.channels):
			raise ValueError('Index should be between 1 and {}.'.format(len(self.channels)))
		return self.channels[index-1]

	def __len__(self):
		'returns the number of channels in the epoch'
		return len(self.channels)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return 'Epoch {}'.format(self.epoch_id)
	
	def __getattr__(self, key):
		if key in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
			row_lower_bound = (self.epoch_id - 1) * self.sampling_rate
			row_upper_bound = self.epoch_id * self.sampling_rate
			try:
				return self.brain_signals.__getattr__(key)[row_lower_bound:row_upper_bound,:]
			except TypeError:
				return None

	def plot(self, band='raw', channels=None, exclude=False):
		dataset = None

		if band not in ['raw', 'delta','theta','alpha','beta','gamma']:
			raise ValueError('{} is not an acceptable parameter.\n'\
							'Allowed parameters are: \'raw\', \'delta\','\
							'\'theta\',\'alpha\',\'beta\', or \'gamma\''.format(band))
		elif band == 'raw':
			dataset = self.dataset
		else:
			dataset = self.__getattr__(band)
		
		if dataset is None:
			print('The {} dataset is empty, cannot proceed!'.format(band))
			return

		channels = verify_list_str_within_bounds(1, self.num_channels, channels)\
					if channels is not None else None
		lines=[]
		names=[]
		if channels is not None:
			for ch in range(self.num_channels):
				if exclude and ch+1 not in channels:
					lines.append(dataset[:,ch])
					names.append('Ch {}'.format(ch+1))
				elif not exclude and ch+1 in channels:
					lines.append(dataset[:,ch])
					names.append('Ch {}'.format(ch+1))
		else:
			for ch in range(self.num_channels):
				lines.append(dataset[:,ch])
				names.append('Ch {}'.format(ch+1))

		plotly_plot.line_chart(names, '{}'.format(self.__str__()), *lines)
		
	@staticmethod
	def get_epoch(brain_signals, epoch_id):
		'''
		Given a brain-signal object, it returns the data frame for the specified
		epoch number
		'''
		return brain_signals.dataset[((epoch_id - 1) \
			   * brain_signals.sampling_rate): \
			   (epoch_id * brain_signals.sampling_rate),:]