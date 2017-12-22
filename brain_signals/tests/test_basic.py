import numpy as np

from brain_signals.brain_signals import BrainSignals

# Fake some data
frequencies = {'delta':[0.5*np.pi, 4*np.pi],
				'theta':[4*np.pi, 8*np.pi],
				'alpha':[8*np.pi, 12*np.pi],
				'beta':[12*np.pi, 30*np.pi],
				'gamma':[30*np.pi, 100*np.pi]}

number_of_epochs = 300
sampling_rate = 1000
channels = 32

data = []
for ch in range(channels):
	temp_list = []
	for f in frequencies:
		t = np.linspace(start=frequencies[f][0],
						stop=frequencies[f][1],
						num=sampling_rate * number_of_epochs)
		temp_list.append(np.sin(t))
	temp_list_sum = np.zeros((sampling_rate * number_of_epochs,))
	for band in temp_list:
		temp_list_sum = np.add(temp_list_sum, band)
	white_noise = np.random.uniform(-250, 250, sampling_rate * number_of_epochs)
	data.append(np.multiply(temp_list_sum, white_noise))

data = np.array(data).T

fake = BrainSignals(data, 'Fake Dataset')
