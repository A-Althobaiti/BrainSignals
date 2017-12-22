from tests import test_basic as tb
from brain_signals.loader import loader

fake = tb.fake
real = loader('/home/aman/datasets/F150410-lfp-5min-1kHz.mat',
				None,
				'pre_pmcao',
				32,
				1000,
				1)