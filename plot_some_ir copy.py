from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import ERBnum2Hz
from gtfblib.fir import FIR


cfs_erb = (np.arange(1, 33.1, .5))
fbFIR = FIR(fs=16000, L = 512, cfs=ERBnum2Hz(np.arange(1, 33.1, .5)), complexresponse=True)
print(f' nb of filters = {fbFIR.nfilt}, cfs size = {fbFIR.cfs.size}, cfs = {fbFIR.cfs}, cfs_erb=  {cfs_erb} size {cfs_erb.size}')
fbFIR.plot_ir(3, 3)

