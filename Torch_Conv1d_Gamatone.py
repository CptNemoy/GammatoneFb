import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from create_kernels_gamatone import create_mel_filters
from create_kernels_gamatone import create_fft_kernels
from create_kernels_gamatone import create_gamatone_filterbank
from gtfblib.gtfb import ERBnum2Hz
from gtfblib.fir import FIR
#from .util import window_sumsquare
import matplotlib.pyplot as plt
class ConvGamma(nn.Module):

    def __init__(self, N=32, sr=16000, nb_gamma_filters=32, gamma_filt_len=1024, initialize_ana_kernels=False, initialize_syn_kernels=False):
        super(ConvGamma, self).__init__()
        self.N = N
        self.gamma_filter_len = gamma_filt_len
        self.pad_amount = int(self.gamma_filter_len -1) // 2
        self.initialize_ana_kernels = initialize_ana_kernels
        self.initialize_syn_kernels = initialize_syn_kernels
        self.sampling_rate = sr
        
        self.nb_gamma_filters = nb_gamma_filters
        #
        
        # 1D input
        self.nb_freq_pts = int((self.N/2)+1)
        #
        # Filterbanks to generate gamatone (real part, imag part)
        #
        self.gamma_filt_r =nn.Conv1d(in_channels=1, out_channels=self.nb_gamma_filters, kernel_size=self.gamma_filter_len, stride=1, padding = (self.pad_amount) , padding_mode = 'zeros', bias=True)
        self.gamma_filt_i =nn.Conv1d(in_channels=1, out_channels=self.nb_gamma_filters, kernel_size=self.gamma_filter_len, stride=1, padding = (self.pad_amount) , padding_mode = 'zeros', bias=True)

        # Bank of Convolution Filters to do inverse Gamma
        self.inv_gamma_filt_r = nn.Conv1d(in_channels=self.nb_gamma_filters, out_channels=1, kernel_size=self.gamma_filter_len, stride=1, padding = (self.pad_amount ), padding_mode = 'zeros', bias=True)
        self.inv_gamma_filt_i = nn.Conv1d(in_channels=self.nb_gamma_filters, out_channels=1, kernel_size=self.gamma_filter_len, stride=1, padding = (self.pad_amount ), padding_mode = 'zeros', bias=True)

        #
        # Weight Initialization
        # =====================

        # Get the Gamatone filters
        fbFIR = FIR(fs=self.sampling_rate, L = self.gamma_filter_len, cfs=ERBnum2Hz(np.arange(1, 33.1, .5)), complexresponse=True, groupdelay=self.gamma_filter_len//2)
        print(f' nb of filters = {fbFIR.nfilt}, cfs size = {fbFIR.cfs.size}, cfs = {fbFIR.cfs}')
        gamma_filts = fbFIR.ir / float(2.0 * self.gamma_filter_len)
        
        # Get the reverse Gamatone filters
        fbinvFIR = FIR(fs=self.sampling_rate, L = self.gamma_filter_len, cfs=ERBnum2Hz(np.arange(1, 33.1, .5)), complexresponse=True, reversetime=True , groupdelay=0)
        inv_gamma_filts = fbinvFIR.ir

                
        # Load the values into the kernels (with the correct dimensions)

        # initialize the kernels of the Gamatone FilterBank
        if (self.initialize_ana_kernels == True):
            print(f'initializing Gammatone Filter Bank')
            init_gamma_filts_r = torch.FloatTensor(gamma_filts.real[:, None, :])
            init_gamma_filts_i = torch.FloatTensor(gamma_filts.imag[:, None, :])
            with torch.no_grad():
                self.gamma_filt_r.weight = torch.nn.Parameter(init_gamma_filts_r)
                self.gamma_filt_i.weight = torch.nn.Parameter(init_gamma_filts_i)
                self.gamma_filt_r.bias = torch.nn.Parameter(torch.zeros(self.nb_gamma_filters))
                self.gamma_filt_i.bias = torch.nn.Parameter(torch.zeros(self.nb_gamma_filters))

        # initialize the kernels of the Gamatone FilterBank
        if (self.initialize_syn_kernels == True):
            print(f'initializing Gammatone Filter Bank')
            init_inv_gamma_filts_r = torch.FloatTensor(inv_gamma_filts.real.copy()[None, :, :])
            init_inv_gamma_filts_i = torch.FloatTensor(inv_gamma_filts.imag.copy()[None, :, :])
            with torch.no_grad():
                self.inv_gamma_filt_r.weight = torch.nn.Parameter(init_inv_gamma_filts_r)
                self.inv_gamma_filt_i.weight = torch.nn.Parameter(init_inv_gamma_filts_i)
                self.inv_gamma_filt_r.bias = torch.nn.Parameter(torch.zeros(1))
                self.inv_gamma_filt_i.bias = torch.nn.Parameter(torch.zeros(1))

        if (self.initialize_syn_kernels == False):
            print(f'initializing zeros Filter Bank')
            z_data = np.zeros([self.nb_gamma_filters, self.gamma_filter_len], dtype=float)
            init_inv_gamma_filts_r = torch.FloatTensor(z_data.copy()[None, :, :])
            init_inv_gamma_filts_i = torch.FloatTensor(z_data.copy()[None, :, :])
            with torch.no_grad():
                self.inv_gamma_filt_r.weight = torch.nn.Parameter(init_inv_gamma_filts_r)
                self.inv_gamma_filt_i.weight = torch.nn.Parameter(init_inv_gamma_filts_i)
                self.inv_gamma_filt_r.bias = torch.nn.Parameter(torch.zeros(1))
                self.inv_gamma_filt_i.bias = torch.nn.Parameter(torch.zeros(1))
            print(f' zero init')



    def forward(self, x):
        #print(f' size of x at input = {x.shape}')

        num_batches = x.shape[0]
        num_samples = x.shape[1]
                
        # put input in format [1,1,L]    where L is the incoming samples
        # The shape of the internal (analysis) kernels is [F, 1, N]     where F is the nb of filters, and N is the filter length
        #
        x = x.view(num_batches, 1, num_samples)
        #print(f' size of x at input = {x.shape}')
        
        # run samples thru analysis banks
        y_real = self.gamma_filt_r(x)           # [1, F, b*L]
        y_imag = self.gamma_filt_i(x)
        #print(f'size of y real, imag  ={y_real.shape}, {y_imag.shape}')


        # run the y_real and y_imag thru the synthesis kernels
        y_inv_r = self.inv_gamma_filt_r(y_real)               # output [1,1,b*L]
        y_inv_i = self.inv_gamma_filt_i(y_imag)               # output [1,1,b*L]
        y_inv = y_inv_r + y_inv_i
        y_inv = y_inv.squeeze()                               # output [1,1,b*L]
        

        y_real = y_real.squeeze(0)                  # y is in shape [F, b*512] where b is the nb of output frames. if L = N, then b = 1
        y_imag = y_imag.squeeze(0)                  # y is in shape [F, b*512] where b is the nb of outputs. if L = N, then b = 1

        
        # combine real and imaginary outputs vertically
        y = torch.cat([y_real, y_imag], dim=0)                   #[2*F, b*L]
        #print(f'size of y ={y.shape}')
        
        # Get the power of the r and imag
        power_frame =  torch.pow((torch.pow(y_real, 2.0) + torch.pow(y_imag, 2.0)), (0.5*0.3) )    # Power 'Spectrum'
        
        return y_real, y_imag, y, power_frame, y_inv                                                       # return filter output , and power output
        
