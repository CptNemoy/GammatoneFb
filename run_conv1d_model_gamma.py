import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
from Torch_Conv1d_Gamatone import ConvGamma
from torchsummary import summary
import librosa
from eval_model_gamma import plot_initial_kernels
from eval_model_gamma import plot_final_kernels
from eval_model_gamma import evaluate_cochleogram

from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb
from pycochleagram import utils

from create_kernels_gamatone import create_gamatone_filterbank

# User-defined parameters
# =======================
device = 'cpu'

N = 512                         # Frame length
NFFT = N
hop_length = NFFT//2
sample_rate = 16000
nb_batches = 200                # nb of epochs
batch_size = 100                # nb of samples used in each epoch
n_gamma = 65  # 29              # nb of gama filters ( used in the model here
n_gamma_demo = 30 # 12          # nb of gama filter (used in the PyCochleogram code (target), and it becomes 2*n + 5 )
gamma_filt_len = (N)+1          # L in the paper

low_lim=26                      # Low frequency limit (Hz)
high_lim=7743

# the following flags are used to determine which mode
# to start from random and train analysis and synthesis , set both flags to False
# To start from initialized analysis bank and train sysnthesis bank, set the first to True and the second to False
initialize_ana_kernels=False
initialize_syn_kernels=False
# ===========================


# create a dataset of several batches of complex vectors of size N
dataset = ComplexNumbersDataset(nb_batches, batch_size, N, content = 'Rand', file = 'ViolinConcertos_1_clean_int.wav', sr = sample_rate, complex=False)
dataset_validation = ComplexNumbersDataset(nb_batches, batch_size, N, content = 'Rand', file = 'ViolinConcertos_4_clean_int.wav', sr = sample_rate, complex=False)

# initialize model
c_gama = ConvGamma(N=N, sr=sample_rate, nb_gamma_filters=n_gamma, gamma_filt_len=gamma_filt_len, initialize_ana_kernels=initialize_ana_kernels, initialize_syn_kernels=initialize_syn_kernels)
print(c_gama)


# Plot the initial kernels
plot_initial_kernels(model=c_gama, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, filename_kernels='Initial_Gamma_Kernels', filename_gamma='Initial_Gama')

# Run an eval prior to training
evaluate_cochleogram(model=c_gama, n_gamma_model=n_gamma, n_gama_coh = n_gamma_demo, low_lim=low_lim, hi_lim=high_lim, sample_rate=sample_rate, eval_filename='initial_model_gamma', librosa_filename='initial_librosa_gamma.png', fr_len=N, synthesis_file='initial_resyn_file.wav')



#optimizer = optim.SGD(c_gama.parameters(), lr=0.1, momentum=0.8)  # for case of init cos/sin kernels    works ok
#optimizer = optim.SGD(c_gama.parameters(), lr=0.01, momentum=0.8)  # no init Epoch: 249 Training Loss:  0.00012715664193819977   Epoch: 449 Training Loss:  2.5586857000234885e-05

loss_function = nn.SmoothL1Loss()
optimizer = optim.SGD(c_gama.parameters(), lr=0.1, momentum=0.75)  # unitialized


# This method is called from the main loop (below)
def train_banks(model, epoch_ind, Sig, Label, batch_size, input_validation, Label_validation, N, loss_fct, optim, analysis_bank_train, synthesis_bank_train):

    if( (analysis_bank_train == False) & (synthesis_bank_train==False)):
        print(f'Nothing to train')
        model.gamma_filt_r.weight.requires_grad = False
        model.gamma_filt_r.bias.requires_grad = False
        model.gamma_filt_i.weight.requires_grad = False
        model.gamma_filt_i.bias.requires_grad = False
        model.inv_gamma_filt_r.weight.requires_grad = False
        model.inv_gamma_filt_r.bias.requires_grad = False
        model.inv_gamma_filt_i.weight.requires_grad = False
        model.inv_gamma_filt_i.bias.requires_grad = False
        #return 0, 0
    if( (analysis_bank_train == True) & (synthesis_bank_train==False)):
        print(f' Training the analysis banks')
        model.gamma_filt_r.weight.requires_grad = True
        model.gamma_filt_r.bias.requires_grad = True
        model.gamma_filt_i.weight.requires_grad = True
        model.gamma_filt_i.bias.requires_grad = True
        model.inv_gamma_filt_r.weight.requires_grad = False
        model.inv_gamma_filt_r.bias.requires_grad = False
        model.inv_gamma_filt_i.weight.requires_grad = False
        model.inv_gamma_filt_i.bias.requires_grad = False

    if( (synthesis_bank_train==True) & (analysis_bank_train == False)):
        print(f' Training the synthesis banks')
        model.gamma_filt_r.weight.requires_grad = False
        model.gamma_filt_r.bias.requires_grad = False
        model.gamma_filt_i.weight.requires_grad = False
        model.gamma_filt_i.bias.requires_grad = False
        model.inv_gamma_filt_r.weight.requires_grad = True
        model.inv_gamma_filt_r.bias.requires_grad = True
        model.inv_gamma_filt_i.weight.requires_grad = True
        model.inv_gamma_filt_i.bias.requires_grad = True
        
    if( (analysis_bank_train == True) & (synthesis_bank_train==True)):
        print(f' Training the analysis banks')
        model.gamma_filt_r.weight.requires_grad = True
        model.gamma_filt_r.bias.requires_grad = True
        model.gamma_filt_i.weight.requires_grad = True
        model.gamma_filt_i.bias.requires_grad = True
        model.inv_gamma_filt_r.weight.requires_grad = True
        model.inv_gamma_filt_r.bias.requires_grad = True
        model.inv_gamma_filt_i.weight.requires_grad = True
        model.inv_gamma_filt_i.bias.requires_grad = True

    #for name, param in model.named_parameters():
    #    if (param.requires_grad == True):
    #        print(f' parameters being trained')
    #        print(name)
    
    train_loss, valid_loss = [], []
    
    model.train()
    for idx in range(batch_size):
        # zero gradients
        optimizer.zero_grad()

        # get one row of input
        input_x = Sig[idx, :]
        input_x = torch.FloatTensor(input_x)
        input_x = input_x.unsqueeze(0)
        input_x = input_x.to(device)

        # run forward
        # Note : for now, we assume the input consists of 1 frame of data of length N
        output_real, output_imag, output, power_frame, y_inv = model(input_x)
        output = output.squeeze()
        
        # get the desired result
        target = Label[idx,:]             # cochleogram output
        target = torch.FloatTensor(target)
        target = target.to(device)
        
        target_signal = Sig[idx, :]
        target_signal = torch.FloatTensor(target_signal)
        target_signal = target_signal.to(device)

        # backward prop
        if( (analysis_bank_train == True) & (synthesis_bank_train==False)):
            loss = loss_fct(output[:,:], target)
            train_loss.append(loss.item())
            loss.backward()
            optim.step()

        if( (synthesis_bank_train==True) & (analysis_bank_train == False)):
            loss_recover_signal =loss_fct(y_inv, target_signal)
            train_loss.append(loss_recover_signal.item())
            loss_recover_signal.backward()
            optim.step()

        if( (synthesis_bank_train==True) & (analysis_bank_train == True)):
            loss_recover_signal =loss_fct(y_inv, target_signal)
            train_loss.append(loss_recover_signal.item())
            loss_recover_signal.backward()
            optim.step()

        if( (synthesis_bank_train==False) & (analysis_bank_train == False)):
            train_loss.append(0)



    ## evaluation part
    model.eval()
    for idx in range(batch_size):
        # get one row of input
        
        input_x_valid = input_validation[idx, :]
        # input_x has to be in [1, L]    shape
        input_x_valid = torch.FloatTensor(input_x_valid)
        input_x_valid = input_x_valid.unsqueeze(0)
        input_x_valid = input_x_valid.to(device)
        #
        # call model
        output_valid_real, output_valid_imag, output_valid, power_frame, y_inv_valid = model(input_x_valid)
        output_valid = output_valid.squeeze()

        #
        # get the desired result
        target_valid = Label_validation[idx, :]
        target_valid = torch.FloatTensor(target_valid)
        target_valid = target_valid.to(device)

        target_signal_valid = input_validation[idx, :]
        target_signal_valid = torch.FloatTensor(target_signal_valid)
        target_signal_valid = target_signal_valid.to(device)


        #
        if( (analysis_bank_train == True) & (synthesis_bank_train==False)):
            loss_valid = loss_fct(output_valid[:,:N], target_valid)
            valid_loss.append(loss_valid.item())
        if( (synthesis_bank_train==True) & (analysis_bank_train == False)):
            loss_valid = loss_fct(y_inv_valid[:N], target_signal_valid)
            valid_loss.append(loss_valid.item())

    return train_loss, valid_loss




#optimizer = optim.Adagrad(c_gama.parameters(), lr=1.0, lr_decay=1e-2, weight_decay=1e-05, initial_accumulator_value=0)  # works for 2 banks

epoch_train_loss, epoch_valid_loss  = [], []

for epoch in range(nb_batches):  # loop over the dataset multiple times
    #scheduler.step()
    # get training data
    # each epoch , use a batch : 2D matrix make
    sig = dataset[epoch]
    sig_validation = dataset_validation[epoch]

    # construct the real and hilbert of the gamma filters
    #
    analytic_subband_signal, env_subband_signal = cgram.cochleagram(signal=sig, sr=sample_rate, n=n_gamma_demo, low_lim=low_lim, hi_lim=high_lim, sample_factor=2,
        padding_size=None, downsample=None, nonlinearity=None, fft_mode='auto', ret_mode='analytic', strict=False)
    
    Label = np.hstack([analytic_subband_signal.real, analytic_subband_signal.imag])
    Label_env = env_subband_signal
    
    # Validation Labels
    # -----------------
    analytic_subband_signal_validation, env_subband_sig_validation = cgram.cochleagram(signal=sig_validation, sr=sample_rate, n=n_gamma_demo, low_lim=low_lim, hi_lim=high_lim, sample_factor=2, padding_size=None, downsample=None, nonlinearity=None, fft_mode='auto', ret_mode='analytic', strict=False)

    Label_validation = np.hstack([analytic_subband_signal_validation.real, analytic_subband_signal_validation.imag])

    if ( (initialize_ana_kernels==False) &  (initialize_syn_kernels==False) ):
        # Do a sequential training : Analysis bank for 100 then synthesis bank for 100 epochs
        if( epoch < 100 ):
            train_loss, valid_loss = train_banks(model=c_gama, epoch_ind=epoch, Sig=sig, Label=Label, batch_size=batch_size, input_validation=sig_validation, Label_validation= Label_validation, N=N, loss_fct=loss_function, optim=optimizer, analysis_bank_train=True, synthesis_bank_train=False)
        else:
            train_loss, valid_loss = train_banks(model=c_gama, epoch_ind=epoch, Sig=sig, Label=Label, batch_size=batch_size, input_validation=sig_validation, Label_validation= Label_validation, N=N, loss_fct=loss_function, optim=optimizer, analysis_bank_train=False, synthesis_bank_train=True)

    if ( (initialize_ana_kernels==True) &  (initialize_syn_kernels==False) ):
        # The analysis bank is loaded w/ solution , just train synthesis bank
        train_loss, valid_loss = train_banks(model=c_gama, epoch_ind=epoch, Sig=sig, Label=Label, batch_size=batch_size, input_validation=sig_validation, Label_validation= Label_validation, N=N, loss_fct=loss_function, optim=optimizer, analysis_bank_train=False, synthesis_bank_train=True)


    epoch_train_loss.append(np.mean(train_loss))
    epoch_valid_loss.append(np.mean(valid_loss))
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))




# plot loss curve

epoch_nb = np.arange(0,nb_batches,1)
epoch_train_loss = np.array(epoch_train_loss)
print(f'epoch nb {epoch_nb.shape}, {epoch_train_loss.shape}')
plt.figure(figsize=(10, 4));
plt.subplot(1, 2, 1);
plt.plot(epoch_nb, epoch_train_loss)
plt.xlabel('Epoch');
plt.ylabel('Training Loss');
plt.subplot(1, 2, 2);
plt.plot(epoch_nb, epoch_valid_loss)
plt.xlabel('Epoch');
plt.ylabel('Validation Loss');

plt.figure(figsize=(5, 4));
plt.plot(epoch_nb, epoch_train_loss)
plt.xlabel('Epoch');
plt.ylabel('Training Loss (Linear)');
plt.savefig('Training_loss_lin.png')


plt.figure(figsize=(5, 4));
plt.plot(epoch_nb, np.log(epoch_train_loss+1.0e-10))
plt.xlabel('Epoch');
plt.ylabel('Training Loss (Log)');
plt.savefig('Training_loss_log.png')


# Plot the final kernels

plot_final_kernels(model=c_gama, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, filename_kernels='Final_Gamma_Kernels', filename_gamma='Final_Gama')

evaluate_cochleogram(model=c_gama, n_gamma_model=n_gamma, n_gama_coh = n_gamma_demo, low_lim=low_lim, hi_lim=high_lim, sample_rate=sample_rate, eval_filename='final_model_gamma', librosa_filename='final_librosa_gamma.png', fr_len=N, synthesis_file='final_resyn_file.wav')

plt.show()

exit()
