-# GammatoneFb
 Cochleogram and Gammatone Analysis Synthesis Banks

This code base implements the networks described in the paper: 

** Audio Cochleogram with Analysis and Synthesis Banks Using 1D Convolutional Networks
**************************************************************************************


Place all .py file in the main folder 

edit the parameters in file :       run_conv1d_model.py

======================================================
N = 512                                    # Frame length
NFFT = N
hop_length = NFFT//2
sample_rate = 16000
nb_batches = 200                            # nb of epochs
batch_size = 100                            # nb of samples used in each epoch
n_gamma = 65                                # nb of gama filters ( used in the model here
n_gamma_demo = 30                           # nb of gama filter (used in the PyCochleogram code (target), and it becomes 2*n + 5 )
gamma_filt_len = (N)+1                      # L in the paper

low_lim=26                      # Low frequency limit (Hz)
high_lim=7743

initialize_ana_kernels = False
initialize_syn_kernels = False

======================================================

then start : 

python run_conv1d_model_gamma.py


it will run for all the epochs, and then will plot a bunch of figures as well as save them to disk.




To start from Random kernels and train analysis and synthesis layers, set the following : 

initialize_ana_kernels = False
initialize_syn_kernels = False


To start from an initialized analysis layer, and train the synthesis layer, set the following : 

initialize_ana_kernels = True
initialize_syn_kernels = False



#  Requirements

see the file requirements.txt for the needed modules to install and for info on other 3rd party modules that were integrated.