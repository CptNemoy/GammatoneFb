import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import librosa
import os
from pycochleagram import utils
from scipy.signal import chirp, spectrogram
from scipy.signal import sweep_poly
from soothingsounds import generator

class ComplexNumbersDataset(Dataset):
    def __init__(self, nb_batches, batch_size, N, file, sr, content = 'Rand', complex=True):
        self.nb_batches = nb_batches
        self.batch_size = batch_size
        
        if ((content == 'Audio') & (complex == False) ):
            if (file == None ):
                f = f = os.path.join('/Users/elinemer/cnn_gamatone_fb/Music/', 'orchestra01.wav')
            else:
                f = os.path.join('/Users/elinemer/cnn_gamatone_fb/Music/', file)
            #audio, sr = librosa.load(f)
            audio, sr = utils.wav_to_array(f)
            
            len = audio.size
            if( (batch_size * nb_batches * N) > len ):
                print(f'Audio file not long enough. len = {len}, product = {batch_size * nb_batches * N}')
                exit()
            else:
                fr_len = N
                nb_frames = len // fr_len
                nb_bat = nb_frames // batch_size
                self.samples = np.resize(audio, [nb_bat, batch_size, fr_len])

        if ((content == 'Audio') & (complex == True) ):
            print(f' Cannot generate complex audio')
            exit()
            
        if ((content == 'Rand') & (complex == True) ):
            self.samples = np.random.randn(nb_batches, batch_size, N) + 1j*np.random.randn(nb_batches, batch_size, N)
        if ((content == 'Rand') & (complex == False) ):
            self.samples = np.zeros((nb_batches, batch_size, N), dtype=float)
            # Loop over the batches
            for b in range (nb_batches):
                # generate random noise
                len = batch_size * N
                # noise_sig = np.random.randn(len)
                noise_sig = generator.white(len)

                # generate a chirp signal for the entire batch
                len_time = len / sr      # duration in seconds
                t = np.linspace(0, len_time, num = len, dtype=float)
                t1 = len_time/2
                f0 = np.random.uniform(10, sr//4, 1)
                f1 = np.random.uniform(sr//4, sr//2, 1)
                phi = np.random.uniform(0, 360, 1)
                #print(f' len = {len}, len time = {len_time}, size of t = {t.shape}, f0 = {f0}, f1 = {f1}, phi = {phi}')
                A = chirp(t, f0, len_time, f1, method='linear', phi = np.round(phi))
                #plt.plot(t, A)
                #plt.title("Linear Chirp, f(0)=6, f(10)=1")
                #plt.xlabel('t (sec)')

                # generate a generalized sweep
                a0 = np.random.uniform(0, 0.125, 1)
                a1 = np.random.uniform(-0.5, 0.5, 1)
                a2 = np.random.uniform(0.75, 1.5, 1)
                a3 = np.random.uniform(1.5, 2.5, 1)
                #print(f' aaa {a0}, {a1}, {a2}, {a3}')
                p = np.poly1d([a0[0], a1[0], a2[0], a3[0]])
                w = sweep_poly(t, p)
                
                
                # generate Amplitude-moduled signal
                AM = self.amplitude_modulation(t, sr)
                #plt.plot(t, AM)
                #plt.title("AM")
                #plt.xlabel('t (sec)')

                PN = generator.pink(len)
                #plt.plot(PN)
                #plt.title("Pink")
                #plt.xlabel('t (sec)')
                BN = generator.blue(len)
                VN = generator.violet(len)
                
                alpha = np.random.uniform(0.25,0.45, 1)
                beta = np.random.uniform(0,0.1, 1)
                delta = np.random.uniform(0,0.1, 1)
                
                if ((b % 6) == 0):
                    Combo_sig = (alpha[0] * noise_sig) + ((1.0-alpha[0]) * A)
                if ((b % 6) == 1):
                    Combo_sig = (alpha[0] * noise_sig) + (delta[0] * w)
                if ((b % 6) == 2):
                    Combo_sig = (alpha[0] * noise_sig) + ((1.0-alpha[0]) * A) + (delta[0] * AM)
                    #Combo_sig = (alpha[0] * noise_sig) + (beta[0] * A) + (delta[0] * AM)
                if ((b % 6) == 3):
                    #Combo_sig = (alpha[0] * noise_sig) + ((1.0-alpha[0]) * A) + (delta[0] * w)
                    Combo_sig = (alpha[0] * noise_sig) + (beta[0] * A) + (delta[0] * w)
                if ((b % 6) == 4):
                    Combo_sig = (alpha[0] * noise_sig) + (delta[0] * AM)
                if ((b % 6) == 5):
                    Combo_sig = (alpha[0] * noise_sig) +  (delta[0] * AM) + (delta[0] * w)

                #Combo_sig = (alpha[0] * noise_sig) + (delta[0] * A) + (delta[0] * w) + (delta[0] * AM) + (delta[0] * PN)
                #Combo_sig = (alpha[0] * noise_sig) + (delta[0] * A) + (delta[0] * w) + (beta[0] * AM) + (beta[0] * PN)
                Combo_sig = (noise_sig)
                #Combo_sig = AM
                self.samples[b,:, :] = np.resize(Combo_sig, [batch_size, N])
                #plt.plot(t, Combo_sig)
                #plt.title("Combo_sig")
                #plt.xlabel('t (sec)')
                #plt.show()
                
                #self.samples = np.random.uniform(-1, 1, (nb_batches, batch_size, N))
        #plt.show()
        print(f'size of samples = {(self.samples).shape}')
    #
    def __len__(self):
        return self.nb_batches
    #
    def __getitem__(self, idx):
        return self.samples[idx, :, :]

    def amplitude_modulation(self,t, sr):
        #A_c = carrier amplitude:
        #f_c = carrier frquency:
        #A_m = message amplitude:
        #f_m = message frquency:
        #modulation_index = modulation index:
        #
        A_c = np.random.uniform(0, 0.5, 1)
        f_c = np.random.uniform(0, sr/2, 1)
        A_m = np.random.uniform(0, 0.75, 1)
        f_m = np.random.uniform(0, sr/10, 1)
        modulation_index = np.random.uniform(0, 1.0, 1)
        #
        carrier = A_c*np.cos(2*np.pi*f_c*t)
        modulator = A_m*np.cos(2*np.pi*f_m*t)
        product = A_c*(1+modulation_index*np.cos(2*np.pi*f_m*t))*np.cos(2*np.pi*f_c*t)
        return product





if __name__ == '__main__':
    dataset = ComplexNumbersDataset(3, 2, 10)   # create a dataset of complex numbers
    print(f'len = {len(dataset)}')
    #print(f' shape of 1st batch = {(dataset[0,:,:]).shape}')
    print(f' dataset of 0 = {dataset[0]}')
    
