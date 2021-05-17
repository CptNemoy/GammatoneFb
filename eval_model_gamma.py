import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
from torchsummary import summary
import librosa
from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb
from pycochleagram import utils
from scipy.io.wavfile import write
from create_kernels_gamatone import create_gamatone_filterbank


device = 'cpu'


def evaluate_cochleogram(model, n_gamma_model, n_gama_coh, low_lim, hi_lim, sample_rate, eval_filename, librosa_filename, fr_len, synthesis_file):
    # evaluate model
    model.eval()


    eval_filename_spect_only = eval_filename + '_spect_only.png'
    eval_filename_all = eval_filename + '_time_spect.png'

    audio, sr = librosa.load('got_s2e9_cake.wav')
    #audio, sr = librosa.load('abba.wav')
    duration = librosa.get_duration(y=audio, sr=sr)
    
    len = audio.size
    nb_frames = len // fr_len
    new_array = np.resize(audio, [nb_frames, fr_len])

    coch_pow_batches = cgram.cochleagram(signal=new_array, sr=sample_rate, n=n_gama_coh, low_lim=low_lim, hi_lim=hi_lim, sample_factor=2,
        padding_size=None, downsample=None, nonlinearity='power', fft_mode='auto', ret_mode='envs', strict=False)
        
    coch_pow = coch_pow_batches[0,:,:]
  
    for fr in range(1,coch_pow_batches.shape[0]):
      coch_pow = np.append(coch_pow, coch_pow_batches[fr, :, :], axis=1)

    coch_pow_all_at_once  = cgram.cochleagram(signal=audio, sr=sample_rate, n=n_gama_coh, low_lim=low_lim, hi_lim=hi_lim, sample_factor=2,
        padding_size=None, downsample=None, nonlinearity='power', fft_mode='auto', ret_mode='envs', strict=False)

    analytic_subband_signal, env_sb = cgram.cochleagram(signal=audio, sr=sample_rate, n=n_gama_coh, low_lim=low_lim, hi_lim=hi_lim, sample_factor=2,padding_size=None, downsample=None, nonlinearity=None, fft_mode='auto', ret_mode='analytic', strict=False)

    new_env = cgram.apply_envelope_nonlinearity(analytic_subband_signal, nonlinearity='power')
    
    img = np.flipud(coch_pow)  # the cochleagram is upside down (i.e., in image coordinates)

    signal=audio
  
  
    plt.figure(figsize=(8, 5))
    plt.subplot(221)
    plt.title('Input Time Signal')
    plt.plot(signal)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (Samples)')

    plt.subplot(222)
    plt.title('Cochleagram (FFT-based)')
    plt.ylabel('Filter Nb')
    plt.xlabel('Time (Samples)')
    utils.cochshow(np.flipud(coch_pow_all_at_once), interact=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # convert to tensor to call model
    # ===============================
    audio_or = audio
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)


    output_real, output_imag, output_eval, power_frames_eval, recovered_signal = model(audio)

    
    p_frames = power_frames_eval.cpu().data.numpy()

    plt.subplot(224)
    plt.title('Cochleagram (Model Output)')
    plt.ylabel('Filter Nb')
    plt.xlabel('Time (Samples)')
    utils.cochshow(np.flipud(p_frames), interact=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    syn_frames = recovered_signal.cpu().data.numpy()
    write(synthesis_file, rate = sample_rate, data=syn_frames)
    
    plt.subplot(223)
    plt.title('Re-Synthesized Signal ')
    plt.plot(syn_frames)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (Samples)')

    plt.savefig(eval_filename_all)


    plt.figure(figsize=(10, 4))
    plt.subplot(211)
    plt.title('Cochleagram (FFT-based)')
    plt.ylabel('Filter Nb')
    plt.xlabel('Time (Samples)')
    utils.cochshow(np.flipud(coch_pow_all_at_once), interact=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.subplot(212)
    plt.title('Cochleagram (Model Output)')
    plt.ylabel('Filter Nb')
    plt.xlabel('Time (Samples)')
    utils.cochshow(np.flipud(p_frames), interact=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(eval_filename_spect_only)


    len = min(p_frames.shape[1], coch_pow_all_at_once.shape[1])
    coc_power_err = np.mean((p_frames[:,:len] - coch_pow_all_at_once[:,:len])**2.0)
    print(f' Difference between cochleogram (all at once) = {coc_power_err}')

    len = min(p_frames.shape[1], coch_pow.shape[1])
    coc_power_err2 = np.mean((p_frames[:,:len] - coch_pow[:,:len])**2.0)
    print(f' Difference between cochleogram (frame based) = {coc_power_err2}')

    
    len = min(syn_frames.shape[0], audio_or.shape[0])
    print(f' shape {syn_frames.shape},  {audio_or.shape}')
    reconst_time_err = np.mean((syn_frames[:len]- audio_or[:len])**2.0)
    print(f' reconstruction error = {reconst_time_err}')
    
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1);
    utils.cochshow((coch_pow), interact=False)
    plt.colorbar()
    plt.title('pycho - Power')
    # plot the librosa chroma
    plt.subplot(1, 2, 2);
    utils.cochshow((p_frames), interact=False)
    plt.colorbar()
    plt.title('Model out ')
    plt.savefig(librosa_filename)
    




def plot_initial_kernels(model, hop_length, NFFT, sr, filename_kernels, filename_gamma):
    #
    filename_mels_filters = 'temp.png'
    filename_gamma_filters_ana = filename_gamma + '_select_kernels_analysis.png'
    filename_gamma_filters_syn = filename_gamma + '_select_kernels_synthesis.png'
    


    # 3D plots
    filename_kernels_analysis = filename_kernels + '_analysis_bank.png'
    filename_kernels_synthesis = filename_kernels + '_synthesis_bank.png'
    filename_kernels_analysis_DFT = filename_kernels + '_analysis_bank_dft.png'
    filename_kernels_synthesis_DFT = filename_kernels + '_synthesis_bank_dft.png'



    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.gamma_filt_r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    librosa.display.specshow(r_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Initial Kernel Analysis Coefficients (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.gamma_filt_i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    librosa.display.specshow(i_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Initial Kernel Analysis Coefficients (Imag)');
    plt.savefig(filename_kernels_analysis)
    #plt.show()



    plt.figure(figsize=(9, 6));
    plt.subplot(2,1,1)
    idxs_to_plot = [5, 11, 31]
    for i in idxs_to_plot:
        plt.plot(r_kernels[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Initial Analysis filters (Real) ');
 
    plt.subplot(2,1,2)
    for i in idxs_to_plot:
        plt.plot(i_kernels[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Initial Analysis filters (Imag) ');
    plt.savefig(filename_gamma_filters_ana)


    real_kernels_dft = np.fft.rfft(r_kernels, axis=-1)
    imag_kernels_dft = np.fft.rfft(i_kernels, axis=-1)

    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    plt.imshow(np.abs(real_kernels_dft), aspect='auto', origin='lower')
    plt.title('DFT-Mag Initial Analysis Filters (Main) ');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Freq Index');
    plt.colorbar();
    plt.subplot(1, 2, 2);
    plt.imshow(np.abs(imag_kernels_dft), aspect='auto', origin='lower')
    plt.title('DFT-Mag Initial Analysis Filters (Hilb) ');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Freq Index');
    plt.colorbar();
    plt.savefig(filename_kernels_analysis_DFT)



    # plot the synthesis filters
    synthesis_kernels_r = model.inv_gamma_filt_r.weight.data
    synthesis_kernels_r = synthesis_kernels_r.squeeze()
    s_kernels_r = np.array(synthesis_kernels_r)

    synthesis_kernels_i = model.inv_gamma_filt_i.weight.data
    synthesis_kernels_i = synthesis_kernels_i.squeeze()
    s_kernels_i = np.array(synthesis_kernels_i)
    
    
    plt.figure(figsize=(9, 6));
    plt.subplot(2,1,1)
    idxs_to_plot = [5, 11, 31]
    for i in idxs_to_plot:
        plt.plot(s_kernels_r[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Initial Synthesis Filters (Real) ');
    plt.subplot(2,1,2)
    for i in idxs_to_plot:
        plt.plot(s_kernels_i[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Initial Synthesis Filters (Imag) ');
    plt.savefig(filename_gamma_filters_syn)
    
    
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    librosa.display.specshow(s_kernels_r, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Initial Synthesis Filters (Real)');
    plt.subplot(1, 2, 2);
    librosa.display.specshow(s_kernels_i, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Initial Synthesis Filters (Imag)');
    plt.savefig(filename_kernels_synthesis)



def plot_final_kernels(model, NFFT, hop_length, sr, filename_kernels, filename_gamma):
    filename_gamma_filters_ana = filename_gamma + '_select_kernels_analysis.png'
    filename_gamma_filters_syn = filename_gamma + '_select_kernels_synthesis.png'
    filename_kernels_analysis = filename_kernels + '_analysis_bank.png'
    filename_kernels_synthesis = filename_kernels + '_synthesis_bank.png'
    filename_kernels_analysis_DFT = filename_kernels + '_analysis_bank_DFT.png'
    filename_kernels_synthesis_DFT = filename_kernels + '_synthesis_bank_DFT.png'
    filename_kernels_analysis_DFT_syn_DFT = filename_kernels + '_analysis_times_synthesis_DFT.png'

    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.gamma_filt_r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    librosa.display.specshow(r_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Final Analysis Filters (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.gamma_filt_i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    librosa.display.specshow(i_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Final Analysis Filters  (Imag)');
    plt.savefig(filename_kernels_analysis)
    #plt.show()


    main_analysis_kernels_dft = np.fft.rfft(r_kernels, axis=-1)
    hilb_analysis_kernels_dft = np.fft.rfft(i_kernels, axis=-1)

    # get the frequency filters from PyCochleogram

    fft_filts, fft_hz_cutoffs, fft_freqs = create_gamatone_filterbank(filter_length=model.gamma_filter_len, sr=sr, nb_filters=30, low_lim=26, hi_lim=7743)
    fft_pts= (model.gamma_filter_len//2) + 1;


    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    plt.imshow(np.abs(main_analysis_kernels_dft), aspect='auto', origin='lower')
    plt.title('DFT-Mag Analysis Filters (Main Bank) ');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Freq Index');
    plt.colorbar();
    plt.clim(0, 0.75);
    plt.subplot(1, 2, 2);
    plt.imshow(np.abs(fft_filts[:,:fft_pts]), aspect='auto', origin='lower')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Freq Index');
    plt.title('DFT-Mag PyCochlegram Filters ');
    plt.colorbar();
    plt.clim(0, 0.75);
    plt.savefig(filename_kernels_analysis_DFT)


    plt.figure(figsize=(9, 6));
    plt.subplot(2,1,1)
    idxs_to_plot = [5, 11, 31]
    for i in idxs_to_plot:
        plt.plot(r_kernels[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Analysis filters (Real) ');
 
    plt.subplot(2,1,2)
    for i in idxs_to_plot:
        plt.plot(i_kernels[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Analysis filters (Imag)');
    plt.savefig(filename_gamma_filters_ana)
    
    
    # plot the synthesis filters
    synthesis_kernels_r = model.inv_gamma_filt_r.weight.data
    synthesis_kernels_r = synthesis_kernels_r.squeeze()
    s_kernels_r = np.array(synthesis_kernels_r)

    synthesis_kernels_i = model.inv_gamma_filt_i.weight.data
    synthesis_kernels_i = synthesis_kernels_i.squeeze()
    s_kernels_i = np.array(synthesis_kernels_i)
    
    main_synthesis_kernels_dft = np.fft.rfft(s_kernels_r, axis=-1)
    hilb_synthesis_kernels_dft = np.fft.rfft(s_kernels_i, axis=-1)


    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    plt.imshow(np.abs(main_synthesis_kernels_dft), aspect='auto', origin='lower')
    plt.title('DFT-Mag Final Synthesis Filters (Main Bank) ');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Freq Index');
    plt.colorbar();
    plt.subplot(1, 2, 2);
    plt.imshow(np.abs(hilb_synthesis_kernels_dft), aspect='auto', origin='lower')
    plt.title('DFT-Mag Final Synthesis Filters (Hilbert) ');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Freq Index');
    plt.colorbar();
    plt.savefig(filename_kernels_synthesis_DFT)



    plt.figure(figsize=(9, 6));
    plt.subplot(2,1,1)
    idxs_to_plot = [5, 11, 31]
    for i in idxs_to_plot:
        plt.plot(s_kernels_r[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Synthesis Filters (Real) ');
    plt.subplot(2,1,2)
    for i in idxs_to_plot:
        plt.plot(s_kernels_i[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Synthesis Filters (Imag)');
    plt.savefig(filename_gamma_filters_syn)


    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    librosa.display.specshow(s_kernels_r, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Final Synthesis Filters (Real)');
    plt.subplot(1, 2, 2);
    librosa.display.specshow(s_kernels_i, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Time Sample');
    plt.colorbar();
    plt.title('Final Synthesis Filters (Imag)');
    plt.savefig(filename_kernels_synthesis)


