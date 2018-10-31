# -*- coding: utf-8 -*-
"""
Created on May 18, 2018

@author: lisu, Bill

Document:

load_audio(filepath, sr=None, mono=True, dtype='float32')
    Parameters:
        sr:(number>0) sample rate;
            default = None(use raw audio sample rate)
        mono:(bool) convert signal to mono;
            default = True
        dtype:(numeric type) data type of x;
            default = 'float32'
    Returns:
        x:(np.ndarray) audio time series
        sr:(number>0) sample rate of x
feature_extraction(x, sr, Hop=320, Window=2049, StartFreq=80.0, StopFreq=1000.0, NumPerOct=48)
    Parameters:
        x:(np.ndarray) audio time series
        sr:(number>0) sample rate of x
        Hop: Hop size
        Window: Window size
        StartFreq: smallest frequency on feature map
        StopFreq: largest frequency on feature map
        NumPerOct: Number of bins per octave
    Returns:
        Z: mix cfp feature
        time: feature map to time
        CenFreq: feature map to frequency
        tfrL0: STFT spectrogram
        tfrLF: generalized cepstrum (GC)
        tfrLQ: generalized cepstrum of spectrum (GCOS)

get_CenFreq(StartFreq=80, StopFreq=1000, NumPerOct=48)
get_time(fs, Hop, end)
midi2hz(midi)
hz2midi(hz)

"""
import soundfile as sf
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
import scipy.signal
import pandas as pd


def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/float(Hop))*Hop, Hop)
    N = int(fs/float(fr))
    window_size = len(h)
    f = fs*np.linspace(0, 0.5, np.round(N/2), endpoint=True)
    Lh = int(np.floor(float(window_size-1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)     
        
    for icol in range(0, len(t)):
        ti = int(t[icol])           
        tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                        int(min([round(N/2.0)-1, Lh, len(x)-ti])))
        indices = np.mod(N + tau, N) + 1                                             
        tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                /np.linalg.norm(h[Lh+tau-1])           
                            
    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))  
    return tfr, f, t, N

def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g!=0:
        X[X<0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X

def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        l = int(round(central_freq[i-1]/fr))
        r = int(round(central_freq[i+1]/fr)+1)
        #rounding1
        if l >= r-1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq

def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1/q
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
            if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

    tfrL = np.dot(freq_band_transformation, ceps)
    return tfrL, central_freq

def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    NumofLayer = np.size(g)

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr # original STFT
    ceps = np.zeros(tfr.shape)

    if NumofLayer >= 2:
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs*tc)
                ceps = np.real(np.fft.fft(tfr, axis=0))/np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc/fr)
                tfr = np.real(np.fft.fft(ceps, axis=0))/np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)

    tfr0 = tfr0[:int(round(N/2)),:]
    tfr = tfr[:int(round(N/2)),:]
    ceps = ceps[:int(round(N/2)),:]

    HighFreqIdx = int(round((1/tc)/fr)+1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx,:]
    tfr = tfr[:HighFreqIdx,:]
    HighQuefIdx = int(round(fs/fc)+1)
    
    q = np.arange(HighQuefIdx)/float(fs)
    
    ceps = ceps[:HighQuefIdx,:]
    
    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies 

def load_audio(filepath, sr=None, mono=True, dtype='float32'):

    if '.mp3' in filepath:
        from pydub import AudioSegment
        import tempfile
        import os
        mp3 = AudioSegment.from_mp3(filepath)
        _, path = tempfile.mkstemp()
        mp3.export(path, format="wav")
        del mp3
        x, fs = sf.read(path)
        os.remove(path)
    else:
        x, fs = sf.read(filepath)

    if mono and len(x.shape)>1:
        x = np.mean(x, axis = 1)
    if sr:
        x = scipy.signal.resample_poly(x, sr, fs)
        fs = sr 
    x = x.astype(dtype)

    return x, fs


def feature_extraction(x, fs, Hop=512, Window=2049, StartFreq=80.0, StopFreq=1000.0, NumPerOct=48):
    
    fr = 2.0 # frequency resolution    
    h = scipy.signal.blackmanharris(Window) # window size
    g = np.array([0.24, 0.6, 1]) # gamma value

    tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, StartFreq, 1/StopFreq, g, NumPerOct)
    Z = tfrLF * tfrLQ
    time = t/fs
    return Z, time, CenFreq, tfrL0, tfrLF, tfrLQ


def midi2hz(midi):
    return 2**((midi-69)/12.0)*440
def hz2midi(hz):
    return 69+ 12*np.log2(hz/440.0)
    
def get_CenFreq(StartFreq=80, StopFreq=1000, NumPerOct=48):
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    return central_freq

def get_time(fs, Hop, end):
    return np.arange(Hop/fs,end,Hop/fs)

def lognorm(x):
    return np.log(1+x)
def norm(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))
def cfp_process(fpath, ypath=None, csv=False, hop=256, model_type='vocal'):
    print('CFP process in '+str(fpath)+ ' ... (It may take some times)')
    y, sr = load_audio(fpath)
    if 'vocal' in model_type:
        Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, StartFreq=31.0, StopFreq=1250.0, NumPerOct=60)
    if 'melody' in model_type:
        Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, StartFreq=20.0, StopFreq=2048.0, NumPerOct=60)
    tfrL0 = norm(lognorm(tfrL0))[np.newaxis,:,:]
    tfrLF = norm(lognorm(tfrLF))[np.newaxis,:,:]
    tfrLQ = norm(lognorm(tfrLQ))[np.newaxis,:,:]
    W = np.concatenate((tfrL0,tfrLF,tfrLQ),axis=0)
    print('Done!')
    print('Data shape: '+str(W.shape))
    if ypath:
        if csv:
            ycsv = pd.read_csv(ypath, names = ["time", "freq"])
            gt0 = ycsv['time'].values
            gt0 = gt0[1:,np.newaxis]

            gt1 = ycsv['freq'].values
            gt1 = gt1[1:,np.newaxis]
            gt = np.concatenate((gt0, gt1), axis=1)
        else:
            gt = np.loadtxt(ypath)
        return W, gt, CenFreq, time
    else:
        return W, CenFreq, time


