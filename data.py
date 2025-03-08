import os 
from tqdm import tqdm #you can watch for loops as it is done
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from python_speech_features import mfcc,logfbank
import librosa


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size = 16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size = 16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y,freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('filter bank Coefficients', size = 16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap = 'hot', interpolation = 'nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency cepstrum Coeffiecients', size = 16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap = 'hot', interpolation = 'nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

# this is to filter the low signals ... by specifying specific threshold
# we get an array of each y whethere it true or fale if it exeeds or under the threshold
def envelope(y,rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)

df = pd.read_csv('reciters.csv')

df.set_index('fname', inplace=True) # instead of specifying the inde of the file now its name 

# for f in df.index:
#     rate, signal = wavfile.read('dataset/'+f)
#     df.at[f,'length'] = signal.shape[0]/rate    #just gives you wav length


# =========================plot length for each instrument
# classes = list(np.unique(df.label)) #
# class_dist = df.groupby(['label'])['length'].mean() #groups all labels and takes its length mean

# fig, ax = plt.subplots()
# ax.set_title('Class Distribution', y = 1.08)
# ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
#         shadow=False, startangle=90)
# ax.axis('equal')

# df.reset_index(inplace=True) #reset df back 






# =================== differect plots =======================
# you need this to filter you data 
signals = {}
fft = {}
fbank = {}
mfccs = {}

# for c in classes:
#     wav_file = df[df.label == c].iloc[0,0]
#     signal, rate = librosa.load('dataset/' + wav_file, sr=44100)#you can get the sample rate using scipy
   
#     mask = envelope(signal,rate, 0.0005)
#     signal = signal[mask]       #the fails values wil be deleted from the aray
   
#     signals[c] = signal
#     fft[c] = calc_fft(signal, rate)

#     bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T#nfft = sr/40 
#     fbank[c] = bank
#     mel = mfcc(signal[:rate],rate, numcep=13, nfilt=26, nfft=1103).T
#     mfccs[c] = mel

# plot_signals(signals)
# plt.show()

# plot_fft(fft)
# plt.show()

# plot_fbank(fbank)
# plt.show()

# plot_mfccs(mfccs)
# plt.show

# cleaning files ,,, ready for modeling 
if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        signal,rate = librosa.load('dataset/'+f, sr=16000)
        mask = envelope(signal,rate,0.0005)
        wavfile.write(filename='clean/'+f,rate=rate,data=signal[mask])

signal, rate = librosa.load('cut_7s_audio.wav', sr= 16000)
mask = envelope(signal,rate,0.0005)
wavfile.write(filename='output_test.wav',rate=rate, data=signal[mask] )
        