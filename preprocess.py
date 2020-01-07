import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from sklearn import model_selection
from sklearn import preprocessing





def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x , fs

def calculate_melsp(x , fs, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    # melsp = librosa.feature.mfcc(S=log_stft, n_mfcc = 128)
    return melsp

def add_white_noise(x, rate=0.002):
    ''' data augmentation: add white noise '''
    return x + rate*np.random.randn(len(x))

def shift_sound(x, rate=2):
    ''' data augmentation: shift sound in timeframe '''
    return np.roll(x, int(len(x)//rate))

def stretch_sound(x, rate=1.1):
    ''' data augmentation: stretch sound '''
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")

def save_np_data(filename, x, y, aug=None, rates=None):
    freq = 128
    time = 1723
    np_data = np.zeros(freq*time*len(x)).reshape(len(x), freq, time)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x, fs = load_wave_data('./data/ESC-50-master/audio/', x[i])
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x , fs)
        np_data[i] = _x
        np_targets[i] = y[i]
    np.savez(filename, x=np_data, y=np_targets)

if __name__ == '__main__':
    metaData = pd.read_csv('./data/ESC-50-master/meta/esc50.csv')

    x = list(metaData.loc[:,"filename"])
    y = list(metaData.loc[:, "target"])
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, stratify=y)




    freq = 128
    time = 1723
    if not os.path.exists("./data/training/esc_melsp_test.npz"):
        save_np_data("./data/training/esc_melsp_test.npz", x_test,  y_test)
    if not os.path.exists("./data/training/esc_melsp_train_raw.npz"):
        save_np_data("./data/training/esc_melsp_train_raw.npz", x_train,  y_train)
    if not os.path.exists("./data/training/esc_melsp_train_wn.npz"):
        rates = np.random.randint(1,50,len(x_train))/10000
        save_np_data("./data/training/esc_melsp_train_wn.npz", x_train,  y_train, aug=add_white_noise, rates=rates)
    if not os.path.exists("./data/training/esc_melsp_train_ss.npz"):
        rates = np.random.choice(np.arange(2,6),len(y_train))
        save_np_data("./data/training/esc_melsp_train_ss.npz", x_train,  y_train, aug=shift_sound, rates=rates)
    if not os.path.exists("./data/training/esc_melsp_train_st.npz"):
        rates = np.random.choice(np.arange(80,120),len(y_train))/100
        save_np_data("./data/training/esc_melsp_train_st.npz", x_train,  y_train, aug=stretch_sound, rates=rates)
    if not os.path.exists("./data/training/esc_melsp_train_com.npz"):
        np_data = np.zeros(freq*time*len(x_train)).reshape(len(x_train), freq, time)
        np_targets = np.zeros(len(y_train))
        for i in range(len(y_train)):
            x, fs = load_wave_data('./data/ESC-50-master/audio/', x_train[i])
            x = add_white_noise(x=x, rate=np.random.randint(1,50)/1000)
            if np.random.choice((True,False)):
                x = shift_sound(x=x, rate=np.random.choice(np.arange(2,6)))
            else:
                x = stretch_sound(x=x, rate=np.random.choice(np.arange(80,120))/100)
            x = calculate_melsp(x , fs)
            np_data[i] = x
            np_targets[i] = y_train[i]
        np.savez("./data/training/esc_melsp_train_com.npz", x=np_data, y=np_targets)