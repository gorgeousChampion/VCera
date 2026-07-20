import librosa
import numpy as np
import os
def extract_mfcc(song_path):
    audio,sr=librosa.load(song_path)
    print(type(audio))
    print(audio.shape)
    print(sr)

    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13)

    mfcc_mean=np.mean(mfcc,axis=1)

    mfcc_std=np.std(mfcc,axis=1)
  
    features = np.concatenate((mfcc_mean, mfcc_std))

    return features

