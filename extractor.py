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

    base=os.path.basename(song_path)
    
    artist = os.path.dirname(song_path)
    
    feature_folder = os.path.join("features", artist)
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    
    name,_=os.path.splitext(base)
    
    feature_path=os.path.join(feature_folder,name)
    
    np.save(feature_path,features)
    
    print(f"Saved features to {feature_path}.npy")
    return features

