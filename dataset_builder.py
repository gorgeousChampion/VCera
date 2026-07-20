import os
import numpy as np
from extractor import extract_mfcc


def build_dataset():
    for artist in os.listdir("dataset"):
        artist_path = os.path.join("dataset", artist)

        if not os.path.isdir(artist_path):
            continue

        for song in os.listdir(artist_path):
            if song.endswith(".wav"):
                song_path = os.path.join(artist_path, song)
                print(f"Processing {artist}/{song}")
                vector = extract_mfcc(song_path)
                
                base = os.path.basename(song_path)
                name, _ = os.path.splitext(base)

                feature_folder = os.path.join("features", artist)

                if not os.path.exists(feature_folder):
                    os.makedirs(feature_folder)

                feature_path = os.path.join(feature_folder, name)

                np.save(feature_path, vector)

                print(f"Saved {feature_path}.npy")

build_dataset()