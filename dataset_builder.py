import os
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
                extract_mfcc(song_path)

build_dataset()
os.listdir("dataset")