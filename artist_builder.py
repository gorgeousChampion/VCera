import os
import numpy as np

def build_artist_vectors():
    for artist in os.listdir("features"):
        artist_path = os.path.join("features", artist)

        if not os.path.isdir(artist_path):
            continue

        vectors = []

        for file in os.listdir(artist_path):
            if file.endswith(".npy") and file != "artist.npy":
                file_path = os.path.join(artist_path, file)

                vector = np.load(file_path)
                vectors.append(vector)

        if len(vectors) == 0:
            continue

        vectors = np.array(vectors)

        artist_vector = np.mean(vectors, axis=0)
        artist_file = os.path.join(artist_path, "artist.npy")
        np.save(artist_file, artist_vector)
        print(f"Created {artist_file}")
