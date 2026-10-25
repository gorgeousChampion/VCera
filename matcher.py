import os
import numpy as np

def cosine_similarity(a,b):
    dot_product=np.dot(a,b)

    magnitude_a=np.linalg.norm(a)
    magnitude_b=np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    similarity=dot_product / (magnitude_a * magnitude_b)

    return similarity

def match_artists(user_vector, feature_dir="features"):
    result = []

    for artist in os.listdir(feature_dir):
        artist_path = os.path.join(feature_dir, artist)

        if not os.path.isdir(artist_path):
            continue

        vector_path = os.path.join(artist_path, "artist.npy")

        if not os.path.isfile(vector_path):
            continue

        artist_vector = np.load(vector_path)

        if user_vector.shape!=artist_vector.shape:
            continue

        score = cosine_similarity(user_vector,artist_vector)
        
        result.append((artist,score))

    result.sort(key= lambda result:result[1], reverse=True)
    return result
        
