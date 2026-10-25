
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def build_standardized_features():
    source_dir = "features"
    output_dir = "features_standardized"

    song_vectors = []
    song_records = []

    # Collect individual training-song vectors.
    for artist in sorted(os.listdir(source_dir)):
        artist_path = os.path.join(source_dir, artist)

        if not os.path.isdir(artist_path):
            continue

        for filename in sorted(os.listdir(artist_path)):
            if not filename.endswith(".npy") or filename == "artist.npy":
                continue

            source_path = os.path.join(artist_path, filename)
            vector = np.load(source_path, allow_pickle=False)

            if vector.shape != (26,) or not np.all(np.isfinite(vector)):
                print(f"Skipping invalid vector: {source_path}")
                continue

            song_vectors.append(vector)
            song_records.append((artist, filename))

    if not song_vectors:
        raise RuntimeError("No valid training vectors found.")

    # Shape: (number_of_songs, 26)
    X = np.stack(song_vectors)

    # Fit ONE scaler using all training songs.
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save the fitted scaler for inference.
    joblib.dump(scaler, "models/scaler.joblib")

    artist_vectors = {}

    # Save every standardized song vector separately.
    for (artist, filename), vector in zip(song_records, X_standardized):
        artist_output = os.path.join(output_dir, artist)
        os.makedirs(artist_output, exist_ok=True)

        np.save(os.path.join(artist_output, filename), vector)

        artist_vectors.setdefault(artist, []).append(vector)

    # Rebuild each artist's average in standardized feature space.
    for artist, vectors in artist_vectors.items():
        average_vector = np.mean(np.stack(vectors), axis=0)

        output_path = os.path.join(output_dir, artist, "artist.npy")
        np.save(output_path, average_vector)

        print(f"{artist}: {len(vectors)} songs standardized")

    print(f"\nProcessed {len(song_vectors)} training songs.")
    print("Scaler saved to models/scaler.joblib")
    print("Standardized features saved to features_standardized/")


if __name__ == "__main__":
    build_standardized_features()
