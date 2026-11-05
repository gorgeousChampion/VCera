
from pathlib import Path
import numpy as np

from processing.ecapa import extract_embedding


DATASET_DIR = Path("dataset")
FEATURES_DIR = Path("features_ecapa")


def build_dataset():
    # Find every training audio file.
    audio_files = sorted(DATASET_DIR.rglob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {DATASET_DIR}")

    # Group embeddings by artist.
    artist_embeddings = {}
    failures = []

    for index, audio_path in enumerate(audio_files, start=1):
        artist = audio_path.parent.name
        song_name = audio_path.stem

        output_dir = FEATURES_DIR / artist
        output_dir.mkdir(parents=True, exist_ok=True)

        song_feature_path = output_dir / f"{song_name}.npy"

        print(f"[{index}/{len(audio_files)}] {artist} - {song_name}")

        try:
            # Reuse an existing embedding if available.
            if song_feature_path.exists():
                embedding = np.load(song_feature_path)
            else:
                embedding = extract_embedding(str(audio_path))
                np.save(song_feature_path, embedding)

            if embedding.shape != (192,):
                raise ValueError(
                    f"Expected a 192-D embedding, got {embedding.shape}"
                )

            if not np.isfinite(embedding).all():
                raise ValueError("Embedding contains NaN or infinite values")

            artist_embeddings.setdefault(artist, []).append(embedding)

        except Exception as exc:
            failures.append((audio_path, str(exc)))
            print(f"  FAILED: {exc}")

    # Average the song embeddings to create one vector per artist.
    for artist, embeddings in artist_embeddings.items():
        artist_vector = np.mean(np.stack(embeddings), axis=0)
        artist_path = FEATURES_DIR / artist / "artist.npy"
        np.save(artist_path, artist_vector)
        print(f"Saved artist vector: {artist_path}")

    print("\nECAPA dataset build complete.")
    print(f"Successful songs: {sum(map(len, artist_embeddings.values()))}")
    print(f"Failed songs: {len(failures)}")

    if failures:
        print("\nFailures:")
        for audio_path, error in failures:
            print(f"- {audio_path}: {error}")


if __name__ == "__main__":
    build_dataset()
