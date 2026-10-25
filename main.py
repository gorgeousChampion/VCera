
import os
import sys
import joblib
import numpy as np

from user import get_user_vector
from matcher import match_artists


audio_path = input("Input a user song file: ").strip().strip('"')

if not os.path.isfile(audio_path):
    print("Invalid audio path")
    sys.exit(1)

user_vector = get_user_vector(audio_path)

if user_vector is None:
    print("Audio processing failed")
    sys.exit(1)

# E0: Original MFCC features.
baseline_results = match_artists(user_vector, feature_dir="features")

# E1: Apply the TRAINING scaler to the same user vector.
scaler = joblib.load("models/scaler.joblib")

standardized_vector = scaler.transform(
    np.asarray(user_vector).reshape(1, -1)
)[0]

standardized_results = match_artists(
    standardized_vector,
    feature_dir="features_standardized"
)

print("\n=== E0: ORIGINAL MFCC ===")
for rank, (artist, score) in enumerate(baseline_results, start=1):
    print(f"{rank}. {artist}: {float(score):.6f}")

print("\n=== E1: STANDARDIZED MFCC ===")
for rank, (artist, score) in enumerate(standardized_results, start=1):
    print(f"{rank}. {artist}: {float(score):.6f}")
