
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from matcher import match_artists
from processing.extractor import extract_mfcc
from processing.ecapa import extract_embedding


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "comparison_outputs"


def prepare_audio(input_path, already_isolated=False, work_dir=None):
    if already_isolated:
        print("[AUDIO] Using supplied isolated vocals.")
        return input_path

    if work_dir is None:
        raise ValueError("A temporary workspace is required for Demucs.")

    print("[DEMUCS] Separating vocals...")
    start = time.perf_counter()

    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        "htdemucs",
        "-o",
        str(work_dir),
        str(input_path),
    ]

    subprocess.run(command, check=True, cwd=ROOT)

    vocals_path = (
        work_dir
        / "htdemucs"
        / input_path.stem
        / "vocals.wav"
    )

    if not vocals_path.is_file():
        raise FileNotFoundError(
            f"Demucs completed but vocals.wav was not found: {vocals_path}"
        )

    elapsed = time.perf_counter() - start
    print(f"[DEMUCS] Completed in {elapsed:.2f}s.")

    return vocals_path


def timed_extraction(function, audio_path):
    start = time.perf_counter()
    vector = function(audio_path)
    elapsed = time.perf_counter() - start

    return np.asarray(vector), elapsed


def run_method(name, vector, feature_dir):
    start = time.perf_counter()

    matches = match_artists(
        np.asarray(vector),
        feature_dir=str(ROOT / feature_dir),
    )

    elapsed = time.perf_counter() - start

    if not matches:
        raise RuntimeError(
            f"No valid artist matches found for {name} in {feature_dir}."
        )

    rankings = []

    for rank, (artist, score) in enumerate(matches, start=1):
        rankings.append(
            {
                "rank": rank,
                "artist": str(artist),
                "similarity": float(score),
            }
        )

    return {
        "method": name,
        "prediction": rankings[0]["artist"],
        "top_score": rankings[0]["similarity"],
        "matching_seconds": elapsed,
        "rankings": rankings,
    }


def print_results(results):
    print("\n" + "=" * 65)
    print("VCera - ARTIST MATCHING RESULTS")
    print("=" * 65)

    for result in results:
        print(f"\n[{result['method']}]")

        for item in result["rankings"][:3]:
            print(
                f"  {item['rank']}. "
                f"{item['artist']:<22} "
                f"{item['similarity']:.4f}"
            )

        print(
            f"  Extraction: {result['extraction_seconds']:.2f}s | "
            f"Matching: {result['matching_seconds']:.2f}s | "
            f"Total: {result['total_seconds']:.2f}s"
        )

    print("\n" + "-" * 65)
    print("PREDICTION SUMMARY")
    print("-" * 65)

    predictions = []

    for result in results:
        print(f"{result['method']}: {result['prediction']}")
        predictions.append(result["prediction"])

    if len(set(predictions)) == 1:
        print("\nAll methods agree.")
    else:
        print("\nMethods produced different predictions.")

    print(
        "\nNote: Cosine similarities are not probabilities. "
        "Scores from different feature methods are not directly comparable."
    )


def run_comparison(input_name, vocals_path, args, pipeline_start):
    results = []

    # Extract MFCC once for both E0 and E1.
    print("\n[E0/E1] Extracting MFCC features...")

    mfcc_vector, mfcc_seconds = timed_extraction(
        extract_mfcc,
        str(vocals_path),
    )

    mfcc_vector = mfcc_vector.reshape(-1)

    if mfcc_vector.shape != (26,) or not np.all(np.isfinite(mfcc_vector)):
        raise ValueError(
            f"Invalid MFCC vector. Expected 26 finite values, "
            f"received shape {mfcc_vector.shape}."
        )

    # E0: Raw MFCC
    e0 = run_method(
        "E0 - Raw MFCC",
        mfcc_vector,
        "features",
    )

    e0["extraction_seconds"] = mfcc_seconds
    e0["total_seconds"] = (
        mfcc_seconds + e0["matching_seconds"]
    )

    results.append(e0)

    # E1: Standardized MFCC
    print("[E1] Applying training-fitted scaler...")

    scaler_path = ROOT / "models" / "scaler.joblib"

    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Training-fitted scaler not found: {scaler_path}"
        )

    standardization_start = time.perf_counter()

    scaler = joblib.load(scaler_path)

    standardized_vector = scaler.transform(
        mfcc_vector.reshape(1, -1)
    ).reshape(-1)

    standardization_seconds = (
        time.perf_counter() - standardization_start
    )

    if (
        standardized_vector.shape != (26,)
        or not np.all(np.isfinite(standardized_vector))
    ):
        raise ValueError("Invalid standardized MFCC vector.")

    e1 = run_method(
        "E1 - Standardized MFCC",
        standardized_vector,
        "features_standardized",
    )

    e1["extraction_seconds"] = (
        mfcc_seconds + standardization_seconds
    )

    e1["total_seconds"] = (
        e1["extraction_seconds"] + e1["matching_seconds"]
    )

    results.append(e1)

    # E2: ECAPA speaker embedding
    print("[E2] Extracting ECAPA embedding...")

    ecapa_vector, ecapa_seconds = timed_extraction(
        extract_embedding,
        str(vocals_path),
    )

    ecapa_vector = ecapa_vector.reshape(-1)

    if (
        ecapa_vector.shape != (192,)
        or not np.all(np.isfinite(ecapa_vector))
    ):
        raise ValueError(
            f"Invalid ECAPA vector. Expected 192 finite values, "
            f"received shape {ecapa_vector.shape}."
        )

    e2 = run_method(
        "E2 - ECAPA",
        ecapa_vector,
        "features_ecapa",
    )

    e2["extraction_seconds"] = ecapa_seconds
    e2["total_seconds"] = (
        ecapa_seconds + e2["matching_seconds"]
    )

    results.append(e2)

    pipeline_seconds = time.perf_counter() - pipeline_start

    print_results(results)

    # No input audio, embeddings, or absolute input paths are persisted.
    report = {
        "input_file": input_name,
        "processed_audio": (
            "supplied isolated vocals"
            if args.isolated
            else "temporary Demucs vocals"
        ),
        "demucs_skipped": bool(args.isolated),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline_seconds": pipeline_seconds,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_name = (
        "comparison_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )

    report_path = OUTPUT_DIR / report_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print(f"\n[REPORT] Saved: {report_path}")
    print(f"[TOTAL] Pipeline completed in {pipeline_seconds:.2f}s.")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="VCera: Compare user audio against stored artist features."
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the input audio file.",
    )

    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Skip Demucs when the supplied audio already contains isolated vocals.",
    )

    args = parser.parse_args()

    input_path = args.audio.expanduser().resolve()

    if not input_path.is_file():
        parser.error(f"Input audio file does not exist: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_name = input_path.name
    pipeline_start = time.perf_counter()

    print("\n[VCera] Starting artist comparison...")
    print(f"[INPUT] {input_name}")

    # All VCera-managed user audio lives inside this temporary workspace.
    # The original user-owned file is never modified or deleted.
    #
    # TemporaryDirectory automatically cleans up on normal exit and
    # on exceptions raised inside the context.
    with tempfile.TemporaryDirectory(
        prefix="vcera_",
        dir=OUTPUT_DIR,
    ) as temporary_path:

        workspace = Path(temporary_path)

        managed_input = workspace / input_name

        print("[INPUT] Creating temporary managed copy...")

        shutil.copy2(input_path, managed_input)

        vocals_path = prepare_audio(
            managed_input,
            already_isolated=args.isolated,
            work_dir=workspace,
        )

        run_comparison(
            input_name=input_name,
            vocals_path=vocals_path,
            args=args,
            pipeline_start=pipeline_start,
        )

    print("[CLEANUP] Temporary user audio and Demucs files removed.")
    print("[VCera] Done.")


if __name__ == "__main__":
    main()
