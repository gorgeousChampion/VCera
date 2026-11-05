
import csv
from pathlib import Path

from matcher import match_artists
from processing.ecapa import extract_embedding

FEATURE_DIR = "features_ecapa"
RESULTS_FILE = Path("experiments") / "results.csv"

TEST_CLIPS = [
    (
        "WILDFLOWER isolated vocals",
        "Billie Eilish",
        Path(r"C:\Users\shrey\OneDrive\Desktop\VCera\Billie Eilish - WILDFLOWER Isolated Vocals-[AudioTrimmer.com].mp3"),
    ),
    (
        "Maruvarthai",
        "Sid Sriram",
        Path(r"C:\Users\shrey\OneDrive\Desktop\VCera\Maruvarthai unplugged Sid Sriram-[AudioTrimmer.com].mp3"),
    ),
    (
        "Self Control",
        "Frank Ocean",
        Path(r"C:\Users\shrey\OneDrive\Desktop\VCera\Frank Ocean - Self Control-[AudioTrimmer.com].mp3"),
    ),
    (
        "Valerie",
        "Amy Winehouse",
        Path(r"C:\Users\shrey\OneDrive\Desktop\VCera\Amy Winehouse - Valerie Acoustic Live Best Quality-[AudioTrimmer.com].mp3"),
    ),
]


def evaluate():
    results = []

    for clip_name, expected_artist, audio_path in TEST_CLIPS:
        if not audio_path.is_file():
            print(f"\nFile not found: {audio_path}")
            continue

        print(f"\nEvaluating: {clip_name}")

        user_vector = extract_embedding(str(audio_path))
        rankings = match_artists(user_vector, feature_dir=FEATURE_DIR)

        if not rankings:
            print("No compatible artist vectors found.")
            continue

        expected = next(
            (
                item for item in rankings
                if item[0].casefold() == expected_artist.casefold()
            ),
            None,
        )

        expected_rank = next(
            (
                rank
                for rank, (artist, _) in enumerate(rankings, start=1)
                if artist.casefold() == expected_artist.casefold()
            ),
            "",
        )

        top_artist, top_score = rankings[0]
        expected_score = expected[1] if expected else ""

        print(f"Expected: {expected_artist}")
        print(f"Predicted: {top_artist}")
        print(f"Expected rank: {expected_rank}")
        print("Top matches:")

        for rank, (artist, score) in enumerate(rankings[:3], start=1):
            print(f"  {rank}. {artist}: {score:.4f}")

        results.append({
            "experiment_id": "E2",
            "input": clip_name,
            "expected_artist": expected_artist,
            "top1_prediction": top_artist,
            "expected_rank": expected_rank,
            "top1_score": top_score,
            "expected_artist_score": expected_score,
            "notes": "Pretrained ECAPA-TDNN embeddings; cosine similarity",
        })

    if not results:
        print("\nNo results generated.")
        return

    fieldnames = [
        "experiment_id",
        "input",
        "expected_artist",
        "top1_prediction",
        "expected_rank",
        "top1_score",
        "expected_artist_score",
        "notes",
    ]

    # Replace existing E2 rows instead of accumulating duplicates.
    existing_rows = []
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_rows = [
                row for row in reader
                if row.get("experiment_id") != "E2"
            ]

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(results)

    correct = sum(
        row["top1_prediction"].casefold()
        == row["expected_artist"].casefold()
        for row in results
    )

    print(f"\nE2 accuracy: {correct}/{len(results)}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate()
