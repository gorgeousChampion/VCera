# VCera — Voice-Based Artist Matching

VCera is an experimental audio machine-learning project that identifies the closest-matching artist for an input vocal recording.

It explores three approaches to audio representation and similarity-based artist matching:

- **Raw MFCC features** — a traditional audio-feature baseline.
- **Standardized MFCC features** — the same features transformed using a training-fitted scaler.
- **ECAPA-TDNN embeddings** — deep speaker representations extracted using a pretrained SpeechBrain model.

Rather than relying on a single representation, VCera runs all three approaches against the same input and produces a side-by-side comparison.

> **Project status:** Experimental. The current implementation supports a CLI-based comparison pipeline. Model evaluation is ongoing, and the available test set is small.

## How It Works

Given an audio recording, VCera:

1. Accepts an input audio file.
2. Optionally separates vocals from the accompaniment using Demucs.
3. Extracts audio representations using three different methods.
4. Compares each representation against stored artist reference vectors using cosine similarity.
5. Ranks artists for each method.
6. Generates a JSON report containing predictions, rankings, similarity scores, and execution timings.
7. Removes its temporary copy of the input audio and intermediate Demucs output.

### Pipeline

```text
                  Input Audio
                       |
                       v
             Temporary Audio Copy
                       |
                       v
              Vocal Separation
                   Demucs
                       |
                       v
                Isolated Vocals
                       |
          +------------+------------+
          |            |            |
          v            v            v
       Raw MFCC   Standardized   ECAPA-TDNN
          |          MFCC        Embedding
          |            |            |
          v            v            v
       26-D Vector  26-D Vector  192-D Vector
          |            |            |
          v            v            v
      MFCC Artist  Standardized  ECAPA Artist
      References   References    References
          |            |            |
          v            v            v
       Cosine       Cosine        Cosine
      Similarity   Similarity    Similarity
          |            |            |
          +------------+------------+
                       |
                       v
              Ranked Predictions
                       |
                       v
                JSON Comparison
                    Report
```

## Feature Extraction Approaches

### E0 — Raw MFCC

The baseline implementation uses Mel-Frequency Cepstral Coefficients (MFCCs).

For each recording:

- Extracts 13 MFCC coefficients.
- Calculates the mean of each coefficient across time.
- Calculates the standard deviation of each coefficient across time.
- Concatenates both sets into a 26-dimensional feature vector.

[
V = [\mu_1,\ldots,\mu\_{13},\sigma_1,\ldots,\sigma\_{13}]
]

Each artist is represented by an average feature vector constructed from their available training recordings.

### E1 — Standardized MFCC

This approach investigates whether standardizing MFCC features improves artist matching.

- Extracts the same 26-dimensional MFCC representation.
- Fits a `StandardScaler` on individual training-song feature vectors.
- Applies the fitted scaler to both training and inference features.
- Rebuilds artist reference vectors in standardized feature space.

The scaler is saved as `models/scaler.joblib` and reused during inference.

### E2 — ECAPA-TDNN

VCera also experiments with deep speaker embeddings using SpeechBrain's pretrained ECAPA-TDNN model:

`speechbrain/spkrec-ecapa-voxceleb`

The model generates a 192-dimensional embedding for an input recording.

These embeddings are compared against stored artist reference embeddings using cosine similarity.

**Important:** ECAPA-TDNN is pretrained for speaker recognition, not artist identification. VCera explores whether its speaker representations can also be useful for matching recordings to artist reference profiles. This is an experimental application, not a claim that the pretrained model was trained to identify musical artists.

## Experimental Results

The following results come from four manually evaluated recordings in the current experiment log.

| Input        | Expected artist | Raw MFCC               | Standardized MFCC        | ECAPA-TDNN             |
| ------------ | --------------- | ---------------------- | ------------------------ | ---------------------- |
| WILDFLOWER   | Billie Eilish   | Sid Sriram (Rank 2)    | Billie Eilish (Rank 1)   | Billie Eilish (Rank 1) |
| Maruvarthai  | Sid Sriram      | Sid Sriram (Rank 1)    | Michael Jackson (Rank 2) | Sid Sriram (Rank 1)    |
| Self Control | Frank Ocean     | Adele (Rank 3)         | Adele (Rank 2)           | Frank Ocean (Rank 1)   |
| Valerie      | Amy Winehouse   | Amy Winehouse (Rank 1) | Michael Jackson (Rank 2) | Amy Winehouse (Rank 1) |

### Preliminary top-1 results

| Method            | Correct predictions | Test recordings |
| ----------------- | ------------------: | --------------: |
| Raw MFCC          |                   2 |               4 |
| Standardized MFCC |                   1 |               4 |
| ECAPA-TDNN        |                   4 |               4 |

These results suggest that ECAPA embeddings are promising for this small test set.

However, four recordings are insufficient to establish generalization or a reliable accuracy estimate. More artists, recordings, recording conditions, and independent test samples are needed.

Cosine similarity scores are not probabilities. Scores from different feature representations should not be compared directly.

## Tech Stack

| Component               | Technology                  |
| ----------------------- | --------------------------- |
| Language                | Python                      |
| Audio processing        | Librosa, FFmpeg             |
| Vocal separation        | Demucs                      |
| Audio features          | MFCC                        |
| Deep embeddings         | SpeechBrain, ECAPA-TDNN     |
| Numerical computation   | NumPy                       |
| Feature standardization | Scikit-learn                |
| Model persistence       | Joblib                      |
| Similarity metric       | Cosine similarity           |
| Audio downloading       | yt-dlp                      |
| Hardware acceleration   | PyTorch CUDA when available |

## Project Structure

```text
VCera/
│
├── compare_input.py          # Main three-method comparison pipeline
├── main.py                   # CLI entry point
├── matcher.py                # Cosine similarity and artist ranking
│
├── artist_builder.py         # Builds average raw MFCC artist vectors
├── dataset_builder.py        # Extracts MFCCs from dataset recordings
├── downloader.py             # Downloads artist recordings
├── user.py                   # User-audio MFCC processing
│
├── dataset/                  # Labeled training audio
│   ├── artist_1/
│   ├── artist_2/
│   └── ...
│
├── features/                 # Raw MFCC feature vectors
├── features_standardized/    # Standardized MFCC feature vectors
├── features_ecapa/           # ECAPA embedding references
│
├── models/
│   └── scaler.joblib         # Fitted MFCC scaler
│
├── pretrained_models/
│   └── ecapa/                # SpeechBrain model files
│
├── processing/
│   ├── extractor.py          # MFCC extraction
│   ├── demucs.py             # Vocal separation
│   ├── standardize.py        # Standardized feature generation
│   ├── ecapa.py              # ECAPA embedding extraction
│   ├── build_ecapa.py        # ECAPA feature-building workflow
│   └── vad.py                # Voice activity detection experiments
│
├── experiments/
│   ├── results.csv           # Experiment results
│   └── evaluate_ecapa.py     # ECAPA evaluation
│
└── comparison_outputs/       # Generated JSON comparison reports
```

Generated feature databases, model files, downloaded audio, temporary processing data, and virtual environments are excluded from version control.

## Getting Started

### Prerequisites

- Python 3.12 recommended.
- FFmpeg installed and available through `PATH`.
- Git.
- NVIDIA GPU with compatible CUDA/PyTorch support is optional. CPU inference is also supported.

### 1. Clone the repository

```bash
git clone https://github.com/gorgeousChampion/VCera.git
cd VCera
```

### 2. Create a virtual environment

For Windows PowerShell:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install numpy librosa joblib scikit-learn torch speechbrain demucs yt-dlp
```

Install a PyTorch build compatible with your CUDA setup if you intend to use GPU acceleration. Otherwise, the CPU build is sufficient.

### 4. Install FFmpeg

Install FFmpeg and add its `bin` directory to your system `PATH`.

Verify the installation:

```powershell
ffmpeg -version
```

### 5. Run VCera

```powershell
python main.py "path/to/audio.mp3"
```

The first run may download the pretrained ECAPA model if it is not already available locally.

## Usage

### Compare an ordinary audio recording

```powershell
python compare_input.py "path/to/song.mp3"
```

VCera will attempt vocal separation using Demucs before extracting features.

### Compare an already-isolated vocal recording

If the supplied audio already contains isolated vocals, skip Demucs:

```powershell
python compare_input.py "path/to/vocals.wav" --isolated
```

### View available command-line options

```powershell
python compare_input.py --help
```

## Output

Each successful comparison generates a timestamped JSON report inside:

```text
comparison_outputs/
```

The report contains:

- Input filename.
- Whether Demucs was skipped.
- Processing timestamp.
- Total pipeline execution time.
- Predicted artist for each method.
- Ranked artist matches.
- Cosine similarity scores.
- Feature extraction and matching timings.

Temporary copies of user audio and intermediate Demucs files are removed after processing. The original input file is not modified or deleted.

## Dataset and Artist References

VCera uses labeled artist recordings to construct reference representations.

The expected dataset layout is:

```text
dataset/
├── Adele/
│   ├── song_001.wav
│   ├── song_002.wav
│   └── ...
│
├── Billie Eilish/
│   ├── song_001.wav
│   └── ...
│
└── ...
```

Raw MFCC feature extraction is implemented in `dataset_builder.py`. Average raw MFCC artist references are constructed using `artist_builder.py`.

The standardized MFCC workflow uses `processing/standardize.py`.

The project also contains an ECAPA feature-building workflow in `processing/build_ecapa.py`.

**Note:** The generated feature directories and fitted model artifacts are not committed to the repository. They must be created from an appropriately prepared dataset before the comparison pipeline can run successfully.

## Current Limitations

- The evaluation dataset is currently very small.
- ECAPA-TDNN is a pretrained speaker-recognition model being explored for artist matching.
- Similarity scores are not calibrated confidence probabilities.
- Average artist embeddings may not represent variations across songs, vocal styles, or recording conditions.
- Vocal separation can introduce artifacts that affect feature extraction.
- Differences in recording quality, background noise, mixing, and vocal processing may affect results.
- The current implementation uses reference-vector similarity rather than a separately trained artist-classification model.

## Future Work

Potential directions for further development include:

- Expanding the artist dataset and evaluation set.
- Evaluating recordings from unseen songs and different recording conditions.
- Investigating voice activity detection and its effect on embeddings.
- Comparing alternative audio representations.
- Exploring artist-specific embedding aggregation strategies.
- Improving evaluation with top-k accuracy, confusion matrices, and more robust validation.
- Investigating embedding normalization and score calibration.
- Building a more accessible interface for audio submission and comparison.

## Motivation

VCera began as an exploration of audio feature extraction and similarity-based matching. It has since evolved into an experimental comparison of traditional signal-processing features and deep-learning-based speaker representations.

The central question is:

**How much does the choice of audio representation affect the quality of voice-based artist matching?**

VCera is an attempt to investigate that question through implementation, experimentation, and measurable comparisons.

---

*Built as an independent machine-learning project.*
