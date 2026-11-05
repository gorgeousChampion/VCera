# VCera
### Find Your Voice. Find Your Artists. Find Your Songs.

VCera is an experimental machine-learning project that helps people discover musical artists whose vocal characteristics closely resemble their own.

The idea goes beyond simply finding out which singer someone sounds like. By identifying artists with similar vocal characteristics, VCera aims to help users discover songs that may be a more natural fit for their voices when they sing.

Ever struggled to find songs that actually suit your voice? Or wondered which artists you should explore if you want to sing songs that feel comfortable and natural rather than out of place?

VCera explores a way to approach that problem using audio processing, feature extraction, and similarity-based machine learning.

The project currently compares three approaches:

- **Raw MFCC:** Traditional audio features using statistical summaries of Mel-Frequency Cepstral Coefficients.
- **Standardized MFCC:** MFCC features transformed using a scaler fitted on training recordings.
- **ECAPA-TDNN:** Deep speaker embeddings extracted using a pretrained SpeechBrain model.

Each approach generates a representation of a voice, which is compared against reference representations of musical artists to identify the closest matches.

> **Project status:** Experimental. The current implementation provides a command-line comparison pipeline. Preliminary experiments have been conducted on a small set of musical recordings. Evaluating how effectively the system matches user voices to artists, and whether those matches help users discover suitable songs, remains ongoing work.

---

## Table of Contents

- [The Idea Behind VCera](#the-idea-behind-vcera)
- [How It Works](#how-it-works)
- [Feature Extraction Approaches](#feature-extraction-approaches)
- [Preliminary Experiments](#preliminary-experiments)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Output](#output)
- [Dataset Preparation](#dataset-preparation)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)

---

## The Idea Behind VCera

Choosing a song to sing is not just about liking the song.

A song can sound incredible when performed by its original artist but feel uncomfortable or out of place when someone else tries to sing it. Differences in vocal characteristics, pitch range, tone, and delivery can make certain songs a better fit for some people than others.

This led to the central idea behind VCera:

**What if people could discover songs by first finding the artists whose voices most closely resemble their own?**

Instead of randomly choosing songs or relying entirely on subjective recommendations, VCera explores using voice similarity as a starting point.

The intended experience is straightforward:

1. A user provides a recording of their voice.
2. VCera analyzes its vocal characteristics.
3. The system compares the recording against reference representations of musical artists.
4. It returns a ranked list of the closest-matching artists.
5. Users can explore those artists' songs as potential choices that may better suit their voices.

The broader motivation is to make discovering songs to sing more personal and intuitive.

Voice similarity alone cannot guarantee that a particular song will suit someone. Factors such as vocal range, technique, comfort, and musical expression also matter. However, artist similarity could provide a useful starting point for narrowing down the possibilities.

VCera is an exploration of that idea through machine learning.

---

## How It Works

VCera represents an input recording using three different approaches and compares each representation against stored artist references.

The current pipeline:

1. Accepts an input audio file.
2. Creates a temporary working copy.
3. Optionally separates vocals from accompaniment using Demucs.
4. Extracts three different audio representations.
5. Compares each representation against stored artist reference vectors.
6. Calculates cosine similarity and ranks candidate artists.
7. Generates a JSON report containing predictions, rankings, similarity scores, and execution timings.
8. Removes temporary audio and intermediate processing files.

### System Pipeline

```text
                     User Voice
                         |
                         v
                 Audio Preprocessing
                         |
                         v
                  Processed Audio
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
         Raw MFCC   Standardized   ECAPA-TDNN
                        MFCC       Embedding
             |           |           |
             v           v           v
          26-D       26-D          192-D
          Vector     Vector        Vector
             |           |           |
             v           v           v
          Artist     Artist        Artist
        References  References    References
             |           |           |
             v           v           v
          Cosine      Cosine        Cosine
         Similarity  Similarity    Similarity
             |           |           |
             +-----------+-----------+
                         |
                         v
                 Ranked Artist Matches
                         |
                         v
                  Song Discovery
                    Direction
                         |
                         v
                Songs to Explore
```

The three approaches operate independently on the same input recording. Their rankings can then be compared side by side.

The song-discovery stage represents the intended application of the artist matches. The current implementation focuses on audio representation and artist ranking; it does not yet implement a complete song recommendation or vocal-range compatibility system.

---

## Feature Extraction Approaches

### E0 — Raw MFCC

The baseline uses Mel-Frequency Cepstral Coefficients (MFCCs) to represent an audio recording.

For each recording:

1. Extract 13 MFCC coefficients.
2. Calculate the mean of each coefficient across time.
3. Calculate the standard deviation of each coefficient across time.
4. Concatenate both sets into a 26-dimensional feature vector.

Mathematically:

\[
V = [\mu_1,\ldots,\mu_{13},\sigma_1,\ldots,\sigma_{13}]
\]

Artist reference vectors are constructed by averaging the available training-song feature vectors for each artist.

### E1 — Standardized MFCC

This experiment investigates whether standardizing MFCC features improves similarity-based artist matching.

The process:

1. Extract the same 26-dimensional MFCC representation.
2. Fit a `StandardScaler` using individual training-song feature vectors.
3. Transform training and inference features using the fitted scaler.
4. Rebuild artist reference vectors in the standardized feature space.

The fitted scaler is persisted using Joblib and reused during inference.

### E2 — ECAPA-TDNN

VCera also experiments with deep speaker embeddings using SpeechBrain's pretrained model:

`speechbrain/spkrec-ecapa-voxceleb`

ECAPA-TDNN is a deep neural network architecture designed for speaker verification and speaker recognition.

The model generates a 192-dimensional embedding for an input recording.

These embeddings are compared against stored artist reference embeddings using cosine similarity.

Unlike the statistical MFCC baseline, ECAPA-TDNN learns representations intended to capture speaker-discriminative characteristics.

This makes it an interesting candidate for comparing user voices with musical artist references.

**Important:** ECAPA-TDNN is pretrained for speaker recognition, not musical artist identification or song suitability prediction. Its effectiveness for VCera's intended application is something the project aims to investigate.

---

## Preliminary Experiments

The following results come from four manually evaluated recordings in the current experiment log.

| Input | Expected artist | Raw MFCC | Standardized MFCC | ECAPA-TDNN |
|---|---|---|---|---|
| WILDFLOWER | Billie Eilish | Sid Sriram (2) | Billie Eilish (1) | Billie Eilish (1) |
| Maruvarthai | Sid Sriram | Sid Sriram (1) | Michael Jackson (2) | Sid Sriram (1) |
| Self Control | Frank Ocean | Adele (3) | Adele (2) | Frank Ocean (1) |
| Valerie | Amy Winehouse | Amy Winehouse (1) | Michael Jackson (2) | Amy Winehouse (1) |

### Preliminary Top-1 Results

| Method | Correct predictions | Evaluated recordings | Observed top-1 rate |
|---|---:|---:|---:|
| Raw MFCC | 2 | 4 | 50% |
| Standardized MFCC | 1 | 4 | 25% |
| ECAPA-TDNN | 4 | 4 | 100% |

These results are preliminary and based on only four recordings.

They help evaluate the initial comparison pipeline, but they do not establish how accurately VCera can match different users to artists or whether the resulting matches lead to better song choices.

The intended application requires additional evaluation using user recordings and artist reference recordings, followed by testing whether the recommended songs are comfortable and suitable for users to sing.

**Interpretation notes:**

- Cosine similarity values are not calibrated probabilities.
- Scores from different feature spaces are not directly comparable.
- Artist reference quality and dataset composition can strongly influence rankings.
- Recording quality, background noise, and vocal processing can affect results.
- A correct artist match does not automatically establish song suitability.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Audio processing | Librosa, FFmpeg |
| Vocal separation | Demucs |
| Traditional audio features | MFCC |
| Deep embeddings | SpeechBrain, ECAPA-TDNN |
| Numerical computation | NumPy |
| Feature standardization | Scikit-learn |
| Model persistence | Joblib |
| Similarity metric | Cosine similarity |
| Audio downloading | yt-dlp |
| Deep learning backend | PyTorch |
| GPU acceleration | CUDA when supported |

---

## Project Structure

```text
VCera/
│
├── compare_input.py          # Main three-method comparison pipeline
├── main.py                   # CLI entry point
├── matcher.py                # Cosine similarity and artist ranking
├── user.py                   # User-audio MFCC processing
│
├── artist_builder.py         # Builds average raw MFCC artist vectors
├── dataset_builder.py        # Extracts MFCCs from dataset recordings
├── downloader.py             # Downloads artist recordings
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
│   └── results.csv           # Experiment results
│
├── comparison_outputs/       # Generated JSON comparison reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

Generated datasets, feature vectors, model weights, temporary audio, and virtual environments are excluded from version control.

---

## Getting Started

### Prerequisites

- Python 3.12 recommended.
- Git.
- FFmpeg installed and accessible through `PATH`.
- NVIDIA GPU with a compatible CUDA/PyTorch setup is optional. CPU inference is also supported.

### 1. Clone the repository

```bash
git clone https://github.com/gorgeousChampion/VCera.git
cd VCera
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA environment if GPU acceleration is required. Otherwise, CPU inference is supported.

### 4. Install FFmpeg

Install FFmpeg and ensure its `bin` directory is available through your system `PATH`.

Verify:

```bash
ffmpeg -version
```

### 5. Prepare the dataset and generate references

VCera requires artist reference features before matching can run.

The dataset should follow this structure:

```text
dataset/
├── Adele/
│   ├── song_001.wav
│   └── song_002.wav
│
├── Billie Eilish/
│   ├── song_001.wav
│   └── song_002.wav
│
└── ...
```

The project contains separate workflows for generating reference representations:

- `dataset_builder.py` — raw MFCC feature extraction.
- `artist_builder.py` — raw MFCC artist reference construction.
- `processing/standardize.py` — scaler fitting and standardized feature generation.
- `processing/ecapa.py` — ECAPA embedding extraction.
- `processing/build_ecapa.py` — ECAPA reference-building workflow.

The generated feature directories and fitted scaler are not included in the repository. They must be generated from the prepared dataset before running the comparison pipeline.

### 6. Run a comparison

```powershell
python compare_input.py "path/to/voice_recording.wav" --isolated
```

The `--isolated` flag skips Demucs vocal separation.

The first ECAPA run may download pretrained model files if they are not already available locally.

---

## Usage

### Compare an input voice recording

```powershell
python compare_input.py "path/to/voice_recording.wav" --isolated
```

### Compare a recording containing accompaniment

```powershell
python compare_input.py "path/to/song.mp3"
```

VCera attempts vocal separation using Demucs before feature extraction.

### Compare an already-isolated vocal recording

```powershell
python compare_input.py "path/to/vocals.wav" --isolated
```

### View available command-line options

```powershell
python compare_input.py --help
```

---

## Output

Each successful comparison generates a timestamped JSON report inside:

```text
comparison_outputs/
```

The report includes:

- Input filename.
- Audio processing mode.
- Processing timestamp.
- Total pipeline execution time.
- Predicted artist for each approach.
- Ranked artist candidates.
- Cosine similarity scores.
- Feature extraction and matching timings.

Temporary copies of user audio and intermediate Demucs outputs are cleaned up after processing. The original input file is not modified or deleted.

---

## Dataset Preparation

VCera uses labeled artist recordings to construct reference representations.

The expected dataset structure is:

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

Raw MFCC feature extraction is implemented in `dataset_builder.py`.

Average raw MFCC artist references are constructed using `artist_builder.py`.

Standardized MFCC feature generation is implemented in `processing/standardize.py`.

ECAPA embedding extraction is implemented in `processing/ecapa.py`, with the corresponding feature-building workflow in `processing/build_ecapa.py`.

The generated feature directories and fitted scaler must be available for inference. They are intentionally excluded from Git because they are derived artifacts.

Audio used for experiments should be obtained and used in accordance with applicable copyright and licensing requirements.

---

## Current Limitations

- The current manually evaluated test set is very small.
- Preliminary experiments use musical recordings and do not yet establish performance on user-to-artist matching.
- ECAPA-TDNN is a pretrained speaker-recognition model, not a dedicated musical artist-identification model.
- Similarity between two voices does not necessarily mean their comfortable singing ranges are identical.
- A person's vocal characteristics can vary between speech and singing.
- Differences in recording quality, background noise, mixing, and vocal processing can influence results.
- Cosine similarity is not a calibrated confidence score.
- The current implementation uses reference-vector similarity rather than a separately trained artist-classification model.
- Artist similarity alone cannot guarantee that a particular song will suit a user.

## Future Work

Potential directions include:

- Evaluating ordinary user voice recordings against musical artist reference recordings.
- Expanding the artist dataset and evaluation set.
- Investigating how well speaker embeddings transfer between speaking and singing contexts.
- Exploring vocal-range estimation and song-key compatibility.
- Developing a song-discovery system using the artist matches as an initial recommendation signal.
- Investigating whether matching artists also leads to songs that users find more comfortable to sing.
- Comparing additional audio representations.
- Exploring artist-specific embedding aggregation strategies.
- Evaluating top-k accuracy, confusion matrices, and more robust validation strategies.
- Investigating embedding normalization and score calibration.
- Building a more accessible interface for audio submission and song discovery.
