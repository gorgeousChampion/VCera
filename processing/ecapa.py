
import subprocess
import shutil

import numpy as np
import torch

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None


def get_model():
    global _model

    if _model is None:
        print(f"Loading ECAPA-TDNN on {DEVICE}...")

        _model = EncoderClassifier.from_hparams(
            source=MODEL_ID,
            savedir="pretrained_models/ecapa",
            run_opts={"device": DEVICE},
            local_strategy=LocalStrategy.COPY,
        )

        _model.eval()

    return _model


def extract_embedding(audio_path):
    model = get_model()

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise FileNotFoundError(
            "FFmpeg not found. Add its bin directory to PATH."
        )

    result = subprocess.run(
        [
            ffmpeg,
            "-v", "error",
            "-i", str(audio_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )

    if not result.stdout:
        raise ValueError(
            f"FFmpeg produced no audio samples: {audio_path}"
        )

    waveform = torch.from_numpy(
        np.frombuffer(result.stdout, dtype=np.float32).copy()
    ).unsqueeze(0)

    waveform = waveform.to(DEVICE)

    with torch.inference_mode():
        embedding = model.encode_batch(waveform)

    vector = embedding.squeeze().cpu().numpy()

    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(
            f"Invalid ECAPA embedding: shape={vector.shape}"
        )

    return vector
