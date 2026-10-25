import os
import sys
import shutil
import subprocess


def separate_vocals(song_path):
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        song_path,
    ]

    try:
        subprocess.run(command, check=True)
        print("Separation complete")
    except subprocess.CalledProcessError as e:
        print("Failed:", e)
        return None

    base = os.path.basename(song_path)
    name, _ = os.path.splitext(base)

    vocals_path = os.path.join("separated", "htdemucs", name, "vocals.wav")

    if not os.path.exists(vocals_path):
        print("Vocals not found!")
        return None

    os.remove(song_path)
    shutil.move(vocals_path, song_path)

    song_folder = os.path.join("separated", "htdemucs", name)
    shutil.rmtree(song_folder)

    return song_path