import os
import subprocess
import sys


artist = input("Artist: ")

if not os.path.exists(artist):
    os.mkdir(artist)

urls = input("URLs: ").split()


def download_song(artist, url):
    command = [
    sys.executable, 
    "-m",
    "yt_dlp",
    "--cookies", "cookies.txt",
    "--js-runtimes", "node",
    "--remote-components", "ejs:github",
    "-x",
    "--audio-format", "wav",
    "-o", f"{artist}/%(title)s.%(ext)s",
    url
    ]

    try:
        subprocess.run(command, check=True)
        print("Download complete")
    except subprocess.CalledProcessError as e:
        print("Failed:", e)


for i, url in enumerate(urls):
    print(f"Downloading {i+1}/{len(urls)}")
    download_song(artist, url)

def get_song_path(artist):
    files = os.listdir(artist) 
    for file in files:
        if file.endswith(".wav"):
            return {artist}/{file}
        