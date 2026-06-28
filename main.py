import os
import subprocess
import sys
import shutil

artist = input("Artist: ")

if not os.path.exists(artist):
    os.mkdir(artist)

urls = input("URLs: ").split()
demucs_ip=input("Run Demucs? Yes or No:-")

def download_song(artist, url, filename):
    command = [
    sys.executable, 
    "-m",
    "yt_dlp",
    "--cookies", "cookies.txt",
    "--js-runtimes", "node",
    "--remote-components", "ejs:github",
    "-x",
    "--audio-format", "wav",
    "-o", f"{artist}/{filename}",
    url
    ]

    try:
        subprocess.run(command, check=True)
        print("Download complete")
        return f"{artist}/{filename}"
    except subprocess.CalledProcessError as e:
        print("Failed:", e)
        return None
artist_name = artist.lower().replace(" ", "_")

def separate_song(song_path):
        command=[
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

def get_song_number(artist):
    files = os.listdir(artist)
    count = 0   
    for file in files:
        if file.endswith(".wav"):
            count+=1
    return count

def process_vocals(song_path):
    base=os.path.basename(song_path)
    name,_=os.path.splitext(base)
    vocals_path=os.path.join("separated","htdemucs",name,"vocals.wav")
    if not os.path.exists(vocals_path):
        print("vocals not found!")
        return 
    os.remove(song_path)
    shutil.move(vocals_path,song_path)
    song_folder=os.path.join("separated","htdemucs",name)
    shutil.rmtree(song_folder)

for i, url in enumerate(urls):
    print(f"Downloading {i+1}/{len(urls)}")
    count = get_song_number(artist)
    filename = f"{artist_name}_{count+1:03d}.wav"
    print(filename)
    song_path= download_song(artist,url,filename)
    if song_path is not None :
        if demucs_ip=="Yes":
            separate_song(song_path)
            process_vocals(song_path)