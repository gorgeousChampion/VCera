from processing.demucs import separate_vocals
from processing.extractor import extract_mfcc


def get_user_vector(audio_path):
    audio_path = separate_vocals(audio_path)

    if audio_path is None:
        return None


    vector = extract_mfcc(audio_path)

    return vector