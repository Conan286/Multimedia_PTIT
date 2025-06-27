import librosa
import numpy as np

def tonnetz_function(y, sr):
    """
    Trích xuất 6 thành phần Tonnetz trung bình
    """
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    return np.mean(tonnetz, axis=1)
