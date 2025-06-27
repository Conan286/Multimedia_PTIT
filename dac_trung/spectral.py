import librosa
import numpy as np

def spectral_centroid_function(y, sr):
    """
    Trích xuất độ sáng phổ âm thanh trung bình
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)
