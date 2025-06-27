import librosa
import numpy as np

def chroma_function(y, sr):
    """
    Trích xuất 12 giá trị chroma trung bình
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)
