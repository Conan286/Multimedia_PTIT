import librosa
import numpy as np

def zcr_function(y):
    """
    Trích xuất tỷ lệ zero-crossing trung bình
    """
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr)
