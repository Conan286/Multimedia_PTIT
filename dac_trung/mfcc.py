import librosa
import numpy as np

def mfcc_function(y, sr):
    """
    Trích xuất 13 hệ số MFCC trung bình
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)
