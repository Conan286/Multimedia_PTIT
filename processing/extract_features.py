import librosa
import numpy as np
from dac_trung.mfcc import mfcc_function
from dac_trung.chroma import chroma_function
from dac_trung.zcr import zcr_function
from dac_trung.spectral import spectral_centroid_function
from dac_trung.tonnetz import tonnetz_function

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    return np.concatenate([
        mfcc_function(y, sr),
        chroma_function(y, sr),
        [spectral_centroid_function(y, sr)],
        [zcr_function(y)],
        tonnetz_function(y, sr)
    ])
