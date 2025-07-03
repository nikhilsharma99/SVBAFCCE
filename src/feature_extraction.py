import librosa
import numpy as np
import os
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, sample_rate=16000, duration=2):
        self.sample_rate = sample_rate
        self.duration = duration

    def extract_features(self, audio_path):
        try:
            # Load audio with optimized parameters
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            # Use fewer MFCC coefficients for faster processing
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            return mfcc_scaled
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None