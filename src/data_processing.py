from feature_extraction import FeatureExtractor
import os
import numpy as np
from tqdm import tqdm

class DataProcessor:
    def __init__(self, dataset_path, n_jobs=4):
        self.dataset_path = dataset_path
        self.feature_extractor = FeatureExtractor()
        self.n_jobs = n_jobs
    
    def process_dataset(self):
        features = []
        labels = []
        
        for speaker in tqdm(os.listdir(self.dataset_path)):
            speaker_path = os.path.join(self.dataset_path, speaker)
            if os.path.isdir(speaker_path):
                for audio_file in os.listdir(speaker_path):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(speaker_path, audio_file)
                        feature_vector = self.feature_extractor.extract_features(audio_path)
                        
                        if feature_vector is not None:
                            features.append(feature_vector)
                            labels.append(speaker)
        
        return np.array(features), np.array(labels)