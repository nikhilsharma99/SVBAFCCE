import os
import pickle
import numpy as np
from .feature_extraction import FeatureExtractor

class VoiceBiometricPredictor:
    def __init__(self, model_dir='models'):
        # Load the trained model and label encoder
        with open(os.path.join(model_dir, 'voice_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.feature_extractor = FeatureExtractor()
    
    def predict(self, audio_path):
        # Extract features from the audio file
        features = self.feature_extractor.extract_features(audio_path)
        if features is None:
            return None
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Predict speaker
        prediction = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Get speaker name and confidence
        speaker = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities) * 100
        
        return speaker, confidence

def main():
    predictor = VoiceBiometricPredictor()
    
    # Test directory path
    test_dir = os.path.join('data', 'swift-voice-biometricsdataspeaker-recognition-audio-dataset')
    
    print("Voice Biometric Authentication System")
    print("====================================")
    
    while True:
        print("\nOptions:")
        print("1. Predict from file path")
        print("2. Test with sample from dataset")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            audio_path = input("\nEnter the path to the audio file (.wav): ")
            if not os.path.exists(audio_path):
                print("File not found!")
                continue
                
            result = predictor.predict(audio_path)
            if result:
                speaker, confidence = result
                print(f"\nPredicted Speaker: {speaker}")
                print(f"Confidence: {confidence:.2f}%")
            else:
                print("Error processing audio file")
                
        elif choice == '2':
            # Get a random sample from the dataset
            speakers = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            speaker = np.random.choice(speakers)
            speaker_dir = os.path.join(test_dir, speaker)
            audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
            test_file = np.random.choice(audio_files)
            test_path = os.path.join(speaker_dir, test_file)
            
            print(f"\nTesting with file: {test_file}")
            print(f"Actual Speaker: {speaker}")
            
            result = predictor.predict(test_path)
            if result:
                predicted_speaker, confidence = result
                print(f"Predicted Speaker: {predicted_speaker}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"Correct: {'✓' if predicted_speaker == speaker else '✗'}")
            else:
                print("Error processing audio file")
                
        elif choice == '3':
            print("\nThank you for using the Voice Biometric System!")
            break
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()
