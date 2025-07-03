import os
import sys
import multiprocessing
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_processing import DataProcessor
from model_comparison import ModelComparison

from data_processing import DataProcessor
from model_comparison import ModelComparison

def main():
    # Configure paths
    dataset_path = os.path.join('data', 'swift-voice-biometricsdataspeaker-recognition-audio-dataset')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        return
    
    # Use optimal number of processes
    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} CPU cores for parallel processing")
    
    # Process dataset
    print("Processing audio files...")
    processor = DataProcessor(dataset_path, n_jobs=n_jobs)
    features, labels = processor.process_dataset()
    
    if len(features) == 0:
        print("No features extracted. Exiting...")
        return
    
    print(f"\nDataset processed successfully:")
    print(f"Total samples: {len(features)}")
    print(f"Unique speakers: {len(set(labels))}")
    
    # Train and compare models
    print("\nTraining and comparing models...")
    trainer = ModelComparison(n_jobs=n_jobs)
    trainer.train_and_evaluate(features, labels)
    trainer.save_models()
    
    print("\nModels saved successfully!")
    print("Model comparison plot saved as 'models/accuracy_comparison.png'")
    print("Individual confusion matrices saved in 'models/' directory")

if __name__ == "__main__":
    main()