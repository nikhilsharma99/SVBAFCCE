from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

class ModelTrainer:
    def __init__(self, n_jobs=-1, n_estimators=100):
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.results = {}
    
    def train(self, features, labels):
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\nTraining and evaluating models...")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            self.results[name] = {
                'predictions': y_pred,
                'true_labels': y_test,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"\n{name} Performance:")
            print(classification_report(y_test, y_pred))
        
        self.visualize_comparison()
        return X_test_scaled, y_test
    
    def visualize_comparison(self):
        # Create accuracy comparison plot
        plt.figure(figsize=(10, 6))
        accuracies = [self.results[name]['report']['accuracy'] 
                     for name in self.models.keys()]
        
        plt.bar(self.models.keys(), accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0.8, 1.0)  # Assuming high accuracies
        
        # Add value labels on top of each bar
        for i, v in enumerate(accuracies):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('models/accuracy_comparison.png')
        plt.close()
        
        # Create confusion matrices
        for name, result in self.results.items():
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'models/confusion_matrix_{name}.png')
            plt.close()
    
    def save_model(self, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            with open(os.path.join(save_dir, f'{name.lower()}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Save label encoder and scaler
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)