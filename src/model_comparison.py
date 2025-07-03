from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

class ModelComparison:
    def __init__(self, n_jobs=-1):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, n_jobs=n_jobs, random_state=42)
        }
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.accuracies = {}
    
    def train_and_evaluate(self, features, labels):
        encoded_labels = self.label_encoder.fit_transform(labels)
        scaled_features = self.scaler.fit_transform(features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, encoded_labels, test_size=0.2, random_state=42
        )
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracies[name] = accuracy * 100
            
            print(f"\n{name} Performance:")
            print(classification_report(y_test, y_pred))
              # Create confusion matrix
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'models/confusion_matrix_{name}.png')
            plt.close()
        
        self.plot_comparison()
        return X_test, y_test
    
    def plot_comparison(self):
        plt.figure(figsize=(12, 6))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        
        bars = plt.bar(self.accuracies.keys(), self.accuracies.values(), color=colors)
        
        plt.title('Model Accuracy Comparison', fontsize=14, pad=20)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('models/accuracy_comparison.png')
        plt.close()
    
    def save_models(self, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            with open(os.path.join(save_dir, f'{name.lower()}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)