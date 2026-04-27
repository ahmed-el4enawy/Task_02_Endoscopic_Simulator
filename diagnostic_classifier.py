"""
AI Diagnostic Classifier for Electronic Spirometry
Trains a Random Forest model to classify respiratory patterns
(Normal, Obstructive, Restrictive) based on % Predicted FVC and FEV1/FVC ratio.
Features persistent model saving/loading (.pkl).
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class DiagnosticClassifier:
    def __init__(self, model_path='diagnostic_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes = ['Normal', 'Obstructive (COPD/Asthma)', 'Restrictive']

        # Automatically load the model if it exists to save startup time
        self.load_model()

    def generate_synthetic_data(self, n_samples=2000):
        """Generates synthetic spirometry data based on ATS/ERS clinical guidelines using % Predicted"""
        np.random.seed(42)
        X = []
        y = []

        # 1. Normal (Ratio >= 70%, FVC >= 80% Predicted)
        for _ in range(n_samples // 3):
            pct_fvc = np.random.uniform(80.0, 120.0)
            ratio = np.random.uniform(70.0, 85.0)
            pct_fev1 = pct_fvc * (ratio / 100.0) / 0.8  # Rough approximation for features
            X.append([pct_fvc, pct_fev1, ratio])
            y.append(0) # Normal

        # 2. Obstructive (Ratio < 70%, FVC can be anything, usually normal or slightly low)
        for _ in range(n_samples // 3):
            pct_fvc = np.random.uniform(60.0, 110.0)
            ratio = np.random.uniform(40.0, 69.9)
            pct_fev1 = pct_fvc * (ratio / 100.0) / 0.8
            X.append([pct_fvc, pct_fev1, ratio])
            y.append(1) # Obstructive

        # 3. Restrictive (Ratio >= 70%, Low FVC < 80% predicted)
        for _ in range(n_samples // 3):
            pct_fvc = np.random.uniform(40.0, 79.9)
            ratio = np.random.uniform(70.0, 90.0)
            pct_fev1 = pct_fvc * (ratio / 100.0) / 0.8
            X.append([pct_fvc, pct_fev1, ratio])
            y.append(2) # Restrictive

        return np.array(X), np.array(y)

    def train(self):
        """Train the Random Forest model and save it"""
        print("[AI] Generating clinical spirometry training data...")
        X, y = self.generate_synthetic_data()

        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("[AI] Diagnostic model trained successfully.")

        self.save_model()

    def save_model(self):
        """Persist the trained model and scaler to disk"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"[AI] Model saved to {self.model_path}")

    def load_model(self):
        """Load the model from disk if available"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
            print(f"[AI] Pre-trained model loaded from {self.model_path}")

    def predict(self, pct_fvc, pct_fev1, ratio):
        """Predict the clinical diagnosis based on % Predicted metrics"""
        if not self.is_trained:
            self.train()

        X_new = np.array([[pct_fvc, pct_fev1, ratio]])
        X_scaled = self.scaler.transform(X_new)

        prediction_idx = self.model.predict(X_scaled)[0]
        confidence = np.max(self.model.predict_proba(X_scaled)[0])

        return self.classes[prediction_idx], confidence