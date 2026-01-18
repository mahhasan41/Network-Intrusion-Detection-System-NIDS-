"""
Real-Time Prediction Module for Network Intrusion Detection System
Handles inference on new network traffic data
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PredictionEngine:
    """
    Real-time prediction engine for network intrusion detection
    """
    
    def __init__(self, model_path=None, scaler_path=None, feature_names_path=None):
        """
        Initialize prediction engine
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
            feature_names_path (str): Path to saved feature names
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.models_dir = 'models'
        
        if model_path:
            self.load_model(model_path, scaler_path, feature_names_path)
    
    def load_model(self, model_path, scaler_path=None, feature_names_path=None):
        """
        Load trained model and preprocessing objects
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
            feature_names_path (str): Path to saved feature names
        """
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        if feature_names_path:
            with open(feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
        
        print("✓ Model loaded successfully")
    
    def save_model(self, model, scaler=None, feature_names=None, model_name="intrusion_detector"):
        """
        Save trained model and preprocessing objects
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_names: Feature names
            model_name (str): Name for the saved model
        """
        os.makedirs(self.models_dir, exist_ok=True)
        
        model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
        scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
        features_path = os.path.join(self.models_dir, f"{model_name}_features.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        if scaler is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        if feature_names is not None:
            with open(features_path, 'wb') as f:
                pickle.dump(feature_names, f)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
        print(f"✓ Features saved: {features_path}")
        
        return model_path, scaler_path, features_path
    
    def preprocess_single_sample(self, X_sample):
        """
        Preprocess a single sample for prediction
        
        Args:
            X_sample (np.ndarray or pd.Series): Single traffic sample
            
        Returns:
            np.ndarray: Preprocessed sample
        """
        if isinstance(X_sample, pd.Series):
            X_sample = X_sample.values
        
        # Ensure correct shape
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)  # type: ignore
        
        # Pad with zeros if fewer features than expected (50)
        if X_sample.shape[1] < 50:
            padding = np.zeros((X_sample.shape[0], 50 - X_sample.shape[1]))
            X_sample = np.hstack([X_sample, padding])
        elif X_sample.shape[1] > 50:
            # Truncate if more than 50 features
            X_sample = X_sample[:, :50]
        
        # Scale
        if self.scaler is not None:
            X_sample = self.scaler.transform(X_sample)
        
        return X_sample
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X (np.ndarray or pd.DataFrame): Features for prediction
            
        Returns:
            dict: Predictions and confidence scores
        """
        # Preprocess
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_processed = self.preprocess_single_sample(X)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_processed)  # type: ignore
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                y_pred = 0
                confidence = 1.0
            else:
                confidence = np.max(proba, axis=1)[0]
                y_pred = np.argmax(proba, axis=1)[0]
        else:
            y_pred = self.model.predict(X_processed)[0]  # type: ignore
            confidence = np.abs(y_pred)
        
        # Classify
        is_attack = bool(y_pred == 1)
        
        return {
            'prediction': 'Attack' if is_attack else 'Normal',
            'is_attack': is_attack,
            'confidence': float(confidence),
            'probability': float(confidence)
        }
    
    def predict_batch(self, X_batch):
        """
        Make predictions on batch of data
        
        Args:
            X_batch (np.ndarray or pd.DataFrame): Batch of traffic samples
            
        Returns:
            pd.DataFrame: Predictions for all samples
        """
        print(f"Making predictions on {len(X_batch)} samples...")
        
        if isinstance(X_batch, pd.DataFrame):
            X = X_batch.values
        else:
            X = X_batch
        
        # Preprocess batch
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)  # type: ignore
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                y_pred = np.zeros(len(X), dtype=int)
                confidence = np.ones(len(X))
            else:
                y_pred = (proba[:, 1] > 0.5).astype(int)
                confidence = np.max(proba, axis=1)
        else:
            y_pred = self.model.predict(X)
            confidence = np.ones(len(X)) * 0.5
        
        # Create results dataframe
        results = pd.DataFrame({
            'Prediction': ['Attack' if pred == 1 else 'Normal' for pred in y_pred],
            'Confidence': confidence,
            'Is_Attack': y_pred == 1,
            'Timestamp': [datetime.now()] * len(X)
        })
        
        print(f"✓ Predictions complete")
        print(f"  Normal: {(y_pred == 0).sum()}")
        print(f"  Attacks: {(y_pred == 1).sum()}")
        
        return results
    
    def predict_from_csv(self, filepath):
        """
        Make predictions on data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Predictions
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Remove label if present
        if 'Label' in df.columns:
            df = df.drop('Label', axis=1)
        
        # Align to required features: add missing columns with zeros and reorder
        if self.feature_names:
            missing = [col for col in self.feature_names if col not in df.columns]
            if missing:
                # Add missing columns filled with zeros
                for col in missing:
                    df[col] = 0
            # Subset and reorder to match training feature order
            df = df[[col for col in self.feature_names if col in df.columns]]
            # Ensure final column count matches expected by padding if any were still missing
            if len(df.columns) < len(self.feature_names):
                # Create any still-missing columns that may not have been caught
                for col in self.feature_names:
                    if col not in df.columns:
                        df[col] = 0
                df = df[self.feature_names]
        
        return self.predict_batch(df)
    
    def predict_from_dict(self, data_dict):
        """
        Make prediction from dictionary of feature values
        
        Args:
            data_dict (dict): Dictionary with feature names and values
            
        Returns:
            dict: Prediction result
        """
        # Create dataframe from dict
        df = pd.DataFrame([data_dict])
        
        # Select only required features
        if self.feature_names:
            df = df[self.feature_names]
        
        return self.predict(df.values)
    
    def explain_prediction(self, X, feature_names=None, top_k=10):
        """
        Explain individual prediction using feature importance
        
        Args:
            X (np.ndarray): Single sample
            feature_names (list): Feature names
            top_k (int): Top k features to show
            
        Returns:
            dict: Explanation
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_  # type: ignore
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        sample = X[0]
        
        # Get top important features
        top_indices = np.argsort(importances)[-top_k:][::-1]
        
        explanation = {
            'top_features': [feature_names[i] if feature_names else f'Feature_{i}' 
                            for i in top_indices],
            'importances': importances[top_indices],
            'values': sample[top_indices]
        }
        
        return explanation
    
    def detect_anomalies(self, X_batch, threshold=0.7):
        """
        Detect anomalies in batch data
        
        Args:
            X_batch (np.ndarray): Batch of samples
            threshold (float): Confidence threshold
            
        Returns:
            pd.DataFrame: Anomaly detection results
        """
        results = self.predict_batch(X_batch)
        results['Is_Anomaly'] = results['Confidence'] > threshold
        
        return results
    
    def generate_alert(self, prediction_result, alert_threshold=0.8):
        """
        Generate alert for high-confidence attacks
        
        Args:
            prediction_result (dict): Prediction result
            alert_threshold (float): Confidence threshold for alerts
            
        Returns:
            dict: Alert information
        """
        alert = {
            'timestamp': datetime.now(),
            'triggered': False,
            'severity': 'Low',
            'message': ''
        }
        
        if prediction_result['is_attack']:
            if prediction_result['confidence'] > alert_threshold:
                alert['triggered'] = True
                alert['severity'] = 'Critical'
                alert['message'] = f"CRITICAL: High-confidence attack detected ({prediction_result['confidence']:.2%})"
            elif prediction_result['confidence'] > 0.6:
                alert['triggered'] = True
                alert['severity'] = 'Medium'
                alert['message'] = f"WARNING: Potential attack detected ({prediction_result['confidence']:.2%})"
        
        return alert
