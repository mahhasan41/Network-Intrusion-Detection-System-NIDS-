"""
Model Training Module for Network Intrusion Detection System
Trains and compares multiple ML models for intrusion detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow (optional)
try:
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, Model  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ModelTrainer:
    """
    Train and manage multiple ML models for intrusion detection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """
        Train Random Forest classifier
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            n_estimators (int): Number of trees
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        print("✓ Random Forest training complete")
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators=100):
        """
        Train XGBoost classifier
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            n_estimators (int): Number of boosting rounds
            
        Returns:
            xgb.XGBClassifier: Trained model
        """
        print("Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        print("✓ XGBoost training complete")
        return model
    
    def train_isolation_forest(self, X_train, contamination=0.1):
        """
        Train Isolation Forest for anomaly detection
        
        Args:
            X_train (np.ndarray): Training features
            contamination (float): Expected proportion of outliers
            
        Returns:
            IsolationForest: Trained model
        """
        print("Training Isolation Forest...")
        model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train)
        print("✓ Isolation Forest training complete")
        return model
    
    def build_autoencoder(self, X_train, encoding_dim=32, epochs=50):
        """
        Build and train an Autoencoder for anomaly detection
        
        Args:
            X_train (np.ndarray): Training features
            encoding_dim (int): Dimension of encoding layer
            epochs (int): Number of training epochs
            
        Returns:
            Model: Trained autoencoder model or None if TensorFlow unavailable
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠ TensorFlow not available - skipping Autoencoder")
            return None
            
        print("Building and training Autoencoder...")
        
        input_dim = X_train.shape[1]
        
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train
        autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        print("✓ Autoencoder training complete")
        return autoencoder
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        if model_name == "Autoencoder":
            # For Autoencoder, compute reconstruction error
            y_pred_reconstruction = model.predict(X_test, verbose=0)
            reconstruction_error = np.mean(np.abs(y_pred_reconstruction - X_test), axis=1)
            # Use median error as threshold
            threshold = np.median(reconstruction_error)
            y_pred_proba = 1 - (reconstruction_error / (reconstruction_error.max() + 1e-8))
            y_pred = (reconstruction_error > threshold).astype(int)
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                print(f"⚠ Warning: Only 1 class in training data. All predictions will be class 0.")
                y_pred_proba = np.zeros(len(X_test))
                y_pred = np.zeros(len(X_test), dtype=int)
            else:
                y_pred_proba = proba[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.0
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Results and trained models
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        results_list = []
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        self.models['Random Forest'] = rf_model
        rf_results = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        results_list.append(rf_results)
        
        # Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        xgb_results = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results_list.append(xgb_results)
        
        # Train Isolation Forest
        if_model = self.train_isolation_forest(X_train)
        self.models['Isolation Forest'] = if_model
        if_results = self.evaluate_model(if_model, X_test, y_test, "Isolation Forest")
        results_list.append(if_results)
        
        # Train Autoencoder (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            ae_model = self.build_autoencoder(X_train)
            self.models['Autoencoder'] = ae_model
            ae_results = self.evaluate_model(ae_model, X_test, y_test, "Autoencoder")
            results_list.append(ae_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        self.results = results_df
        
        # Select best model based on F1-Score and Recall
        self.results['Score'] = (self.results['F1-Score'] * 0.6 + 
                                  self.results['Recall'] * 0.4)
        best_idx = self.results['Score'].idxmax()
        self.best_model_name = self.results.iloc[best_idx]['Model']  # type: ignore
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(self.results.to_string(index=False))
        print(f"\n✓ Best Model: {self.best_model_name}")
        print("="*60)
        
        return {
            'results_df': results_df,
            'models': self.models,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name
        }
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from tree-based models
        
        Args:
            feature_names (list): Names of features
            
        Returns:
            dict: Feature importance for each model
        """
        importance_dict = {}
        
        if 'Random Forest' in self.models:
            rf_importance = self.models['Random Forest'].feature_importances_
            importance_dict['Random Forest'] = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_importance
            }).sort_values('Importance', ascending=False)
        
        if 'XGBoost' in self.models:
            xgb_importance = self.models['XGBoost'].feature_importances_
            importance_dict['XGBoost'] = pd.DataFrame({
                'Feature': feature_names,
                'Importance': xgb_importance
            }).sort_values('Importance', ascending=False)
        
        return importance_dict
