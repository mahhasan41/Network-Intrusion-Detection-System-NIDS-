"""
Evaluation and Explainability Module for Network Intrusion Detection System
Provides comprehensive evaluation metrics and model interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization
    """
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def plot_confusion_matrix(self, model, X_test, y_test, model_name):
        """
        Plot confusion matrix
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of model
        """
        print(f"Plotting confusion matrix for {model_name}...")
        
        # Predictions
        if model_name == "Autoencoder":
            # For Autoencoder, compute reconstruction error
            y_pred_reconstruction = model.predict(X_test, verbose=0)
            reconstruction_error = np.mean(np.abs(y_pred_reconstruction - X_test), axis=1)
            threshold = np.median(reconstruction_error)
            y_pred = (reconstruction_error > threshold).astype(int)
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                y_pred = np.zeros(len(X_test), dtype=int)
            else:
                y_pred = (proba[:, 1] > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            # Ensure binary
            if isinstance(y_pred, np.ndarray) and y_pred.dtype in [np.float32, np.float64]:
                y_pred = (y_pred > 0.5).astype(int)
        
        # Ensure y_pred is integer
        y_pred = np.asarray(y_pred).astype(int).flatten()
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {filepath}")
    
    def plot_roc_curve(self, model, X_test, y_test, model_name):
        """
        Plot ROC curve
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of model
        """
        print(f"Plotting ROC curve for {model_name}...")
        
        # Get probability predictions
        if model_name == "Autoencoder":
            # For Autoencoder, compute reconstruction error
            y_pred_reconstruction = model.predict(X_test, verbose=0)
            reconstruction_error = np.mean(np.abs(y_pred_reconstruction - X_test), axis=1)
            y_pred_proba = 1 - (reconstruction_error / (reconstruction_error.max() + 1e-8))
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                y_pred_proba = np.zeros(len(X_test))
            else:
                y_pred_proba = proba[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = model.predict(X_test)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim((0.0, 1.05))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {filepath}")
        
        return roc_auc
    
    def plot_precision_recall_curve(self, model, X_test, y_test, model_name):
        """
        Plot Precision-Recall curve
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of model
        """
        print(f"Plotting Precision-Recall curve for {model_name}...")
        
        # Get probability predictions
        if model_name == "Autoencoder":
            # For Autoencoder, compute reconstruction error
            y_pred_reconstruction = model.predict(X_test, verbose=0)
            reconstruction_error = np.mean(np.abs(y_pred_reconstruction - X_test), axis=1)
            y_pred_proba = 1 - (reconstruction_error / (reconstruction_error.max() + 1e-8))
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                y_pred_proba = np.zeros(len(X_test))
            else:
                y_pred_proba = proba[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = model.predict(X_test)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, f'precision_recall_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {filepath}")
        
        return avg_precision
    
    def plot_feature_importance(self, feature_importance_dict):
        """
        Plot feature importance from models
        
        Args:
            feature_importance_dict (dict): Feature importance from models
        """
        for model_name, importance_df in feature_importance_dict.items():
            print(f"Plotting feature importance for {model_name}...")
            
            # Get top 20 features
            top_features = importance_df.head(20)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features['Importance'].values)
            plt.yticks(range(len(top_features)), top_features['Feature'].values)
            plt.xlabel('Importance Score')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            
            filepath = os.path.join(self.results_dir, 
                                   f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved to {filepath}")
    
    def plot_model_comparison(self, results_df):
        """
        Plot comparison of all models
        
        Args:
            results_df (pd.DataFrame): Results from all models
        """
        print("Plotting model comparison...")
        
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(metrics))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for idx, (i, row) in enumerate(results_df.iterrows()):
            values = [row[metric] for metric in metrics]
            ax.bar(x + idx * width, values, width, label=row['Model'])
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim((0, 1.05))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {filepath}")
    
    def generate_classification_report(self, model, X_test, y_test, model_name):
        """
        Generate detailed classification report
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of model
        """
        print(f"Generating classification report for {model_name}...")
        
        # Predictions - ensure binary output
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                y_pred = np.zeros(len(X_test), dtype=int)
        # Predictions - ensure binary output
        if model_name == "Autoencoder":
            # For Autoencoder, compute reconstruction error
            y_pred_reconstruction = model.predict(X_test, verbose=0)
            reconstruction_error = np.mean(np.abs(y_pred_reconstruction - X_test), axis=1)
            threshold = np.median(reconstruction_error)
            y_pred = (reconstruction_error > threshold).astype(int)
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # Handle case with only 1 class
            if proba.shape[1] < 2:
                y_pred = np.zeros(len(X_test), dtype=int)
            else:
                y_pred = (proba[:, 1] > 0.5).astype(int)
        elif hasattr(model, 'decision_function'):
            y_pred = (model.decision_function(X_test) > 0).astype(int)
        else:
            raw_pred = model.predict(X_test)
            # Ensure binary classification
            if isinstance(raw_pred, np.ndarray):
                if raw_pred.dtype in [np.float32, np.float64]:
                    y_pred = (raw_pred > 0.5).astype(int).flatten()
                else:
                    y_pred = np.asarray(raw_pred).astype(int).flatten()
            else:
                y_pred = np.asarray(raw_pred).astype(int).flatten()
        
        # Ensure y_pred is 1D and contains only 0 and 1
        y_pred = np.asarray(y_pred).flatten()
        y_pred = np.clip(y_pred, 0, 1).astype(int)
        
        # Classification report - use labels parameter to handle missing classes
        report = classification_report(y_test, y_pred, 
                                      labels=[0, 1],
                                      target_names=['Normal', 'Attack'],
                                      zero_division=0)
        
        print(f"\nClassification Report - {model_name}")
        print("="*60)
        print(report)
        
        # Save report
        filepath = os.path.join(self.results_dir, 
                               f'classification_report_{model_name.lower().replace(" ", "_")}.txt')
        with open(filepath, 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("="*60 + "\n")
            f.write(str(report))  # type: ignore
        
        return report
    
    def evaluate_all_models(self, models_dict, X_test, y_test, 
                           feature_importance_dict, results_df):
        """
        Comprehensive evaluation of all models
        
        Args:
            models_dict (dict): Dictionary of trained models
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            feature_importance_dict (dict): Feature importance
            results_df (pd.DataFrame): Results DataFrame
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        for model_name, model in models_dict.items():
            print(f"\n--- {model_name} ---")
            self.plot_confusion_matrix(model, X_test, y_test, model_name)
            self.plot_roc_curve(model, X_test, y_test, model_name)
            self.plot_precision_recall_curve(model, X_test, y_test, model_name)
            self.generate_classification_report(model, X_test, y_test, model_name)
        
        # Plot feature importance
        if feature_importance_dict:
            self.plot_feature_importance(feature_importance_dict)
        
        # Plot model comparison
        self.plot_model_comparison(results_df)
        
        print("\n" + "="*60)
        print("All evaluation plots saved to 'results' directory")
        print("="*60)


class ExplainabilityModule:
    """
    Module for model explainability and interpretation
    """
    
    def __init__(self):
        pass
    
    def get_feature_contribution(self, model, X_sample, feature_names, top_k=10):
        """
        Get feature contribution for a sample prediction
        
        Args:
            model: Trained model
            X_sample (np.ndarray): Single sample
            feature_names (list): Feature names
            top_k (int): Top k features to display
            
        Returns:
            dict: Feature contributions
        """
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        
        # Normalize
        importances = importances / importances.sum()
        
        # Get top features
        top_indices = np.argsort(importances)[-top_k:][::-1]
        
        contributions = {
            'features': [feature_names[i] for i in top_indices],
            'importance': importances[top_indices],
            'values': X_sample[top_indices] if X_sample.ndim == 1 else X_sample[0, top_indices]
        }
        
        return contributions
    
    def analyze_attack_type(self, predictions, confidence_scores):
        """
        Analyze attack type and confidence
        
        Args:
            predictions (np.ndarray): Model predictions
            confidence_scores (np.ndarray): Confidence scores
            
        Returns:
            dict: Analysis results
        """
        analysis = {
            'total_samples': len(predictions),
            'normal_count': sum(predictions == 0),
            'attack_count': sum(predictions == 1),
            'avg_confidence': np.mean(confidence_scores),
            'high_confidence_attacks': sum((predictions == 1) & (confidence_scores > 0.9))
        }
        
        return analysis
