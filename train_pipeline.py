"""
Main Training Script for Network Intrusion Detection System
End-to-end pipeline for data preprocessing, model training, and evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.explain import ModelEvaluator, ExplainabilityModule
from src.predict import PredictionEngine


def main(sample_size=None):
    """
    Main pipeline for NIDS training
    
    Args:
        sample_size (int): Number of samples to use (None = all). 
                          Use for quick testing: e.g., 5000
    """
    
    print("\n" + "="*70)
    print("NETWORK INTRUSION DETECTION SYSTEM - TRAINING PIPELINE")
    print("="*70)
    
    if sample_size:
        print(f"⚠ Running on {sample_size:,} samples (quick test mode)")
    
    # Configuration
    dataset_path = 'combinenew.csv'
    results_dir = 'results'
    models_dir = 'models'
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure the CSV file is in the project root directory")
        return
    
    # Create necessary directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n[1/5] DATA PREPROCESSING")
    print("-" * 70)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=42)
    
    # Prepare data
    data_dict = preprocessor.prepare_data(
        filepath=dataset_path,
        test_size=0.2,
        apply_smote=True,
        feature_selection=True,
        k_features=50,
        nrows=sample_size
    )
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_names = data_dict['feature_names']
    scaler = data_dict['scaler']
    
    print("\n[2/5] MODEL TRAINING")
    print("-" * 70)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train all models
    training_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    results_df = training_results['results_df']
    models_dict = training_results['models']
    best_model = training_results['best_model']
    best_model_name = training_results['best_model_name']
    
    # Save results
    results_df.to_csv(os.path.join(results_dir, 'model_results.csv'), index=False)
    
    print("\n[3/5] MODEL EVALUATION & VISUALIZATION")
    print("-" * 70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(results_dir=results_dir)
    
    # Get feature importance
    feature_importance_dict = trainer.get_feature_importance(feature_names)
    
    # Comprehensive evaluation
    evaluator.evaluate_all_models(
        models_dict=models_dict,
        X_test=X_test,
        y_test=y_test,
        feature_importance_dict=feature_importance_dict,
        results_df=results_df
    )
    
    print("\n[4/5] SAVE BEST MODEL")
    print("-" * 70)
    
    # Initialize prediction engine
    predictor = PredictionEngine()
    
    # Save best model
    model_path, scaler_path, features_path = predictor.save_model(
        model=best_model,
        scaler=scaler,
        feature_names=feature_names,
        model_name=f"intrusion_detector_{best_model_name.lower().replace(' ', '_')}"
    )
    
    print("\n[5/5] TESTING REAL-TIME PREDICTION")
    print("-" * 70)
    
    # Load the saved model for testing
    predictor.load_model(model_path, scaler_path, features_path)
    
    # Test on a few samples
    print("\nTesting predictions on sample data...")
    predictions = predictor.predict_batch(X_test[:10])
    print("\nSample Predictions:")
    print(predictions.head(10))
    
    # Save sample predictions
    predictions.to_csv(os.path.join(results_dir, 'sample_predictions.csv'), index=False)
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  • Best Model: {best_model_name}")
    print(f"  • Accuracy: {results_df.loc[results_df['Model'] == best_model_name, 'Accuracy'].values[0]:.4f}")
    print(f"  • F1-Score: {results_df.loc[results_df['Model'] == best_model_name, 'F1-Score'].values[0]:.4f}")
    print(f"  • Recall: {results_df.loc[results_df['Model'] == best_model_name, 'Recall'].values[0]:.4f}")
    print(f"  • ROC-AUC: {results_df.loc[results_df['Model'] == best_model_name, 'ROC-AUC'].values[0]:.4f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Results: {results_dir}/")
    print(f"  • Models: {models_dir}/")
    print(f"  • Visualizations: PNG files in results/")
    
    print(f"\n🚀 TO RUN THE WEB DASHBOARD:")
    print(f"  python app.py")
    
    print(f"\n📈 RESULTS SAVED TO: {results_dir}/")
    print("\nTraining complete!")


if __name__ == "__main__":
    import sys
    
    # Check if sample size argument is provided
    sample_size = None
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid sample size: {sys.argv[1]}")
            sys.exit(1)
    
    main(sample_size=sample_size)
