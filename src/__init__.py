"""
Network Intrusion Detection System
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "ML Engineer"
__description__ = "Production-ready Network Intrusion Detection System"

# Lazy imports - only import when needed
def get_preprocessor():
    from src.preprocessing import DataPreprocessor
    return DataPreprocessor

def get_trainer():
    from src.train import ModelTrainer
    return ModelTrainer

def get_evaluator():
    from src.explain import ModelEvaluator, ExplainabilityModule
    return ModelEvaluator, ExplainabilityModule

def get_predictor():
    from src.predict import PredictionEngine
    return PredictionEngine

__all__ = [
    'DataPreprocessor',
    'ModelTrainer',
    'ModelEvaluator',
    'ExplainabilityModule',
    'PredictionEngine'
]
