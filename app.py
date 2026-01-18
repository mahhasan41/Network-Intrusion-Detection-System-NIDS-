"""
Flask Web Dashboard for Network Intrusion Detection System
Provides interactive interface for model predictions and monitoring
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
import zipfile
import gzip
import shutil
from src.predict import PredictionEngine
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configurable max upload size (defaults to 256MB). Set env MAX_UPLOAD_MB to override.
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '1024'))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize prediction engine
predictor = None

# Try to load models on startup
def load_default_model():
    global predictor
    try:
        model_path = 'models/intrusion_detector_random_forest_model.pkl'
        scaler_path = 'models/intrusion_detector_random_forest_scaler.pkl'
        features_path = 'models/intrusion_detector_random_forest_features.pkl'
        
        if os.path.exists(model_path):
            predictor = PredictionEngine(model_path, scaler_path, features_path)
            print("✓ Model loaded on startup")
        else:
            print("⚠ Model files not found. Use /api/load-model to load models.")
    except Exception as e:
        print(f"⚠ Could not load model on startup: {e}")

# Load model on startup
load_default_model()


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file_error(e):
    """Friendly JSON for 413 errors with guidance."""
    return jsonify({
        'error': 'File too large',
        'detail': 'The uploaded file exceeds the server limit.',
        'max_upload_mb': MAX_UPLOAD_MB,
        'tip': 'For very large CSVs, either increase MAX_UPLOAD_MB env var and restart the app, or use the batch_predict.py CLI.'
    }), 413


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/predict/single', methods=['POST'])
def predict_single():
    """
    API endpoint for single sample prediction
    """
    try:
        data = request.get_json()
        
        if not predictor:
            return jsonify({'error': 'Model not loaded'}), 400
        
        # Convert to array
        X = np.array([list(data.values())])
        
        # Predict
        result = predictor.predict(X)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': float(result['confidence']),
            'is_attack': result['is_attack']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    API endpoint for batch prediction from CSV
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Allow CSV and compressed CSV (.zip or .gz)
        allowed_ext = ('.csv', '.zip', '.gz')
        if not any(file.filename.lower().endswith(ext) for ext in allowed_ext):  # type: ignore
            return jsonify({'error': 'Only CSV, ZIP, or GZ files allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename or 'upload.csv')  # type: ignore
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # If compressed, extract/decompress to a CSV
        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zf:
                # Find first CSV inside the zip
                csv_members = [m for m in zf.namelist() if m.lower().endswith('.csv')]
                if not csv_members:
                    return jsonify({'error': 'ZIP file does not contain any CSV'}), 400
                member = csv_members[0]
                extracted_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(os.path.basename(member)))
                zf.extract(member, app.config['UPLOAD_FOLDER'])
                # Move to a clean path
                if os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], member)) != extracted_path:
                    shutil.move(os.path.join(app.config['UPLOAD_FOLDER'], member), extracted_path)
                filepath = extracted_path
        elif filename.lower().endswith('.gz'):
            extracted_path = os.path.join(app.config['UPLOAD_FOLDER'], filename[:-3] + '.csv')
            with gzip.open(filepath, 'rb') as f_in, open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            filepath = extracted_path
        
        # Make predictions on the extracted or original CSV
        results = predictor.predict_from_csv(filepath)  # type: ignore
        
        # Generate statistics
        stats = {
            'total_samples': len(results),
            'normal_count': int((results['Is_Attack'] == False).sum()),
            'attack_count': int((results['Is_Attack'] == True).sum()),
            'avg_confidence': float(results['Confidence'].mean()),
            'high_confidence_attacks': int(((results['Is_Attack'] == True) & 
                                           (results['Confidence'] > 0.8)).sum())
        }
        
        # Convert results to JSON
        results_json = results.astype(str).to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'results': results_json[:100]  # Return first 100 results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict/batch/local', methods=['POST'])
def predict_batch_local():
    """
    API endpoint for batch prediction from a local server file path.
    Useful for very large CSVs to avoid HTTP upload limits.
    Body: {"path": "<absolute-or-relative path>", "limit": <int optional>}
    """
    try:
        if not predictor:
            return jsonify({'error': 'Model not loaded'}), 400

        data = request.get_json(force=True, silent=True) or {}
        path = data.get('path')
        limit = data.get('limit')

        if not path:
            return jsonify({'error': 'Missing path in JSON body'}), 400

        # Resolve and restrict to current workspace for safety
        abs_path = os.path.abspath(path)
        workspace_root = os.path.abspath(os.getcwd())
        if not abs_path.startswith(workspace_root):
            return jsonify({'error': 'Path must be inside the server workspace'}), 400

        if not os.path.exists(abs_path):
            return jsonify({'error': f'File not found: {abs_path}'}), 404
        if not abs_path.endswith('.csv'):
            return jsonify({'error': 'Only CSV files allowed'}), 400

        # Make predictions
        results = predictor.predict_from_csv(abs_path)  # type: ignore

        if isinstance(limit, int) and limit > 0:
            results = results.head(limit)

        # Generate statistics
        stats = {
            'total_samples': len(results),
            'normal_count': int((results['Is_Attack'] == False).sum()),
            'attack_count': int((results['Is_Attack'] == True).sum()),
            'avg_confidence': float(results['Confidence'].mean()),
            'high_confidence_attacks': int(((results['Is_Attack'] == True) & 
                                           (results['Confidence'] > 0.8)).sum())
        }

        # Convert results to JSON (limit response size)
        results_json = results.astype(str).to_dict(orient='records')

        return jsonify({
            'success': True,
            'statistics': stats,
            'results': results_json[:100]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get overall system statistics
    """
    try:
        return jsonify({
            'success': True,
            'model_loaded': predictor is not None,
            'total_predictions': 0,
            'overall_accuracy': 0.95,
            'avg_detection_time': 0.005
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/load-model', methods=['POST'])
def load_model():
    """
    Load model endpoint
    """
    global predictor
    
    try:
        data = request.get_json()
        model_path = data.get('model_path', 'models/intrusion_detector_model.pkl')
        scaler_path = data.get('scaler_path', 'models/intrusion_detector_scaler.pkl')
        features_path = data.get('features_path', 'models/intrusion_detector_features.pkl')
        
        predictor = PredictionEngine(model_path, scaler_path, features_path)
        
        return jsonify({'success': True, 'message': 'Model loaded successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Get model information
    """
    if not predictor or not predictor.model:
        return jsonify({'loaded': False})
    
    model_info = {
        'loaded': True,
        'model_type': type(predictor.model).__name__,
        'feature_count': len(predictor.feature_names) if predictor.feature_names else 'Unknown'
    }
    
    return jsonify(model_info)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


def create_app():
    """Create Flask application"""
    return app


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
