# Quick Start Guide - Network Intrusion Detection System

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager
- 10GB free disk space
- 4GB RAM minimum

## 🚀 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_pipeline.py
```
⏱️ **Expected Time**: 30-60 minutes (first run)

**What happens:**
- ✅ Loads 830MB dataset
- ✅ Cleans and preprocesses data
- ✅ Trains 4 different ML models
- ✅ Evaluates model performance
- ✅ Saves best model to `models/` folder
- ✅ Generates evaluation plots

### Step 3: Run Web Dashboard
```bash
python app.py
```

Then open: **http://localhost:5000**

## 🎯 What to Try First

### Option A: Single Prediction
1. Go to "Single Prediction" tab
2. Enter sample values:
   - Destination Port: 80
   - Flow Duration: 1000
   - Total Packets: 50
3. Click "Predict"
4. See result: "Normal" or "Attack"

### Option B: Batch Prediction
1. Go to "Batch Prediction" tab
2. Upload a CSV file with network traffic data
3. Click "Upload & Predict"
4. View statistics and results

### Option C: View Results
1. Check `results/` folder for:
   - `model_results.csv` - All model metrics
   - `confusion_matrix_*.png` - Accuracy visualization
   - `roc_curve_*.png` - ROC curves
   - `feature_importance_*.png` - Top features

## 💻 Python API Usage

### Make a Single Prediction
```python
from src.predict import PredictionEngine
import numpy as np

# Load model
predictor = PredictionEngine(
    'models/intrusion_detector_model.pkl',
    'models/intrusion_detector_scaler.pkl',
    'models/intrusion_detector_features.pkl'
)

# Create sample traffic data (50 features)
traffic = np.random.randn(1, 50)

# Predict
result = predictor.predict(traffic)
print(f"Result: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing
```python
# Process CSV file
results = predictor.predict_from_csv('traffic_data.csv')
print(results)

# Save predictions
results.to_csv('predictions.csv')
```

## 🔍 Understanding Results

### Metrics Explained

| Metric | Meaning | Target |
|--------|---------|--------|
| Accuracy | How often model is correct | > 95% |
| Precision | False alarms rate | > 94% |
| Recall | Attack detection rate | > 90% |
| F1-Score | Balance between precision & recall | > 92% |
| ROC-AUC | Discrimination ability | > 0.98 |

### Confusion Matrix
```
                 Predicted
                Normal  Attack
Actual Normal      TN     FP
       Attack      FN     TP

TN = True Negative (Correct: Normal traffic)
FP = False Positive (Wrong: Normal flagged as attack)
FN = False Negative (Wrong: Attack missed)
TP = True Positive (Correct: Attack detected)
```

## ⚙️ Configuration

### Change Dataset
Edit `train_pipeline.py`:
```python
dataset_path = 'your_dataset.csv'  # Change this
```

### Adjust Feature Count
Edit `train_pipeline.py`:
```python
data_dict = preprocessor.prepare_data(
    filepath=dataset_path,
    k_features=30  # Was 50, now 30 features
)
```

### Change Train/Test Split
Edit `src/preprocessing.py`:
```python
train_test_split(
    X_normalized, y,
    test_size=0.3  # Was 0.2 (20%), now 0.3 (30%)
)
```

## 📊 Output Files

After training, you'll find:

```
results/
├── model_results.csv              # Model performance comparison
├── model_comparison.png            # Chart comparing all models
├── confusion_matrix_xgboost.png    # Confusion matrix
├── roc_curve_xgboost.png          # ROC curve
├── feature_importance_xgboost.png # Top features
└── classification_report_*.txt     # Detailed metrics

models/
├── intrusion_detector_model.pkl    # Trained model
├── intrusion_detector_scaler.pkl   # Data normalizer
└── intrusion_detector_features.pkl # Feature names
```

## 🆘 Troubleshooting

### Problem: "Module not found" error
**Solution**: Make sure you're in the project directory
```bash
cd Network-Intrusion-Detection-System
```

### Problem: "Dataset not found"
**Solution**: Ensure `combinenew.csv` is in the project root directory

### Problem: "Out of memory" error
**Solution**: Reduce features or use smaller batch size
```python
k_features=30  # Reduce from 50
```

### Problem: Flask server won't start
**Solution**: Check if port 5000 is in use
```bash
python app.py --port 5001  # Use different port
```

## 🎓 Learning Path

### Beginner
1. Run `train_pipeline.py`
2. View results in web dashboard
3. Make single predictions

### Intermediate
4. Batch process CSV files
5. Analyze feature importance
6. Modify hyperparameters

### Advanced
7. Implement custom models
8. Integrate with your system
9. Deploy to production

## 📚 File Structure Explained

```
preprocessing.py
├── load_data()           # Read CSV
├── explore_data()        # Statistics
├── handle_missing_values()
├── normalize_features()  # Standardization
└── prepare_data()        # Complete pipeline

train.py
├── train_random_forest()
├── train_xgboost()
├── train_isolation_forest()
├── build_autoencoder()
└── evaluate_model()

explain.py
├── plot_confusion_matrix()
├── plot_roc_curve()
├── plot_feature_importance()
└── generate_classification_report()

predict.py
├── load_model()
├── predict()             # Single sample
├── predict_batch()       # Multiple samples
└── generate_alert()      # Alert system
```

## 🔗 Useful Links

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CIC-IDS 2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📞 Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (30-60 min)
python train_pipeline.py

# Start web app
python app.py

# Use in Python
from src.predict import PredictionEngine
predictor = PredictionEngine(...)
result = predictor.predict(data)
```

---

**Remember**: Always backup your trained models! 💾

Enjoy exploring Network Intrusion Detection! 🛡️
