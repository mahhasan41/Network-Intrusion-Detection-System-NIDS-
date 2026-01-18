"""
Data Preprocessing Module for Network Intrusion Detection System
Handles data loading, cleaning, encoding, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for intrusion detection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self, filepath, nrows=None):
        """
        Load dataset from CSV file
        
        Args:
            filepath (str): Path to CSV file
            nrows (int): Number of rows to load (None = all rows)
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading dataset from {filepath}...")
        if nrows:
            print(f"Loading first {nrows} rows...")
            df = pd.read_csv(filepath, nrows=nrows)
        else:
            df = pd.read_csv(filepath)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def explore_data(self, df):
        """
        Perform exploratory data analysis
        
        Args:
            df (pd.DataFrame): Input dataset
        """
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        print(f"\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"\nMissing Values:\n{df.isnull().sum().sum()} total missing values")
        
        if df.isnull().sum().sum() > 0:
            print(f"Columns with missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        
        print(f"\nDuplicate Rows: {df.duplicated().sum()}")
        print(f"\nData Types:\n{df.dtypes}")
        
        # Label distribution
        if 'Label' in df.columns:
            print(f"\nLabel Distribution:\n{df['Label'].value_counts()}")
            print(f"\nLabel Distribution (%):\n{df['Label'].value_counts(normalize=True) * 100}")
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        print(f"\nHandling missing values using strategy: {strategy}...")
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # For numerical columns, fill with mean/median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"Missing values remaining: {df.isnull().sum().sum()}")
        return df
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset without duplicates
        """
        print(f"Removing duplicate rows...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} duplicate rows")
        return df
    
    def handle_infinite_values(self, df):
        """
        Replace infinite values with NaN and then handle them
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with infinite values handled
        """
        print("Handling infinite values...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with mean
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using LabelEncoder
        
        Args:
            df (pd.DataFrame): Input dataset
            fit (bool): Whether to fit the encoder
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        print("\nEncoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove Label column if it exists (handle separately)
        if 'Label' in categorical_cols:
            categorical_cols.remove('Label')
        
        self.categorical_features = categorical_cols
        
        for col in categorical_cols:
            if fit:
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))
            else:
                try:
                    df[col] = self.label_encoder.transform(df[col].astype(str))
                except:
                    # Handle unknown categories
                    df[col] = df[col].astype(str)
        
        print(f"Encoded {len(categorical_cols)} categorical features")
        return df
    
    def encode_label(self, df, fit=True):
        """
        Encode target label (Normal vs Attack types)
        
        Args:
            df (pd.DataFrame): Input dataset
            fit (bool): Whether to fit the encoder
            
        Returns:
            pd.DataFrame: Dataset with encoded labels
        """
        if 'Label' not in df.columns:
            print("Warning: No 'Label' column found")
            return df
        
        print("\nEncoding target labels...")
        
        if fit:
            # Create a mapping for binary classification (Normal vs Attack)
            label_mapping = {}
            unique_labels = df['Label'].unique()
            
            for i, label in enumerate(unique_labels):
                if 'BENIGN' in str(label).upper() or 'NORMAL' in str(label).upper():
                    label_mapping[label] = 0
                else:
                    label_mapping[label] = 1
            
            df['Label'] = df['Label'].map(label_mapping)
            self.label_mapping = label_mapping
            print(f"Label Encoding: {label_mapping}")
        else:
            df['Label'] = df['Label'].map(self.label_mapping)
        
        return df
    
    def select_features(self, X, y, k=50, method='f_classif'):
        """
        Select top k features using statistical methods
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            k (int): Number of top features to select
            method (str): Feature selection method ('f_classif' or 'mutual_info')
            
        Returns:
            tuple: (X_selected, selected_feature_names)
        """
        print(f"\nSelecting top {k} features using {method}...")
        
        if method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            score_func = f_classif
        
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def normalize_features(self, X, fit=True):
        """
        Normalize numerical features using StandardScaler
        
        Args:
            X (pd.DataFrame): Features
            fit (bool): Whether to fit the scaler
            
        Returns:
            np.ndarray: Normalized features
        """
        print("\nNormalizing features...")
        
        if fit:
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = self.scaler.transform(X)
        
        return X_normalized
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """
        Handle class imbalance using SMOTE or other methods
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target variable
            method (str): Method for handling imbalance ('smote')
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        # Check if we have more than 1 class
        unique_classes = len(np.unique(y))
        if unique_classes < 2:
            print(f"\n⚠ Warning: Only {unique_classes} class found in data. Skipping SMOTE balancing.")
            print(f"Original class distribution:\n{pd.Series(y).value_counts()}")
            return X, y
        
        print(f"\nHandling class imbalance using {method}...")
        print(f"Original class distribution:\n{pd.Series(y).value_counts()}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)  # type: ignore
        
        print(f"Balanced class distribution:\n{pd.Series(list(y_balanced)).value_counts()}")
        print(f"Original samples: {len(X)}, Balanced samples: {len(X_balanced)}")
        
        return X_balanced, y_balanced
    
    def prepare_data(self, filepath, test_size=0.2, apply_smote=True, 
                    feature_selection=True, k_features=50, nrows=None):
        """
        Complete data preparation pipeline
        
        Args:
            filepath (str): Path to dataset
            test_size (float): Test set size
            apply_smote (bool): Whether to apply SMOTE
            feature_selection (bool): Whether to perform feature selection
            k_features (int): Number of features to select
            nrows (int): Number of rows to load (None = all rows)
            
        Returns:
            dict: Dictionary containing train and test sets
        """
        # Load data
        df = self.load_data(filepath, nrows=nrows)
        
        # Explore data
        self.explore_data(df)
        
        # Data cleaning
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.handle_infinite_values(df)
        
        # Separate features and target
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        # Encode categorical features in X
        X = self.encode_categorical_features(X)
        
        # Encode target labels
        y = self.encode_label(df)['Label']
        
        self.numerical_features = X.columns.tolist()
        self.feature_names = X.columns.tolist()
        
        # Feature selection
        if feature_selection and X.shape[1] > k_features:
            X, selected_features = self.select_features(X, y, k=k_features)
            self.feature_names = selected_features
        
        # Normalize features
        X_normalized = self.normalize_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Handle class imbalance
        if apply_smote:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train)
        
        print("\n" + "="*50)
        print("DATA PREPARATION COMPLETE")
        print("="*50)
        print(f"Train set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
