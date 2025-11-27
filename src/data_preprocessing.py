"""
Data preprocessing module for Urban Flood Risk Prediction.

This module provides functions for:
- Loading raw data
- Handling missing values
- Feature scaling
- Train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

from .config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, PROCESSED_DATA_DIR,
    TARGET_COLUMN, ID_COLUMN, FEATURE_COLUMNS,
    TEST_SIZE, RANDOM_STATE
)


def load_data(train_path=None, test_path=None):
    """
    Load raw training and test data from CSV files.
    
    Parameters
    ----------
    train_path : str or Path, optional
        Path to training data CSV. Defaults to config path.
    test_path : str or Path, optional
        Path to test data CSV. Defaults to config path.
    
    Returns
    -------
    tuple
        (train_df, test_df) DataFrames
    """
    train_path = train_path or TRAIN_DATA_PATH
    test_path = test_path or TEST_DATA_PATH
    
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  → Loaded {len(train_df):,} training instances")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  → Loaded {len(test_df):,} test instances")
    
    return train_df, test_df


def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str
        Imputation strategy ('median', 'mean', 'most_frequent')
    
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values
    """
    imputer = SimpleImputer(strategy=strategy)
    
    # Get numeric columns (excluding id and target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [c for c in numeric_cols if c not in [ID_COLUMN, TARGET_COLUMN]]
    
    if df[cols_to_impute].isnull().sum().sum() > 0:
        print(f"Imputing missing values using {strategy} strategy...")
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
        print("  → Missing values imputed")
    else:
        print("  → No missing values found")
    
    return df


def scale_features(X_train, X_test=None, save_scaler=True):
    """
    Standardize features using StandardScaler.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    X_test : np.ndarray or pd.DataFrame, optional
        Test features
    save_scaler : bool
        Whether to save the fitted scaler
    
    Returns
    -------
    tuple
        Scaled (X_train, X_test) or just X_train if X_test is None
    """
    scaler = StandardScaler()
    
    print("Scaling features...")
    X_train_scaled = scaler.fit_transform(X_train)
    print("  → Training features scaled (fit + transform)")
    
    if save_scaler:
        scaler_path = PROCESSED_DATA_DIR / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"  → Scaler saved to: {scaler_path}")
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        print("  → Test features scaled (transform only)")
        return X_train_scaled, X_test_scaled
    
    return X_train_scaled


def preprocess_data(train_df, test_df=None, scale=True):
    """
    Full preprocessing pipeline.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame
    test_df : pd.DataFrame, optional
        Test DataFrame (for Kaggle submission)
    scale : bool
        Whether to scale features
    
    Returns
    -------
    dict
        Preprocessed data dictionary
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Handle missing values
    train_df = handle_missing_values(train_df.copy())
    if test_df is not None:
        test_df = handle_missing_values(test_df.copy())
    
    # Extract features and target
    X = train_df[FEATURE_COLUMNS].values
    y = train_df[TARGET_COLUMN].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Target mean: {y.mean():.3f} ± {y.std():.3f}")
    
    # Split data
    print(f"\nSplitting data ({int((1-TEST_SIZE)*100)}-{int(TEST_SIZE*100)} split)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  → Training set: {len(X_train):,} instances")
    print(f"  → Validation set: {len(X_val):,} instances")
    
    # Scale features
    if scale:
        X_train, X_val = scale_features(X_train, X_val)
    
    result = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'feature_names': FEATURE_COLUMNS
    }
    
    # Process Kaggle test set if provided
    if test_df is not None:
        X_kaggle = test_df[FEATURE_COLUMNS].values
        if scale:
            scaler = joblib.load(PROCESSED_DATA_DIR / "scaler.joblib")
            X_kaggle = scaler.transform(X_kaggle)
        result['X_kaggle'] = X_kaggle
        result['kaggle_ids'] = test_df[ID_COLUMN].values
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60 + "\n")
    
    return result


def load_and_preprocess(train_path=None, test_path=None, scale=True):
    """
    Convenience function to load and preprocess data in one step.
    
    Parameters
    ----------
    train_path : str or Path, optional
        Path to training data
    test_path : str or Path, optional
        Path to test data
    scale : bool
        Whether to scale features
    
    Returns
    -------
    tuple
        (X_train, X_val, y_train, y_val) arrays
    """
    train_df, test_df = load_data(train_path, test_path)
    result = preprocess_data(train_df, test_df, scale=scale)
    
    return result['X_train'], result['X_val'], result['y_train'], result['y_val']


def save_processed_data(data_dict, prefix='processed'):
    """
    Save preprocessed data to disk.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing preprocessed data arrays
    prefix : str
        Filename prefix
    """
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            path = PROCESSED_DATA_DIR / f"{prefix}_{key}.npy"
            np.save(path, value)
            print(f"Saved: {path}")


def load_processed_data(prefix='processed'):
    """
    Load preprocessed data from disk.
    
    Parameters
    ----------
    prefix : str
        Filename prefix
    
    Returns
    -------
    dict
        Dictionary containing preprocessed data arrays
    """
    data_dict = {}
    keys = ['X_train', 'X_val', 'y_train', 'y_val']
    
    for key in keys:
        path = PROCESSED_DATA_DIR / f"{prefix}_{key}.npy"
        if path.exists():
            data_dict[key] = np.load(path)
            print(f"Loaded: {path}")
    
    return data_dict


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing data preprocessing pipeline...")
    
    try:
        X_train, X_val, y_train, y_val = load_and_preprocess()
        
        print("\nPreprocessing test successful!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_val shape: {y_val.shape}")
        
    except FileNotFoundError as e:
        print(f"\nNote: Data files not found. Please download the dataset first.")
        print(f"See data/README.md for instructions.")
