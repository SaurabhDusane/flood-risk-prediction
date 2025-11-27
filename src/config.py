"""
Configuration settings and hyperparameters for Urban Flood Risk Prediction.

This module contains all configurable parameters for:
- Data paths
- Model hyperparameters
- Training settings
- Evaluation metrics
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"

# =============================================================================
# RANDOM STATE
# =============================================================================

RANDOM_STATE = 42

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Train-test split ratio
TEST_SIZE = 0.4  # 60-40 split

# Cross-validation folds
CV_FOLDS = 5

# Target variable
TARGET_COLUMN = "FloodProbability"

# ID column (to exclude from features)
ID_COLUMN = "id"

# Feature columns (20 features)
FEATURE_COLUMNS = [
    # Environmental Factors
    "MonsoonIntensity",
    "TopographyDrainage",
    "ClimateChange",
    "Watersheds",
    "CoastalVulnerability",
    # Water Management
    "RiverManagement",
    "DamsQuality",
    "Siltation",
    "DrainageSystems",
    "WetlandLoss",
    # Human Activities
    "Deforestation",
    "Urbanization",
    "AgriculturalPractices",
    "Encroachments",
    "PopulationScore",
    # Infrastructure
    "DeterioratingInfrastructure",
    "IneffectiveDisasterPreparedness",
    "InadequatePlanning",
    "PoliticalFactors",
    # Natural Hazards
    "Landslides"
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

class Config:
    """Configuration class containing all model hyperparameters."""
    
    # SGD Regressor
    SGD_PARAMS = {
        'loss': 'squared_error',
        'penalty': 'l2',
        'alpha': 0.0001,
        'max_iter': 1000,
        'tol': 1e-3,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 5
    }
    
    # KNN Regressor
    KNN_PARAMS = {
        'n_neighbors': 5,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,  # Euclidean distance
        'n_jobs': -1
    }
    
    # Random Forest
    RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': 1
    }
    
    # Gradient Boosting
    GB_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.5,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'random_state': RANDOM_STATE,
        'verbose': 1
    }
    
    # XGBoost
    XGB_PARAMS = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    # LightGBM
    LGBM_PARAMS = {
        'n_estimators': 200,
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': 1
    }
    
    # Neural Network
    NN_PARAMS = {
        'hidden_layers': [256, 128, 64],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 512,
        'epochs': 50,
        'patience': 10,  # Early stopping patience
        'activation': 'relu',
        'optimizer': 'adam',
        'loss': 'mse'
    }

# =============================================================================
# EVALUATION METRICS
# =============================================================================

# Regression metrics to compute
REGRESSION_METRICS = ['r2', 'rmse', 'mae', 'mse']

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)

# Color palette for models
MODEL_COLORS = {
    'SGD Regressor': '#1f77b4',
    'KNN Regressor': '#ff7f0e',
    'Random Forest': '#2ca02c',
    'Gradient Boosting': '#d62728',
    'XGBoost': '#9467bd',
    'LightGBM': '#8c564b',
    'Neural Network': '#e377c2'
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
