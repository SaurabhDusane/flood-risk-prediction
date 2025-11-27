"""
Urban Flood Risk Prediction

A machine learning pipeline for predicting urban flood probability
using ensemble methods and deep learning.

Authors:
    - Saurabh Nilesh Dusane (sdusane@asu.edu)
    - Hrisika Akila Jagdeep (hjagdeep@asu.edu)
    - Neeraj Suresh Narayanan (nnaray23@asu.edu)

Course: CSE 572 - Data Mining
Institution: Arizona State University
"""

__version__ = "1.0.0"
__author__ = "Saurabh Dusane, Hrisika Jagdeep, Neeraj Narayanan"
__email__ = "sdusane@asu.edu"

from .config import Config
from .data_preprocessing import load_data, preprocess_data, load_and_preprocess
from .feature_engineering import FeatureEngineer
from .models import (
    train_sgd_regressor,
    train_knn_regressor,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    train_lightgbm,
    train_neural_network
)
from .evaluation import evaluate_model, cross_validate_model
from .visualization import plot_results, plot_feature_importance

__all__ = [
    "Config",
    "load_data",
    "preprocess_data",
    "load_and_preprocess",
    "FeatureEngineer",
    "train_sgd_regressor",
    "train_knn_regressor",
    "train_random_forest",
    "train_gradient_boosting",
    "train_xgboost",
    "train_lightgbm",
    "train_neural_network",
    "evaluate_model",
    "cross_validate_model",
    "plot_results",
    "plot_feature_importance"
]
