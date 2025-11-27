#!/usr/bin/env python
"""
Training pipeline for Urban Flood Risk Prediction.

This script trains and evaluates all models on the flood prediction dataset.

Usage:
    python train_all_models.py              # Train all models
    python train_all_models.py --model rf   # Train specific model
"""

import argparse
import time
from datetime import datetime

import numpy as np

from data_preprocessing import load_and_preprocess
from models import (
    train_sgd_regressor,
    train_knn_regressor,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    train_lightgbm,
    train_neural_network
)
from evaluation import evaluate_model, save_results, generate_report
from visualization import plot_results
from config import RANDOM_STATE


# Model registry
MODEL_REGISTRY = {
    'sgd': ('SGD Regressor', train_sgd_regressor),
    'knn': ('KNN Regressor', train_knn_regressor),
    'rf': ('Random Forest', train_random_forest),
    'gb': ('Gradient Boosting', train_gradient_boosting),
    'xgb': ('XGBoost', train_xgboost),
    'lgbm': ('LightGBM', train_lightgbm),
    'nn': ('Neural Network', train_neural_network)
}


def train_single_model(model_key, X_train, y_train, X_val, y_val):
    """
    Train and evaluate a single model.
    
    Parameters
    ----------
    model_key : str
        Model key from MODEL_REGISTRY
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    
    Returns
    -------
    dict
        Results dictionary
    """
    model_name, train_func = MODEL_REGISTRY[model_key]
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print('='*60)
    
    # Train model
    if model_key == 'nn':
        result = train_func(X_train, y_train, X_val, y_val)
    else:
        result = train_func(X_train, y_train)
    
    model, train_time = result
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, model_name)
    metrics['training_time'] = train_time
    
    return model_name, model, metrics


def train_all_models(X_train, y_train, X_val, y_val, models_to_train=None):
    """
    Train and evaluate all (or specified) models.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    models_to_train : list, optional
        List of model keys to train. If None, trains all.
    
    Returns
    -------
    dict
        Dictionary of all results
    """
    if models_to_train is None:
        models_to_train = list(MODEL_REGISTRY.keys())
    
    all_results = {}
    all_models = {}
    
    print("\n" + "="*60)
    print("URBAN FLOOD RISK PREDICTION - MODEL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nTraining set: {len(X_train):,} instances")
    print(f"Validation set: {len(X_val):,} instances")
    print(f"Features: {X_train.shape[1]}")
    print(f"Models to train: {len(models_to_train)}")
    
    total_start = time.time()
    
    for model_key in models_to_train:
        if model_key not in MODEL_REGISTRY:
            print(f"\nWarning: Unknown model key '{model_key}', skipping...")
            continue
        
        model_name, model, metrics = train_single_model(
            model_key, X_train, y_train, X_val, y_val
        )
        
        all_results[model_name] = metrics
        all_models[model_name] = model
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Models trained: {len(all_results)}")
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1].get('r2', 0))
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   R² Score: {best_model[1]['r2']:.4f}")
    print(f"   RMSE: {best_model[1]['rmse']:.4f}")
    
    return all_results, all_models


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train flood prediction models'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=list(MODEL_REGISTRY.keys()) + ['all'],
        default='all',
        help='Model to train (default: all)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save models and results'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not generate plots'
    )
    
    args = parser.parse_args()
    
    # Determine models to train
    if args.model == 'all':
        models_to_train = None
    else:
        models_to_train = [args.model]
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    try:
        X_train, X_val, y_train, y_val = load_and_preprocess()
    except FileNotFoundError:
        print("\nError: Data files not found!")
        print("Please download the dataset first. See data/README.md for instructions.")
        return
    
    # Train models
    results, models = train_all_models(
        X_train, y_train, X_val, y_val,
        models_to_train=models_to_train
    )
    
    # Save results
    if not args.no_save:
        print("\nSaving results...")
        save_results(results)
        
        # Generate report
        report = generate_report(results)
        print("\n" + report)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualization plots...")
        plot_results(results)
    
    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
