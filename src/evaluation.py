"""
Evaluation module for Urban Flood Risk Prediction.

This module provides:
- Model evaluation metrics
- Cross-validation utilities
- Results comparison
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error
)
from sklearn.model_selection import cross_val_score, KFold
import json
from pathlib import Path

from .config import METRICS_DIR, CV_FOLDS, RANDOM_STATE


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }
    
    return metrics


def evaluate_model(model, X_test, y_test, model_name=None):
    """
    Evaluate a trained model on test data.
    
    Parameters
    ----------
    model : object
        Trained model with predict method
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    model_name : str, optional
        Name of the model for display
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Flatten if needed (for Keras models)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    
    # Display results
    if model_name:
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS: {model_name}")
        print('='*50)
    
    print(f"  R² Score:  {metrics['r2']:.4f} ({metrics['r2']*100:.2f}% variance explained)")
    print(f"  RMSE:      {metrics['rmse']:.4f}")
    print(f"  MAE:       {metrics['mae']:.4f}")
    print(f"  MSE:       {metrics['mse']:.6f}")
    
    return metrics


def cross_validate_model(model, X, y, cv=None, scoring='r2'):
    """
    Perform cross-validation on a model.
    
    Parameters
    ----------
    model : object
        Model with fit and predict methods
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    cv : int or object, optional
        Number of folds or CV splitter
    scoring : str
        Scoring metric
    
    Returns
    -------
    dict
        Cross-validation results
    """
    cv = cv or CV_FOLDS
    
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    # Create KFold if integer provided
    if isinstance(cv, int):
        kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    else:
        kfold = cv
    
    # Compute CV scores
    scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
    
    results = {
        'scores': scores.tolist(),
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'cv': float(np.std(scores) / np.mean(scores) * 100) if np.mean(scores) != 0 else 0
    }
    
    print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean ± Std:  {results['mean']:.4f} ± {results['std']:.4f}")
    print(f"  CV (%):      {results['cv']:.2f}%")
    
    return results


def compare_models(results_dict, sort_by='r2'):
    """
    Compare multiple model results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to their metrics
    sort_by : str
        Metric to sort by
    
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame sorted by performance
    """
    # Create comparison DataFrame
    rows = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by specified metric (descending for r2, ascending for error metrics)
    ascending = sort_by not in ['r2', 'r2_score']
    df = df.sort_values(sort_by, ascending=ascending)
    
    return df


def save_results(results_dict, filename='model_results'):
    """
    Save evaluation results to files.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of model results
    filename : str
        Base filename (without extension)
    """
    # Save as JSON
    json_path = METRICS_DIR / f"{filename}.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Save as CSV
    df = compare_models(results_dict)
    csv_path = METRICS_DIR / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return df


def load_results(filename='model_results'):
    """
    Load evaluation results from file.
    
    Parameters
    ----------
    filename : str
        Base filename (without extension)
    
    Returns
    -------
    dict
        Dictionary of model results
    """
    json_path = METRICS_DIR / f"{filename}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_report(results_dict, include_cv=False):
    """
    Generate a formatted evaluation report.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of model results
    include_cv : bool
        Whether to include CV results
    
    Returns
    -------
    str
        Formatted report string
    """
    report = []
    report.append("="*70)
    report.append("MODEL EVALUATION REPORT")
    report.append("Urban Flood Risk Prediction - CSE 572")
    report.append("="*70)
    report.append("")
    
    # Summary table
    report.append("PERFORMANCE SUMMARY")
    report.append("-"*70)
    report.append(f"{'Model':<25} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'Time (s)':>10}")
    report.append("-"*70)
    
    # Sort by R² (descending)
    sorted_models = sorted(
        results_dict.items(),
        key=lambda x: x[1].get('r2', 0),
        reverse=True
    )
    
    for model_name, metrics in sorted_models:
        r2 = metrics.get('r2', 0)
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0)
        time = metrics.get('training_time', 0)
        
        report.append(f"{model_name:<25} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f} {time:>10.2f}")
    
    report.append("-"*70)
    report.append("")
    
    # Best model
    best_model = sorted_models[0][0]
    best_r2 = sorted_models[0][1].get('r2', 0)
    report.append(f"BEST MODEL: {best_model} (R² = {best_r2:.4f})")
    report.append("")
    
    # Improvement over baseline
    if len(sorted_models) > 1:
        baseline_r2 = sorted_models[-1][1].get('r2', 0)
        improvement = (best_r2 - baseline_r2) / baseline_r2 * 100 if baseline_r2 > 0 else 0
        report.append(f"Improvement over baseline: +{improvement:.1f}%")
    
    report.append("")
    report.append("="*70)
    
    return "\n".join(report)


def print_comparison_table(results_dict):
    """
    Print a formatted comparison table.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of model results
    """
    df = compare_models(results_dict)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Format for display
    display_cols = ['Model', 'r2', 'rmse', 'mae']
    if 'training_time' in df.columns:
        display_cols.append('training_time')
    
    df_display = df[display_cols].copy()
    
    # Rename columns for display
    df_display.columns = ['Model', 'R² Score', 'RMSE', 'MAE'] + \
                        (['Time (s)'] if 'training_time' in display_cols else [])
    
    print(df_display.to_string(index=False))
    print("="*80)
    
    # Highlight best
    best_idx = df['r2'].idxmax()
    print(f"\n🏆 Best Model: {df.loc[best_idx, 'Model']} (R² = {df.loc[best_idx, 'r2']:.4f})")


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation module...")
    
    # Generate sample predictions
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    print("\nSample metrics:", metrics)
    
    print("\nEvaluation module test complete!")
