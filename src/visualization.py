"""
Visualization module for Urban Flood Risk Prediction.

This module provides functions for:
- Model comparison plots
- Feature importance visualization
- Error analysis plots
- Publication-ready figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .config import FIGURES_DIR, FIGURE_DPI, FIGURE_SIZE, MODEL_COLORS


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def setup_plot_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight'
    })


def plot_model_comparison(results_dict, metric='r2', save_path=None):
    """
    Create bar chart comparing model performance.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to metrics
    metric : str
        Metric to plot ('r2', 'rmse', 'mae')
    save_path : str or Path, optional
        Path to save figure
    """
    setup_plot_style()
    
    # Extract data
    models = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) for m in models]
    
    # Sort by performance
    if metric == 'r2':
        sorted_indices = np.argsort(values)[::-1]  # Descending
    else:
        sorted_indices = np.argsort(values)  # Ascending for error metrics
    
    models = [models[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Colors
    colors = [MODEL_COLORS.get(m, '#1f77b4') for m in models]
    
    # Create bars
    bars = ax.barh(models, values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontsize=10)
    
    # Labels and title
    metric_labels = {
        'r2': 'R² Score',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mse': 'MSE'
    }
    ax.set_xlabel(metric_labels.get(metric, metric))
    ax.set_title(f'Model Comparison: {metric_labels.get(metric, metric)}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    save_path : str or Path, optional
        Path to save figure
    """
    setup_plot_style()
    
    # Get top features
    df = importance_df.head(top_n).copy()
    df = df.sort_values('importance', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bars with gradient colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    
    bars = ax.barh(df['feature'], df['importance'], color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_residuals(y_true, y_pred, model_name="Model", save_path=None):
    """
    Create residual analysis plots.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Name of the model
    save_path : str or Path, optional
        Path to save figure
    """
    setup_plot_style()
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    
    # 2. Actual vs Predicted
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax2.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Actual vs Predicted')
    ax2.legend()
    
    # 3. Residual Distribution
    ax3 = axes[2]
    ax3.hist(residuals, bins=50, edgecolor='white', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution')
    
    # Add statistics
    stats_text = f'Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}'
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'{model_name} - Residual Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_cross_validation(cv_results, save_path=None):
    """
    Plot cross-validation results as box plot.
    
    Parameters
    ----------
    cv_results : dict
        Dictionary mapping model names to CV score lists
    save_path : str or Path, optional
        Path to save figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Prepare data
    data = []
    labels = []
    for model_name, scores in cv_results.items():
        data.append(scores)
        labels.append(model_name)
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Colors
    for i, (patch, label) in enumerate(zip(bp['boxes'], labels)):
        patch.set_facecolor(MODEL_COLORS.get(label, '#1f77b4'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('R² Score')
    ax.set_title('Cross-Validation Performance')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_correlation_heatmap(X, feature_names, save_path=None):
    """
    Plot feature correlation heatmap.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : list
        List of feature names
    save_path : str or Path, optional
        Path to save figure
    """
    setup_plot_style()
    
    # Compute correlation
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.5},
                annot=True, fmt='.2f', annot_kws={'size': 8})
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_results(results_dict, output_dir=None):
    """
    Generate all standard visualization plots.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of model results
    output_dir : str or Path, optional
        Directory to save figures
    """
    output_dir = Path(output_dir) if output_dir else FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualization plots...")
    
    # R² comparison
    plot_model_comparison(
        results_dict, 
        metric='r2',
        save_path=output_dir / 'r2_comparison.png'
    )
    
    # RMSE comparison
    plot_model_comparison(
        results_dict,
        metric='rmse',
        save_path=output_dir / 'rmse_comparison.png'
    )
    
    # MAE comparison
    plot_model_comparison(
        results_dict,
        metric='mae',
        save_path=output_dir / 'mae_comparison.png'
    )
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization module...")
    
    # Sample data
    sample_results = {
        'SGD Regressor': {'r2': 0.452, 'rmse': 0.046, 'mae': 0.038},
        'KNN Regressor': {'r2': 0.400, 'rmse': 0.050, 'mae': 0.041},
        'Random Forest': {'r2': 0.609, 'rmse': 0.032, 'mae': 0.026},
        'XGBoost': {'r2': 0.821, 'rmse': 0.022, 'mae': 0.017},
        'LightGBM': {'r2': 0.828, 'rmse': 0.021, 'mae': 0.017},
        'Neural Network': {'r2': 0.861, 'rmse': 0.019, 'mae': 0.015}
    }
    
    # Test plot
    plot_model_comparison(sample_results, metric='r2')
    
    print("\nVisualization module test complete!")
