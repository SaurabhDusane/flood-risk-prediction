"""
Run all experiments and generate results.
This script trains all models and generates figures.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.4

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT.parent  # Parent folder has train.csv
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns
FEATURE_COLUMNS = [
    "MonsoonIntensity", "TopographyDrainage", "ClimateChange", "Watersheds",
    "CoastalVulnerability", "RiverManagement", "DamsQuality", "Siltation",
    "DrainageSystems", "WetlandLoss", "Deforestation", "Urbanization",
    "AgriculturalPractices", "Encroachments", "PopulationScore",
    "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness",
    "InadequatePlanning", "PoliticalFactors", "Landslides"
]


def load_data():
    """Load and preprocess data."""
    print("Loading data...")
    
    # Try different paths
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        train_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    
    if not train_path.exists():
        print(f"Error: train.csv not found at {train_path}")
        print("Please copy train.csv to the data/raw/ directory")
        sys.exit(1)
    
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df):,} instances")
    
    X = df[FEATURE_COLUMNS].values
    y = df['FloodProbability'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training set: {len(X_train):,}")
    print(f"Test set: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    """Compute metrics."""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models and collect results."""
    results = {}
    
    # 1. SGD Regressor
    print("\n" + "="*50)
    print("Training: SGD Regressor")
    print("="*50)
    start = time.time()
    model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    train_time = time.time() - start
    metrics = evaluate(y_test, model.predict(X_test))
    metrics['time'] = train_time
    results['SGD Regressor'] = metrics
    print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Time: {train_time:.2f}s")
    
    # 2. KNN Regressor (use subset for speed)
    print("\n" + "="*50)
    print("Training: KNN Regressor")
    print("="*50)
    start = time.time()
    # Use smaller sample for KNN (it's slow on large data)
    sample_size = min(50000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    model.fit(X_train[idx], y_train[idx])
    train_time = time.time() - start
    # Predict on subset of test too
    test_idx = np.random.choice(len(X_test), min(10000, len(X_test)), replace=False)
    metrics = evaluate(y_test[test_idx], model.predict(X_test[test_idx]))
    metrics['time'] = train_time
    results['KNN Regressor'] = metrics
    print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Time: {train_time:.2f}s")
    
    # 3. Random Forest
    print("\n" + "="*50)
    print("Training: Random Forest")
    print("="*50)
    start = time.time()
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=10,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start
    metrics = evaluate(y_test, model.predict(X_test))
    metrics['time'] = train_time
    metrics['feature_importance'] = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    results['Random Forest'] = metrics
    print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Time: {train_time:.2f}s")
    
    # 4. XGBoost
    print("\n" + "="*50)
    print("Training: XGBoost")
    print("="*50)
    start = time.time()
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start
    metrics = evaluate(y_test, model.predict(X_test))
    metrics['time'] = train_time
    results['XGBoost'] = metrics
    print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Time: {train_time:.2f}s")
    
    # 5. LightGBM
    print("\n" + "="*50)
    print("Training: LightGBM")
    print("="*50)
    start = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start
    metrics = evaluate(y_test, model.predict(X_test))
    metrics['time'] = train_time
    results['LightGBM'] = metrics
    print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Time: {train_time:.2f}s")
    
    # 6. Gradient Boosting (slower, use fewer estimators)
    print("\n" + "="*50)
    print("Training: Gradient Boosting")
    print("="*50)
    start = time.time()
    # Use subset for speed
    sample_size = min(100000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=5, learning_rate=0.1,
        subsample=0.5, random_state=RANDOM_STATE, verbose=0
    )
    model.fit(X_train[idx], y_train[idx])
    train_time = time.time() - start
    metrics = evaluate(y_test, model.predict(X_test))
    metrics['time'] = train_time
    results['Gradient Boosting'] = metrics
    print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Time: {train_time:.2f}s")
    
    return results


def generate_figures(results):
    """Generate visualization figures."""
    print("\n" + "="*50)
    print("Generating Figures")
    print("="*50)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Model names and metrics
    models = list(results.keys())
    r2_scores = [results[m]['r2'] for m in models]
    rmse_scores = [results[m]['rmse'] for m in models]
    times = [results[m]['time'] for m in models]
    
    # Sort by R2
    sorted_idx = np.argsort(r2_scores)[::-1]
    models = [models[i] for i in sorted_idx]
    r2_scores = [r2_scores[i] for i in sorted_idx]
    rmse_scores = [rmse_scores[i] for i in sorted_idx]
    times = [times[i] for i in sorted_idx]
    
    # Figure 1: R2 Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, r2_scores, color=colors)
    ax.set_xlabel('R² Score')
    ax.set_title('Model Performance Comparison (R² Score)')
    ax.set_xlim(0, 1)
    for bar, score in zip(bars, r2_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'r2_comparison.png', dpi=300)
    plt.close()
    print("  Saved: r2_comparison.png")
    
    # Figure 2: RMSE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, rmse_scores, color=colors)
    ax.set_xlabel('RMSE')
    ax.set_title('Model Performance Comparison (RMSE)')
    for bar, score in zip(bars, rmse_scores):
        ax.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rmse_comparison.png', dpi=300)
    plt.close()
    print("  Saved: rmse_comparison.png")
    
    # Figure 3: Training Time
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, times, color=colors)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Model Training Time Comparison')
    for bar, t in zip(bars, times):
        ax.text(t + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{t:.1f}s', va='center')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training_time.png', dpi=300)
    plt.close()
    print("  Saved: training_time.png")
    
    # Figure 4: Feature Importance (from Random Forest)
    if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
        importance = results['Random Forest']['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features[:15]]
        scores = [f[1] for f in sorted_features[:15]]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))[::-1]
        bars = ax.barh(features[::-1], scores[::-1], color=colors)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 15 Feature Importances (Random Forest)')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=300)
        plt.close()
        print("  Saved: feature_importance.png")
    
    # Figure 5: Combined comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.barh(models, r2_scores, color='steelblue')
    ax1.set_xlabel('R² Score')
    ax1.set_title('R² Score by Model')
    ax1.set_xlim(0, 1)
    
    ax2 = axes[1]
    ax2.barh(models, rmse_scores, color='coral')
    ax2.set_xlabel('RMSE')
    ax2.set_title('RMSE by Model')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=300)
    plt.close()
    print("  Saved: model_comparison.png")


def save_results(results):
    """Save results to files."""
    print("\n" + "="*50)
    print("Saving Results")
    print("="*50)
    
    # Prepare clean results (remove feature importance for JSON)
    clean_results = {}
    for model, metrics in results.items():
        clean_results[model] = {
            'r2': round(metrics['r2'], 4),
            'rmse': round(metrics['rmse'], 4),
            'mae': round(metrics['mae'], 4),
            'training_time': round(metrics['time'], 2)
        }
    
    # Save as JSON
    with open(METRICS_DIR / 'results.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    print("  Saved: results.json")
    
    # Save as CSV
    df = pd.DataFrame(clean_results).T
    df.index.name = 'Model'
    df = df.reset_index()
    df.to_csv(METRICS_DIR / 'results.csv', index=False)
    print("  Saved: results.csv")
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'Time':>10}")
    print("-"*70)
    
    sorted_models = sorted(clean_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    for model, m in sorted_models:
        print(f"{model:<25} {m['r2']:>10.4f} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['training_time']:>9.2f}s")
    
    print("-"*70)
    best = sorted_models[0]
    print(f"\nBest Model: {best[0]} (R² = {best[1]['r2']:.4f})")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("URBAN FLOOD RISK PREDICTION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train models
    results = train_all_models(X_train, X_test, y_train, y_test)
    
    # Generate figures
    generate_figures(results)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
