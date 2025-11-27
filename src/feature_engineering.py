"""
Feature engineering module for Urban Flood Risk Prediction.

This module provides:
- Feature extraction methods
- Feature selection techniques
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_regression,
    SelectKBest,
    RFE
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from .config import FEATURE_COLUMNS, RANDOM_STATE


class FeatureEngineer:
    """
    Feature engineering class for flood prediction.
    
    Provides methods for feature analysis, selection, and transformation.
    """
    
    def __init__(self, feature_names=None):
        """
        Initialize FeatureEngineer.
        
        Parameters
        ----------
        feature_names : list, optional
            List of feature names
        """
        self.feature_names = feature_names or FEATURE_COLUMNS
        self.selected_features = None
        self.feature_importances = None
        self.mutual_info_scores = None
    
    def compute_mutual_information(self, X, y, n_neighbors=3):
        """
        Compute mutual information between features and target.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_neighbors : int
            Number of neighbors for MI estimation
        
        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and MI scores, sorted by score
        """
        print("Computing mutual information scores...")
        
        mi_scores = mutual_info_regression(
            X, y, 
            n_neighbors=n_neighbors,
            random_state=RANDOM_STATE
        )
        
        self.mutual_info_scores = pd.DataFrame({
            'feature': self.feature_names,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print("Top 10 features by mutual information:")
        print(self.mutual_info_scores.head(10).to_string(index=False))
        
        return self.mutual_info_scores
    
    def compute_correlation_matrix(self, X):
        """
        Compute correlation matrix between features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        df = pd.DataFrame(X, columns=self.feature_names)
        corr_matrix = df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print("\nHighly correlated feature pairs (|r| > 0.5):")
            for pair in high_corr_pairs:
                print(f"  {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")
        else:
            print("\nNo highly correlated feature pairs found (|r| > 0.5)")
        
        return corr_matrix
    
    def compute_rf_importance(self, X, y, n_estimators=100):
        """
        Compute feature importance using Random Forest.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_estimators : int
            Number of trees
        
        Returns
        -------
        pd.DataFrame
            DataFrame with feature importances
        """
        print("Computing Random Forest feature importances...")
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        self.feature_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features by RF importance:")
        print(self.feature_importances.head(10).to_string(index=False))
        
        return self.feature_importances
    
    def select_features_rfe(self, X, y, n_features=15, estimator=None):
        """
        Select features using Recursive Feature Elimination.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_features : int
            Number of features to select
        estimator : object, optional
            Estimator for RFE
        
        Returns
        -------
        list
            Selected feature names
        """
        print(f"Performing RFE to select top {n_features} features...")
        
        if estimator is None:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        
        rfe = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        
        self.selected_features = [
            self.feature_names[i] for i, selected 
            in enumerate(rfe.support_) if selected
        ]
        
        print(f"Selected features: {self.selected_features}")
        
        return self.selected_features
    
    def create_polynomial_features(self, X, degree=2, interaction_only=False):
        """
        Create polynomial features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        degree : int
            Polynomial degree
        interaction_only : bool
            If True, only interaction terms are created
        
        Returns
        -------
        tuple
            (transformed_X, feature_names)
        """
        print(f"Creating polynomial features (degree={degree})...")
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        X_poly = poly.fit_transform(X)
        
        poly_feature_names = poly.get_feature_names_out(self.feature_names)
        
        print(f"  Original features: {X.shape[1]}")
        print(f"  Polynomial features: {X_poly.shape[1]}")
        
        return X_poly, poly_feature_names
    
    def analyze_features(self, X, y):
        """
        Perform comprehensive feature analysis.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        
        Returns
        -------
        dict
            Dictionary containing all analysis results
        """
        print("\n" + "="*60)
        print("FEATURE ANALYSIS")
        print("="*60 + "\n")
        
        # Mutual information
        mi_scores = self.compute_mutual_information(X, y)
        
        # Correlation matrix
        print("\n" + "-"*40)
        corr_matrix = self.compute_correlation_matrix(X)
        
        # RF importance
        print("\n" + "-"*40)
        rf_importance = self.compute_rf_importance(X, y)
        
        print("\n" + "="*60)
        print("FEATURE ANALYSIS COMPLETE")
        print("="*60 + "\n")
        
        return {
            'mutual_information': mi_scores,
            'correlation_matrix': corr_matrix,
            'rf_importance': rf_importance
        }
    
    def get_top_features(self, n=10):
        """
        Get top N features based on computed importance.
        
        Parameters
        ----------
        n : int
            Number of top features
        
        Returns
        -------
        list
            Top feature names
        """
        if self.feature_importances is not None:
            return self.feature_importances.head(n)['feature'].tolist()
        elif self.mutual_info_scores is not None:
            return self.mutual_info_scores.head(n)['feature'].tolist()
        else:
            raise ValueError("No feature importance scores computed. Run analyze_features first.")


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering module...")
    
    # Generate sample data
    np.random.seed(RANDOM_STATE)
    X_sample = np.random.randn(1000, 20)
    y_sample = np.random.randn(1000)
    
    fe = FeatureEngineer()
    results = fe.analyze_features(X_sample, y_sample)
    
    print("\nFeature engineering test complete!")
