"""
K-Nearest Neighbors Regressor - Baseline Model
"""

import time
import joblib
from sklearn.neighbors import KNeighborsRegressor
from ..config import Config, MODELS_DIR


def train_knn_regressor(X_train, y_train, params=None, save_model=True):
    """
    Train K-Nearest Neighbors Regressor.
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    params : dict, optional
        Model parameters
    save_model : bool
        Whether to save the trained model
    
    Returns
    -------
    tuple
        (model, training_time)
    """
    print("\n[KNN Regressor] Training...")
    
    params = params or Config.KNN_PARAMS
    
    start_time = time.time()
    model = KNeighborsRegressor(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    
    if save_model:
        path = MODELS_DIR / "knn_regressor.joblib"
        joblib.dump(model, path)
        print(f"  Model saved to: {path}")
    
    return model, train_time
