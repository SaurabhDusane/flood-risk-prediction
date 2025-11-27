"""
Random Forest Regressor - Ensemble Model
"""

import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from ..config import Config, MODELS_DIR


def train_random_forest(X_train, y_train, params=None, save_model=True):
    """
    Train Random Forest Regressor.
    
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
    print("\n[Random Forest] Training...")
    
    params = params or Config.RF_PARAMS
    
    start_time = time.time()
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    
    if save_model:
        path = MODELS_DIR / "random_forest.joblib"
        joblib.dump(model, path)
        print(f"  Model saved to: {path}")
    
    return model, train_time
