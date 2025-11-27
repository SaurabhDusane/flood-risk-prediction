"""
Gradient Boosting Regressor - Ensemble Model
"""

import time
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from ..config import Config, MODELS_DIR


def train_gradient_boosting(X_train, y_train, params=None, save_model=True):
    """
    Train Gradient Boosting Regressor.
    
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
    print("\n[Gradient Boosting] Training...")
    
    params = params or Config.GB_PARAMS
    
    start_time = time.time()
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    
    if save_model:
        path = MODELS_DIR / "gradient_boosting.joblib"
        joblib.dump(model, path)
        print(f"  Model saved to: {path}")
    
    return model, train_time
