"""
SGD Regressor - Baseline Model
"""

import time
import joblib
from sklearn.linear_model import SGDRegressor
from ..config import Config, MODELS_DIR


def train_sgd_regressor(X_train, y_train, params=None, save_model=True):
    """
    Train SGD Regressor model.
    
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
    print("\n[SGD Regressor] Training...")
    
    params = params or Config.SGD_PARAMS
    
    start_time = time.time()
    model = SGDRegressor(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    
    if save_model:
        path = MODELS_DIR / "sgd_regressor.joblib"
        joblib.dump(model, path)
        print(f"  Model saved to: {path}")
    
    return model, train_time
