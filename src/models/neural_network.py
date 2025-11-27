"""
Neural Network - Deep Learning Model
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from ..config import Config, MODELS_DIR


def build_neural_network(input_dim, params=None):
    """
    Build Neural Network architecture.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    params : dict, optional
        Network parameters
    
    Returns
    -------
    keras.Model
        Compiled Keras model
    """
    params = params or Config.NN_PARAMS
    
    model = keras.Sequential()
    
    # First hidden layer
    model.add(layers.Dense(
        params['hidden_layers'][0],
        activation=params['activation'],
        input_shape=(input_dim,)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(params['dropout_rate']))
    
    # Additional hidden layers
    for units in params['hidden_layers'][1:]:
        model.add(layers.Dense(units, activation=params['activation']))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params['dropout_rate']))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss=params['loss'], metrics=['mae'])
    
    return model


def train_neural_network(X_train, y_train, X_val=None, y_val=None, 
                         params=None, save_model=True):
    """
    Train Neural Network model.
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_val : array-like, optional
        Validation features
    y_val : array-like, optional
        Validation target
    params : dict, optional
        Model parameters
    save_model : bool
        Whether to save the trained model
    
    Returns
    -------
    tuple
        (model, training_time)
    """
    print("\n[Neural Network] Training...")
    
    params = params or Config.NN_PARAMS
    
    # Build model
    model = build_neural_network(X_train.shape[1], params)
    print(f"  Architecture: Input({X_train.shape[1]}) -> "
          f"{' -> '.join(map(str, params['hidden_layers']))} -> Output(1)")
    
    # Callbacks
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=params['patience'],
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    validation_data = (X_val, y_val) if X_val is not None else None
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=validation_data,
        callbacks=callback_list,
        verbose=1
    )
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    
    if save_model:
        path = MODELS_DIR / "neural_network.keras"
        model.save(path)
        print(f"  Model saved to: {path}")
    
    return model, train_time
