"""
model_training.py
-----------------
Defines and trains the LSTM deep learning model for stock price prediction.

Architecture rationale:
  - LSTM Layer 1: Learns short-to-medium term sequential patterns.
    return_sequences=True feeds the full sequence to the next LSTM layer.
  - Dropout after LSTM 1: Prevents co-adaptation of hidden units (regularisation).
  - LSTM Layer 2: Learns higher-order temporal abstractions.
  - Dropout after LSTM 2: Further regularisation.
  - Dense(32): Intermediate fully-connected layer for non-linear transformation.
  - Dense(1): Single output — the predicted scaled close price.

Optimizer: Adam (adaptive learning rate, good for noisy financial data)
Loss: MSE (penalises large prediction errors heavily)
"""

import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_units: int = 128,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    dense_units: int = 32
) -> tf.keras.Model:
    """
    Construct the stacked LSTM model.

    Args:
        sequence_length: Number of past timesteps per input sample
        n_features: Number of input features per timestep
        lstm_units: Number of hidden units in each LSTM layer
        dropout_rate: Dropout probability for regularisation
        learning_rate: Adam learning rate
        dense_units: Units in the intermediate Dense layer

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=(sequence_length, n_features)),

        # First LSTM layer — return full sequence for stacking
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),

        # Second LSTM layer — return last hidden state only
        LSTM(units=lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),

        # Non-linear projection
        Dense(units=dense_units, activation='relu'),

        # Output: single price prediction
        Dense(units=1)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    print("[ModelTraining] Model summary:")
    model.summary()
    return model


def get_callbacks(model_save_path: str = 'models/lstm_model.h5') -> list:
    """
    Return training callbacks:
      - EarlyStopping: stop when val_loss stops improving (avoids overfitting)
      - ModelCheckpoint: save the best model by val_loss
      - ReduceLROnPlateau: halve learning rate when val_loss plateaus
    """
    os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sequence_length: int,
    n_features: int,
    lstm_units: int = 128,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    model_save_path: str = 'models/lstm_model.h5'
) -> tuple:
    """
    Build, train and save the LSTM model.

    Args:
        X_train, y_train: Training sequences and targets
        X_test, y_test: Validation sequences and targets
        sequence_length: Window size used in preprocessing
        n_features: Number of input features
        lstm_units: LSTM hidden units
        dropout_rate: Dropout rate
        learning_rate: Adam learning rate
        batch_size: Mini-batch size
        epochs: Maximum training epochs (EarlyStopping may cut short)
        model_save_path: Path to save the best model weights

    Returns:
        (trained_model, training_history)
    """
    print(f"[ModelTraining] Building model — seq_len={sequence_length}, "
          f"features={n_features}, lstm_units={lstm_units}")

    model = build_lstm_model(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    callbacks = get_callbacks(model_save_path)

    print(f"[ModelTraining] Training for up to {epochs} epochs "
          f"(batch_size={batch_size})...")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        shuffle=False,       # Preserve temporal order within batches
        verbose=1
    )

    # Persist training config alongside weights
    config = {
        'sequence_length': sequence_length,
        'n_features': n_features,
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss'])
    }
    config_path = model_save_path.replace('.h5', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"[ModelTraining] Training complete. "
          f"Best val_loss: {min(history.history['val_loss']):.6f}")
    return model, history


def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load a previously saved Keras model from disk."""
    model = load_model(model_path)
    print(f"[ModelTraining] Model loaded from {model_path}")
    return model


def predict(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Generate scaled predictions for input sequences.

    Args:
        model: Trained Keras model
        X: Input array of shape (n_samples, sequence_length, n_features)

    Returns:
        Scaled predictions array of shape (n_samples,)
    """
    return model.predict(X, verbose=0).flatten()
