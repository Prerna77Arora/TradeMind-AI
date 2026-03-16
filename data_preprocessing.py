"""
data_preprocessing.py
----------------------
Scales features and creates sliding-window sequences for LSTM training.

Why Min-Max scaling?
  LSTMs are sensitive to input magnitude. Scaling all features to [0, 1]
  prevents larger-valued features (e.g., raw price vs RSI) from dominating
  gradient updates and speeds up convergence.

Why sliding windows?
  LSTMs learn temporal patterns from sequences. A window of 30 days gives
  the model context to learn patterns like double-bottoms, head-and-shoulders,
  moving-average crossovers, etc.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def scale_features(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'Close',
    scaler_save_path: str = None
) -> tuple:
    """
    Fit MinMaxScaler on training data and transform all features.

    Args:
        df: Feature-engineered DataFrame
        feature_cols: List of input feature column names
        target_col: Column to predict (Close price)
        scaler_save_path: If provided, save the fitted scalers here

    Returns:
        (scaled_features np.array, scaled_target np.array,
         feature_scaler, target_scaler)
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(df[feature_cols].values)
    scaled_target = target_scaler.fit_transform(df[[target_col]].values)

    if scaler_save_path:
        os.makedirs(scaler_save_path, exist_ok=True)
        joblib.dump(feature_scaler, os.path.join(scaler_save_path, 'feature_scaler.pkl'))
        joblib.dump(target_scaler, os.path.join(scaler_save_path, 'target_scaler.pkl'))
        print(f"[Preprocessing] Scalers saved to {scaler_save_path}")

    return scaled_features, scaled_target, feature_scaler, target_scaler


def load_scalers(scaler_path: str):
    """Load previously saved scalers from disk."""
    feature_scaler = joblib.load(os.path.join(scaler_path, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(scaler_path, 'target_scaler.pkl'))
    return feature_scaler, target_scaler


def create_sequences(
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
    sequence_length: int = 30
) -> tuple:
    """
    Build sliding-window (X, y) pairs for LSTM training.

    For each timestep t:
      X[i] = scaled_features[t : t + sequence_length]   shape: (seq_len, n_features)
      y[i] = scaled_target[t + sequence_length]          shape: (1,)

    Args:
        scaled_features: 2D array (time_steps, n_features)
        scaled_target: 2D array (time_steps, 1)
        sequence_length: Number of past days in each input window

    Returns:
        X (np.ndarray): shape (n_samples, sequence_length, n_features)
        y (np.ndarray): shape (n_samples,)
    """
    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length: i])
        y.append(scaled_target[i, 0])

    X = np.array(X)
    y = np.array(y)
    print(f"[Preprocessing] Sequences created — X: {X.shape}, y: {y.shape}")
    return X, y


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2
) -> tuple:
    """
    Split data into train and test sets preserving temporal order.
    Random shuffling is intentionally avoided to prevent look-ahead bias.

    Args:
        X: Input sequences
        y: Target values
        test_ratio: Fraction of data reserved for testing

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"[Preprocessing] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def get_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Return all numeric columns suitable as model features.

    Args:
        df: Feature-engineered DataFrame
        exclude: Columns to exclude (e.g., raw OHLCV that overlap with features)

    Returns:
        List of feature column names
    """
    if exclude is None:
        exclude = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def prepare_data(
    df: pd.DataFrame,
    sequence_length: int = 30,
    test_ratio: float = 0.2,
    target_col: str = 'Close',
    scaler_save_path: str = 'models/'
) -> dict:
    """
    Full preprocessing pipeline: scale → sequence → split.

    Args:
        df: Feature-engineered DataFrame
        sequence_length: LSTM input window size
        test_ratio: Fraction for test set
        target_col: Column to predict
        scaler_save_path: Where to persist scalers

    Returns:
        Dict containing X_train, X_test, y_train, y_test,
        feature_scaler, target_scaler, feature_cols
    """
    print("[Preprocessing] Starting data preparation pipeline...")

    feature_cols = get_feature_columns(df)
    print(f"[Preprocessing] Using {len(feature_cols)} features.")

    scaled_features, scaled_target, feat_sc, tgt_sc = scale_features(
        df, feature_cols, target_col, scaler_save_path
    )

    X, y = create_sequences(scaled_features, scaled_target, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_ratio)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_scaler': feat_sc,
        'target_scaler': tgt_sc,
        'feature_cols': feature_cols,
        'sequence_length': sequence_length,
        'n_features': X_train.shape[2],
    }
