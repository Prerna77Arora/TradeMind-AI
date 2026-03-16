"""
model_evaluation.py
--------------------
Evaluates the trained LSTM model using financial and statistical metrics.

Metrics:
  - MAE  (Mean Absolute Error): Average absolute prediction error in price units.
  - RMSE (Root Mean Squared Error): Penalises large errors more than MAE.
  - R²   (Coefficient of Determination): Proportion of variance explained (1 = perfect).
  - Directional Accuracy: % of days where the model correctly predicted up/down movement.
    This is the most actionable metric — a model can have moderate RMSE but high
    directional accuracy, making it useful for trading signals.

Target: Directional Accuracy ≥ 65–70%
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_scale(scaled_array: np.ndarray, scaler) -> np.ndarray:
    """
    Reverse MinMax scaling to get prices back in original units.

    Args:
        scaled_array: 1D array of scaled predictions/actuals
        scaler: Fitted MinMaxScaler for the target column

    Returns:
        1D array in original price scale
    """
    return scaler.inverse_transform(scaled_array.reshape(-1, 1)).flatten()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics.

    Args:
        y_true: Actual prices (original scale)
        y_pred: Predicted prices (original scale)

    Returns:
        Dict with MAE, RMSE, R2, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — guard against division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy = (# correct direction calls) / (total predictions)

    A direction call is correct when:
      (y_pred[t] > y_pred[t-1]) == (y_true[t] > y_true[t-1])

    Args:
        y_true: Actual prices in original scale
        y_pred: Predicted prices in original scale

    Returns:
        Directional accuracy as a percentage (0–100)
    """
    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))
    correct = np.sum(actual_direction == predicted_direction)
    return (correct / len(actual_direction)) * 100


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler,
    verbose: bool = True
) -> dict:
    """
    Run full model evaluation on the test set.

    Args:
        model: Trained Keras model
        X_test: Test input sequences
        y_test: Test targets (scaled)
        target_scaler: Fitted scaler for the target column
        verbose: Print evaluation results

    Returns:
        Dict with all metrics and arrays of actual/predicted prices
    """
    # Generate and inverse-transform predictions
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = inverse_scale(y_pred_scaled, target_scaler)
    y_actual = inverse_scale(y_test, target_scaler)

    metrics = compute_metrics(y_actual, y_pred)
    dir_acc = directional_accuracy(y_actual, y_pred)
    metrics['Directional_Accuracy'] = dir_acc

    if verbose:
        print("\n" + "═" * 45)
        print("  MODEL EVALUATION REPORT")
        print("═" * 45)
        print(f"  MAE  : {metrics['MAE']:.4f}")
        print(f"  RMSE : {metrics['RMSE']:.4f}")
        print(f"  R²   : {metrics['R2']:.4f}")
        print(f"  MAPE : {metrics['MAPE']:.2f}%")
        print(f"  Directional Accuracy: {dir_acc:.2f}%", end="")
        if dir_acc >= 65:
            print("  ✅ Target met (≥65%)")
        else:
            print("  ⚠️  Below target (<65%)")
        print("═" * 45 + "\n")

    return {
        **metrics,
        'y_actual': y_actual,
        'y_pred': y_pred
    }


def walk_forward_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    target_scaler,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) validation to assess model stability over time.

    Args:
        model: Trained Keras model (used for prediction only, not retrained)
        X, y: Full sequence arrays
        target_scaler: Fitted scaler for inverse transform
        n_splits: Number of folds

    Returns:
        DataFrame with per-fold metrics
    """
    fold_size = len(X) // (n_splits + 1)
    results = []

    for i in range(1, n_splits + 1):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        X_test_fold = X[test_start:test_end]
        y_test_fold = y[test_start:test_end]

        if len(X_test_fold) == 0:
            continue

        y_pred_scaled = model.predict(X_test_fold, verbose=0).flatten()
        y_pred = inverse_scale(y_pred_scaled, target_scaler)
        y_actual = inverse_scale(y_test_fold, target_scaler)

        metrics = compute_metrics(y_actual, y_pred)
        metrics['Directional_Accuracy'] = directional_accuracy(y_actual, y_pred)
        metrics['Fold'] = i
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index('Fold')
    print("\n[Evaluation] Walk-Forward Validation Results:")
    print(df_results.round(4))
    return df_results
