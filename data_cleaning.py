"""
data_cleaning.py
----------------
Cleans raw financial time-series data.

Why each step matters:
- Forward fill: Markets close on weekends/holidays, creating gaps. FFill propagates
  the last known price forward to maintain a continuous series.
- Duplicate removal: Duplicate rows can skew rolling computations and inflated training.
- Chronological sort: LSTM models are order-sensitive; wrong order = wrong sequences.
- Outlier handling: Extreme values (e.g., data-feed spikes) distort normalization.
- Timestamp consistency: Ensures uniform daily frequency for feature alignment.
"""

import pandas as pd
import numpy as np
from scipy import stats


def forward_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values by propagating the last valid observation forward.
    Remaining NaNs at the start are back-filled.
    """
    df = df.ffill().bfill()
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  [Warning] {missing} NaN values remain after fill.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate index entries, keeping the first occurrence."""
    before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    removed = before - len(df)
    if removed:
        print(f"  [Cleaning] Removed {removed} duplicate rows.")
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame is sorted by date ascending."""
    return df.sort_index(ascending=True)


def remove_outliers_iqr(df: pd.DataFrame, columns: list = None, factor: float = 3.0) -> pd.DataFrame:
    """
    Detect and cap outliers in specified numeric columns using the IQR method.
    Values beyond factor × IQR from Q1/Q3 are clipped (not dropped) to preserve
    the time-series continuity.

    Args:
        df: Input DataFrame
        columns: Columns to check (defaults to all numeric)
        factor: IQR multiplier for outlier threshold

    Returns:
        DataFrame with outliers clipped
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if clipped:
            print(f"  [Outlier] Clipped {clipped} values in '{col}'.")
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def ensure_business_day_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex DataFrame to a business-day (Mon–Fri) range and forward-fill any gaps.
    This normalises market data that might skip days due to API quirks.
    """
    bday_range = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(bday_range)
    df.index.name = 'Date'
    df = df.ffill().bfill()
    return df


def clean_stock_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on raw OHLCV data.

    Pipeline order:
        1. Remove duplicates
        2. Sort chronologically
        3. Ensure business-day frequency
        4. Forward-fill missing values
        5. (Optional) Clip outliers via IQR

    Args:
        df: Raw DataFrame from data_collection
        remove_outliers: Whether to apply IQR outlier clipping

    Returns:
        Cleaned DataFrame
    """
    print("[DataCleaning] Starting cleaning pipeline...")
    df = remove_duplicates(df)
    df = sort_chronologically(df)
    df = ensure_business_day_frequency(df)
    df = forward_fill_missing(df)

    if remove_outliers:
        # Only clip price columns; Volume has a different distribution
        df = remove_outliers_iqr(df, columns=['Open', 'High', 'Low', 'Close'])

    # Validate that no negative prices survived
    for col in ['Open', 'High', 'Low', 'Close']:
        neg = (df[col] < 0).sum()
        if neg:
            print(f"  [Warning] {neg} negative values found in '{col}' — replacing with NaN and forward-filling.")
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df[col].ffill()

    print(f"[DataCleaning] Done. Final shape: {df.shape}")
    return df
