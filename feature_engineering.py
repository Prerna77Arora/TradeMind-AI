"""
feature_engineering.py
-----------------------
Generates technical indicators and statistical features from OHLCV data.

Feature categories:
  - Trend indicators (SMA, EMA, MACD): Identify the prevailing direction.
  - Momentum indicators (RSI, Price Momentum): Measure the speed of price change.
  - Volatility indicators (Bollinger Bands, Rolling Std, ATR): Quantify price spread.
  - Volume indicators (OBV, Volume SMA): Confirm price moves with volume.
  - Statistical features (Daily Returns, Rolling Mean/Std): Provide ML-friendly signals.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  Trend Indicators
# ─────────────────────────────────────────────

def add_sma(df: pd.DataFrame, windows: list = [10, 20, 50]) -> pd.DataFrame:
    """
    Simple Moving Average: arithmetic mean over a rolling window.
    Helps smooth noise and identify the trend direction.
    """
    for w in windows:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
    return df


def add_ema(df: pd.DataFrame, spans: list = [12, 26]) -> pd.DataFrame:
    """
    Exponential Moving Average: gives more weight to recent prices.
    Reacts faster to price changes than SMA, useful for short-term trends.
    """
    for s in spans:
        df[f'EMA_{s}'] = df['Close'].ewm(span=s, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence):
      - MACD Line = EMA(fast) − EMA(slow)
      - Signal Line = EMA(MACD Line, signal period)
      - Histogram = MACD − Signal
    Crossovers between MACD and Signal generate buy/sell signals.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    return df


# ─────────────────────────────────────────────
#  Momentum Indicators
# ─────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (RSI):
      RSI = 100 − 100 / (1 + RS)  where RS = avg gain / avg loss over `period` days.
    Values above 70 indicate overbought; below 30 indicate oversold.
    """
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_price_momentum(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Price Momentum: percentage change over N days.
    Captures the rate and direction of recent price movement.
    """
    for p in periods:
        df[f'Momentum_{p}'] = df['Close'].pct_change(periods=p) * 100
    return df


# ─────────────────────────────────────────────
#  Volatility Indicators
# ─────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: price envelope around a SMA ± N standard deviations.
      - Upper Band = SMA + num_std × σ
      - Lower Band = SMA − num_std × σ
      - Width (bandwidth) and %B (price position within bands) are derived signals.
    Useful for mean-reversion and volatility breakout strategies.
    """
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = sma + (num_std * std)
    df['BB_Lower'] = sma - (num_std * std)
    df['BB_Middle'] = sma
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range (ATR): average of the true range over N days.
    True Range = max(High−Low, |High−PrevClose|, |Low−PrevClose|)
    ATR measures market volatility independent of direction.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.ewm(span=period, adjust=False).mean()
    return df


# ─────────────────────────────────────────────
#  Volume Indicators
# ─────────────────────────────────────────────

def add_volume_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume Simple Moving Average: baseline for abnormal volume detection."""
    df['Volume_SMA'] = df['Volume'].rolling(window=window).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    On-Balance Volume (OBV): cumulative volume reflecting buying/selling pressure.
    Rising OBV with rising price confirms an uptrend.
    """
    direction = np.sign(df['Close'].diff()).fillna(0)
    df['OBV'] = (direction * df['Volume']).cumsum()
    return df


# ─────────────────────────────────────────────
#  Statistical Features
# ─────────────────────────────────────────────

def add_statistical_features(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Add rolling statistical features:
      - Daily returns: percentage change day-over-day
      - Rolling mean and std: local price distribution
      - Volatility: rolling std of daily returns (annualised)
    """
    df['Daily_Return'] = df['Close'].pct_change()

    for w in windows:
        df[f'Rolling_Mean_{w}'] = df['Close'].rolling(w).mean()
        df[f'Rolling_Std_{w}'] = df['Close'].rolling(w).std()
        df[f'Volatility_{w}'] = df['Daily_Return'].rolling(w).std() * np.sqrt(252)

    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candlestick-derived features:
      - Body: |Close − Open|
      - Upper shadow: High − max(Open, Close)
      - Lower shadow: min(Open, Close) − Low
    These capture intraday sentiment.
    """
    df['Candle_Body'] = (df['Close'] - df['Open']).abs()
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    return df


# ─────────────────────────────────────────────
#  Master function
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a cleaned OHLCV DataFrame.

    Returns:
        DataFrame with all technical and statistical features added.
        Rows with NaN values (from rolling windows) are dropped.
    """
    print("[FeatureEngineering] Computing features...")
    df = df.copy()

    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_price_momentum(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_sma(df)
    df = add_obv(df)
    df = add_statistical_features(df)
    df = add_candle_features(df)

    rows_before = len(df)
    df.dropna(inplace=True)
    print(f"[FeatureEngineering] Dropped {rows_before - len(df)} NaN rows. "
          f"Feature count: {df.shape[1]}. Final rows: {len(df)}")
    return df
