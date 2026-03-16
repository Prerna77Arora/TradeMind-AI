"""
financial_filters.py
---------------------
Pre-recommendation filters to screen stocks against financial criteria.

Filters prevent the model from recommending stocks that pass the ML threshold
but fail fundamental or technical sanity checks.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  Trend Filter
# ─────────────────────────────────────────────

def trend_filter(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> dict:
    """
    Identify the trend direction using moving average crossover.

    Rules:
      STRONG_UPTREND  : Price > SMA20 > SMA50 (all aligned)
      UPTREND         : Price > SMA50 but SMA20 ≤ SMA50
      DOWNTREND       : Price < SMA50
      SIDEWAYS        : SMA20 ≈ SMA50 within threshold

    Args:
        df: Feature-engineered DataFrame (must include 'Close', SMA columns)
        short_window: Fast SMA period
        long_window: Slow SMA period

    Returns:
        Dict with trend label and supporting values
    """
    current_price = df['Close'].iloc[-1]
    sma_short_col = f'SMA_{short_window}'
    sma_long_col = f'SMA_{long_window}'

    sma_short = df[sma_short_col].iloc[-1] if sma_short_col in df.columns else df['Close'].rolling(short_window).mean().iloc[-1]
    sma_long = df[sma_long_col].iloc[-1] if sma_long_col in df.columns else df['Close'].rolling(long_window).mean().iloc[-1]

    if current_price > sma_short > sma_long:
        trend = 'STRONG_UPTREND'
    elif current_price > sma_long and sma_short > sma_long:
        trend = 'UPTREND'
    elif current_price < sma_long and sma_short < sma_long:
        trend = 'DOWNTREND'
    elif current_price < sma_short < sma_long:
        trend = 'STRONG_DOWNTREND'
    else:
        trend = 'SIDEWAYS'

    # Check if price is above 50-day SMA (bullish bias)
    bullish = current_price > sma_long

    return {
        'trend': trend,
        'bullish': bullish,
        'current_price': current_price,
        f'SMA_{short_window}': sma_short,
        f'SMA_{long_window}': sma_long,
        'passes_trend_filter': bullish
    }


def macd_signal_filter(df: pd.DataFrame) -> dict:
    """
    Check if MACD is signalling a bullish crossover.
    MACD > Signal line = bullish momentum.
    """
    if 'MACD' not in df.columns:
        return {'macd_bullish': None, 'passes_macd_filter': False}

    macd = df['MACD'].iloc[-1]
    signal = df['MACD_Signal'].iloc[-1]
    histogram = df['MACD_Histogram'].iloc[-1]

    # Previous histogram to detect crossover
    prev_histogram = df['MACD_Histogram'].iloc[-2] if len(df) >= 2 else 0

    bullish_crossover = (histogram > 0) and (prev_histogram <= 0)
    macd_bullish = macd > signal

    return {
        'macd': round(macd, 4),
        'macd_signal': round(signal, 4),
        'macd_histogram': round(histogram, 4),
        'macd_bullish': macd_bullish,
        'bullish_crossover': bullish_crossover,
        'passes_macd_filter': macd_bullish
    }


def rsi_filter(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> dict:
    """
    RSI filter:
      - RSI < oversold threshold → potential buy (oversold)
      - RSI > overbought threshold → avoid (overbought)
      - 30 ≤ RSI ≤ 70 → neutral zone

    Args:
        df: DataFrame with 'RSI' column
        oversold: RSI level below which stock is considered oversold
        overbought: RSI level above which stock is considered overbought
    """
    if 'RSI' not in df.columns:
        return {'rsi': None, 'rsi_signal': 'N/A', 'passes_rsi_filter': True}

    rsi = df['RSI'].iloc[-1]

    if rsi < oversold:
        rsi_signal = 'OVERSOLD'
        passes = True   # Potential buying opportunity
    elif rsi > overbought:
        rsi_signal = 'OVERBOUGHT'
        passes = False  # Risk of reversal
    else:
        rsi_signal = 'NEUTRAL'
        passes = True

    return {
        'rsi': round(rsi, 2),
        'rsi_signal': rsi_signal,
        'passes_rsi_filter': passes
    }


# ─────────────────────────────────────────────
#  Fundamental Filter
# ─────────────────────────────────────────────

def pe_ratio_filter(
    pe_ratio: float,
    sector_avg_pe: float = 20.0,
    max_pe: float = 50.0
) -> dict:
    """
    P/E Ratio Filter:
      - pe_ratio < sector_avg_pe × 0.8 → undervalued (attractive)
      - pe_ratio > max_pe → overvalued (risky for value investors)
      - None / negative → earnings are negative, flag cautiously

    Args:
        pe_ratio: Trailing P/E ratio from company info
        sector_avg_pe: Approximate sector average P/E
        max_pe: Maximum acceptable P/E

    Returns:
        Dict with valuation assessment
    """
    if pe_ratio is None or pe_ratio <= 0:
        return {
            'pe_ratio': pe_ratio,
            'valuation': 'NEGATIVE_EARNINGS',
            'passes_pe_filter': False
        }

    if pe_ratio < sector_avg_pe * 0.8:
        valuation = 'UNDERVALUED'
        passes = True
    elif pe_ratio > max_pe:
        valuation = 'OVERVALUED'
        passes = False
    else:
        valuation = 'FAIRLY_VALUED'
        passes = True

    return {
        'pe_ratio': round(pe_ratio, 2),
        'valuation': valuation,
        'passes_pe_filter': passes
    }


def volume_filter(df: pd.DataFrame, min_ratio: float = 0.8) -> dict:
    """
    Volume confirmation filter: is current volume near/above average?
    Low volume breakouts are less reliable.

    Args:
        df: DataFrame with 'Volume' and 'Volume_SMA' columns
        min_ratio: Minimum acceptable Volume/Volume_SMA ratio

    Returns:
        Dict with volume assessment
    """
    if 'Volume_SMA' not in df.columns:
        return {'volume_ratio': None, 'passes_volume_filter': True}

    current_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume_SMA'].iloc[-1]
    ratio = current_volume / avg_volume if avg_volume > 0 else 0

    return {
        'current_volume': int(current_volume),
        'avg_volume': int(avg_volume),
        'volume_ratio': round(ratio, 2),
        'passes_volume_filter': ratio >= min_ratio
    }


def bollinger_filter(df: pd.DataFrame) -> dict:
    """
    Bollinger Band position filter.
    BB_PctB < 0.2 → price near lower band (potential bounce).
    BB_PctB > 0.8 → price near upper band (potential resistance).
    """
    if 'BB_PctB' not in df.columns:
        return {'bb_pctb': None, 'bb_signal': 'N/A', 'passes_bb_filter': True}

    pctb = df['BB_PctB'].iloc[-1]
    width = df['BB_Width'].iloc[-1]

    if pctb < 0.2:
        signal = 'NEAR_LOWER_BAND'
        passes = True
    elif pctb > 0.8:
        signal = 'NEAR_UPPER_BAND'
        passes = False
    else:
        signal = 'MID_BAND'
        passes = True

    return {
        'bb_pctb': round(pctb, 3),
        'bb_width': round(width, 4),
        'bb_signal': signal,
        'passes_bb_filter': passes
    }


# ─────────────────────────────────────────────
#  Master filter
# ─────────────────────────────────────────────

def apply_all_filters(
    df: pd.DataFrame,
    pe_ratio: float = None,
    sector_avg_pe: float = 20.0
) -> dict:
    """
    Apply all filters and return a composite pass/fail decision.

    Args:
        df: Feature-engineered DataFrame
        pe_ratio: Company P/E ratio from fundamental data
        sector_avg_pe: Sector benchmark P/E

    Returns:
        Comprehensive filter results dict with composite 'passes_all' flag
    """
    results = {}
    results.update(trend_filter(df))
    results.update(macd_signal_filter(df))
    results.update(rsi_filter(df))
    results.update(volume_filter(df))
    results.update(bollinger_filter(df))
    results.update(pe_ratio_filter(pe_ratio or 0, sector_avg_pe))

    # A stock must pass trend and RSI filters; PE filter is optional for traders
    essential_passes = [
        results.get('passes_trend_filter', False),
        results.get('passes_rsi_filter', True),
        results.get('passes_macd_filter', False),
    ]
    results['passes_all'] = all(essential_passes)
    results['filter_score'] = sum([
        results.get('passes_trend_filter', False),
        results.get('passes_macd_filter', False),
        results.get('passes_rsi_filter', False),
        results.get('passes_volume_filter', False),
        results.get('passes_bb_filter', False),
        results.get('passes_pe_filter', False),
    ])  # Out of 6

    return results
