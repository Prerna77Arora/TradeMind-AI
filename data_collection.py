"""
data_collection.py
------------------
Fetches historical stock data from Yahoo Finance using yfinance.
Returns a clean Pandas DataFrame with OHLCV data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    period: str = "2y"
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given stock ticker.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'RELIANCE.NS')
        start_date: Start date as 'YYYY-MM-DD' (overrides period if set)
        end_date: End date as 'YYYY-MM-DD'
        period: yfinance period string ('1y', '2y', '5y') used if start_date is None

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    try:
        stock = yf.Ticker(ticker)

        if start_date:
            end = end_date or datetime.today().strftime('%Y-%m-%d')
            df = stock.history(start=start_date, end=end, auto_adjust=True)
        else:
            df = stock.history(period=period, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

        # Keep only core OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'

        print(f"[DataCollection] Fetched {len(df)} rows for {ticker} "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df

    except Exception as e:
        raise RuntimeError(f"[DataCollection] Failed to fetch data: {e}")


def fetch_multiple_stocks(
    tickers: list,
    period: str = "2y"
) -> dict:
    """
    Fetch data for multiple tickers at once.

    Args:
        tickers: List of ticker symbols
        period: Historical data period

    Returns:
        Dict mapping ticker → DataFrame
    """
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = fetch_stock_data(ticker, period=period)
        except RuntimeError as e:
            print(f"  [Warning] Skipping {ticker}: {e}")
    return result


def get_company_info(ticker: str) -> dict:
    """
    Fetch basic company fundamentals (P/E ratio, market cap, sector, etc.)

    Args:
        ticker: Stock symbol

    Returns:
        Dict with fundamental data
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "pe_ratio": info.get("trailingPE", None),
        "market_cap": info.get("marketCap", None),
        "52w_high": info.get("fiftyTwoWeekHigh", None),
        "52w_low": info.get("fiftyTwoWeekLow", None),
        "dividend_yield": info.get("dividendYield", None),
    }
