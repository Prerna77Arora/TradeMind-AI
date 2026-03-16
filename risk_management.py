"""
risk_management.py
------------------
Implements risk management techniques for trade and portfolio level controls.

Functions cover:
  - Stop Loss placement (volatility-adjusted)
  - Take Profit targets
  - Maximum Drawdown calculation
  - Sharpe Ratio
  - Position sizing (Kelly Criterion and fixed-fraction)
  - Portfolio risk assessment
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Trade Level Risk
# ─────────────────────────────────────────────

def calculate_stop_loss(
    entry_price: float,
    atr: float,
    multiplier: float = 2.0
) -> float:
    """
    Volatility-adjusted stop loss.

    Stop Loss = Entry Price − (ATR × multiplier)

    Using ATR ensures the stop is wide enough to avoid being triggered by
    normal daily volatility, while still limiting downside.

    Args:
        entry_price: Trade entry price
        atr: Average True Range (current ATR value)
        multiplier: ATR multiplier (higher = wider stop, more risk tolerance)

    Returns:
        Stop loss price
    """
    stop_loss = entry_price - (atr * multiplier)
    return round(max(stop_loss, 0), 4)  # Price cannot be negative


def calculate_take_profit(
    entry_price: float,
    stop_loss: float,
    risk_reward_ratio: float = 2.0,
    predicted_price: float = None
) -> float:
    """
    Take profit based on Risk:Reward ratio or predicted price (whichever is closer).

    Take Profit (RR-based) = Entry + (Entry − Stop Loss) × risk_reward_ratio
    Take Profit (predicted) = model's predicted next-day price

    Args:
        entry_price: Trade entry price
        stop_loss: Stop loss level
        risk_reward_ratio: Minimum desired R:R (default 2:1)
        predicted_price: Optional model-predicted target price

    Returns:
        Take profit price
    """
    risk_per_share = entry_price - stop_loss
    rr_target = entry_price + (risk_per_share * risk_reward_ratio)

    if predicted_price is not None:
        # Use the lower of RR target and predicted price as a conservative target
        return round(min(rr_target, predicted_price) if predicted_price > entry_price else rr_target, 4)

    return round(rr_target, 4)


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    risk_per_trade_pct: float = 1.0
) -> dict:
    """
    Fixed-fraction position sizing.

    Risk = capital × risk_per_trade_pct / 100
    Shares = Risk / (Entry − Stop Loss)

    This ensures each trade risks at most risk_per_trade_pct% of total capital.

    Args:
        capital: Total available capital
        entry_price: Trade entry price
        stop_loss: Stop loss level
        risk_per_trade_pct: Percentage of capital to risk per trade (1–2% recommended)

    Returns:
        Dict with position size details
    """
    if entry_price <= stop_loss:
        return {'error': 'Entry price must be above stop loss.', 'shares': 0, 'investment': 0}

    risk_amount = capital * (risk_per_trade_pct / 100)
    risk_per_share = entry_price - stop_loss
    shares = int(risk_amount / risk_per_share)
    total_investment = shares * entry_price

    # Guard against over-investing
    if total_investment > capital:
        shares = int(capital / entry_price)
        total_investment = shares * entry_price

    return {
        'shares': shares,
        'investment': round(total_investment, 2),
        'risk_amount': round(risk_amount, 2),
        'risk_per_share': round(risk_per_share, 4),
        'pct_of_capital': round((total_investment / capital) * 100, 2)
    }


# ─────────────────────────────────────────────
#  Portfolio Level Risk
# ─────────────────────────────────────────────

def calculate_max_drawdown(portfolio_values: pd.Series) -> dict:
    """
    Maximum Drawdown: largest peak-to-trough decline in portfolio value.

    MDD = (Trough Value − Peak Value) / Peak Value × 100

    Args:
        portfolio_values: Time series of portfolio values (or cumulative returns)

    Returns:
        Dict with MDD percentage, peak date, trough date
    """
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max

    max_drawdown_pct = drawdown.min() * 100
    trough_date = drawdown.idxmin()
    peak_date = portfolio_values[:trough_date].idxmax() if trough_date else None

    return {
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'trough_date': trough_date,
        'peak_date': peak_date,
        'drawdown_series': drawdown
    }


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    annualise: bool = True
) -> float:
    """
    Sharpe Ratio = (Mean Return − Risk-Free Rate) / Std Dev of Returns

    Annualised by multiplying by √252 (trading days per year).

    Args:
        returns: Daily returns series (percentage or decimal)
        risk_free_rate: Annual risk-free rate (default 5% ≈ Indian T-bill rate)
        annualise: Whether to annualise the ratio

    Returns:
        Sharpe Ratio (higher is better; >1 is acceptable, >2 is excellent)
    """
    if returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = excess_returns.mean() / excess_returns.std()

    if annualise:
        sharpe *= np.sqrt(252)

    return round(sharpe, 4)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05
) -> float:
    """
    Sortino Ratio: like Sharpe but penalises only downside volatility.

    Sortino = (Mean Return − RF) / Downside Std Dev × √252
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    downside = excess_returns[excess_returns < 0]

    if downside.std() == 0:
        return 0.0

    sortino = (excess_returns.mean() / downside.std()) * np.sqrt(252)
    return round(sortino, 4)


def portfolio_risk_assessment(
    df: pd.DataFrame,
    predicted_price: float,
    entry_price: float,
    capital: float,
    risk_tolerance: str = 'medium'
) -> dict:
    """
    Generate a full risk assessment for a single stock trade.

    Args:
        df: Feature-engineered DataFrame
        predicted_price: Model's predicted next price
        entry_price: Current market price (entry point)
        capital: Total investment capital
        risk_tolerance: 'low' | 'medium' | 'high'

    Returns:
        Comprehensive risk metrics dict
    """
    # ATR-based stop loss multiplier varies with risk tolerance
    sl_multipliers = {'low': 1.5, 'medium': 2.0, 'high': 3.0}
    risk_pct = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
    rr_ratios = {'low': 3.0, 'medium': 2.0, 'high': 1.5}

    multiplier = sl_multipliers.get(risk_tolerance, 2.0)
    trade_risk_pct = risk_pct.get(risk_tolerance, 1.0)
    rr = rr_ratios.get(risk_tolerance, 2.0)

    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else entry_price * 0.02
    stop_loss = calculate_stop_loss(entry_price, atr, multiplier)
    take_profit = calculate_take_profit(entry_price, stop_loss, rr, predicted_price)
    position = calculate_position_size(capital, entry_price, stop_loss, trade_risk_pct)

    # Historical returns-based metrics
    daily_returns = df['Close'].pct_change().dropna()
    sharpe = calculate_sharpe_ratio(daily_returns)
    sortino = calculate_sortino_ratio(daily_returns)

    # Cumulative portfolio value (simulated, equal-weight)
    cum_returns = (1 + daily_returns).cumprod() * capital
    mdd = calculate_max_drawdown(cum_returns)

    volatility_pct = daily_returns.std() * np.sqrt(252) * 100  # Annualised volatility

    return {
        'entry_price': round(entry_price, 4),
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward_ratio': round((take_profit - entry_price) / max(entry_price - stop_loss, 0.01), 2),
        'atr': round(atr, 4),
        'position_size': position,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': mdd['max_drawdown_pct'],
        'annualised_volatility_pct': round(volatility_pct, 2),
        'risk_tolerance_used': risk_tolerance
    }
