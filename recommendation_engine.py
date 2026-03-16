"""
recommendation_engine.py
-------------------------
Combines model predictions, technical filters, and investor profile
to generate personalised BUY / SELL / HOLD recommendations.

Decision logic:
  1. Model predicts a positive price move AND
  2. Key financial filters pass AND
  3. Investor's risk profile allows the position size AND
  4. Risk:Reward ratio meets the investor's minimum threshold
  → BUY

  1. Model predicts a negative price move OR
  2. Critical filters fail (e.g., strong downtrend)
  → SELL (if currently holding) or AVOID

  Otherwise → HOLD / MONITOR
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from investor_profile import InvestorProfile
from financial_filters import apply_all_filters
from risk_management import portfolio_risk_assessment


@dataclass
class Recommendation:
    """Structured recommendation output."""
    ticker: str
    action: str                   # BUY | SELL | HOLD
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    stop_loss: float
    take_profit: float
    suggested_investment: float
    shares: int
    risk_reward_ratio: float
    confidence: str               # HIGH | MEDIUM | LOW
    strategy_summary: str
    filter_results: dict
    risk_metrics: dict

    def display(self, currency: str = '₹') -> str:
        """Format the recommendation as a readable report."""
        action_icons = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
        icon = action_icons.get(self.action, '⚪')

        lines = [
            "",
            "╔══════════════════════════════════════════════════╗",
            f"  {icon}  RECOMMENDATION: {self.action}  —  {self.ticker}",
            "══════════════════════════════════════════════════",
            f"  Current Price     : {currency}{self.current_price:,.2f}",
            f"  Predicted Price   : {currency}{self.predicted_price:,.2f}  "
            f"({self.predicted_change_pct:+.2f}%)",
            f"  Confidence        : {self.confidence}",
            "",
            f"  Stop Loss         : {currency}{self.stop_loss:,.2f}",
            f"  Take Profit       : {currency}{self.take_profit:,.2f}",
            f"  Risk:Reward       : 1 : {self.risk_reward_ratio:.1f}",
            "",
            f"  Suggested Amount  : {currency}{self.suggested_investment:,.2f}",
            f"  Shares            : {self.shares}",
            "",
            f"  Sharpe Ratio      : {self.risk_metrics.get('sharpe_ratio', 'N/A')}",
            f"  Max Drawdown      : {self.risk_metrics.get('max_drawdown_pct', 'N/A')}%",
            f"  Annual Volatility : {self.risk_metrics.get('annualised_volatility_pct', 'N/A')}%",
            "",
            "  STRATEGY SUMMARY",
            "  ─────────────────",
            f"  {self.strategy_summary}",
            "╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


def _determine_confidence(
    predicted_change_pct: float,
    filter_score: int,
    directional_accuracy: float
) -> str:
    """
    Confidence scoring:
      - HIGH   : Strong move prediction + most filters pass + good model accuracy
      - MEDIUM : Moderate signals
      - LOW    : Weak prediction or poor filter alignment
    """
    score = 0

    if abs(predicted_change_pct) > 2.0:
        score += 2
    elif abs(predicted_change_pct) > 0.5:
        score += 1

    score += min(filter_score, 3)  # Up to 3 points from filters

    if directional_accuracy >= 65:
        score += 2
    elif directional_accuracy >= 55:
        score += 1

    if score >= 6:
        return 'HIGH'
    elif score >= 3:
        return 'MEDIUM'
    return 'LOW'


def _determine_action(
    predicted_change_pct: float,
    filter_results: dict,
    profile: InvestorProfile,
    risk_metrics: dict
) -> str:
    """
    Core decision logic mapping signals to a trading action.
    """
    trend = filter_results.get('trend', 'SIDEWAYS')
    rr = risk_metrics.get('risk_reward_ratio', 0)
    min_rr = profile.profile_params['min_rr_ratio']

    bullish_signal = (
        predicted_change_pct > 0.3
        and filter_results.get('passes_trend_filter', False)
        and filter_results.get('passes_rsi_filter', False)
        and rr >= min_rr
    )

    bearish_signal = (
        predicted_change_pct < -0.3
        or trend in ('STRONG_DOWNTREND', 'DOWNTREND')
        and not filter_results.get('passes_trend_filter', True)
    )

    if bullish_signal:
        return 'BUY'
    elif bearish_signal:
        return 'SELL'
    else:
        return 'HOLD'


def _build_strategy_summary(
    action: str,
    profile: InvestorProfile,
    filter_results: dict,
    risk_metrics: dict,
    predicted_change_pct: float
) -> str:
    """Generate a natural-language strategy explanation."""
    trend = filter_results.get('trend', 'N/A')
    rsi = filter_results.get('rsi', 'N/A')
    macd_bull = filter_results.get('macd_bullish', False)
    rr = risk_metrics.get('risk_reward_ratio', 0)

    if action == 'BUY':
        macd_str = "MACD is confirming bullish momentum. " if macd_bull else ""
        return (
            f"Based on your {profile.profile_params['label']} profile and "
            f"{profile.investment_horizon_years}-year horizon, this is a "
            f"{profile.strategy.replace('_', ' ')} opportunity. "
            f"The model forecasts a {predicted_change_pct:+.2f}% price move. "
            f"Trend is {trend}. RSI at {rsi}. {macd_str}"
            f"Risk:Reward of {rr:.1f}:1 meets your minimum of "
            f"{profile.profile_params['min_rr_ratio']}:1. "
            f"Position sized to risk {profile.profile_params['risk_per_trade_pct']}% of capital."
        )
    elif action == 'SELL':
        return (
            f"The model projects a {predicted_change_pct:+.2f}% decline and the trend "
            f"is {trend}. Exit or avoid this position to protect capital. "
            f"Consider reviewing when trend improves above 50-day SMA."
        )
    else:
        return (
            f"Signals are mixed. Predicted move of {predicted_change_pct:+.2f}% is "
            f"insufficient for a conviction trade. Hold existing positions or wait "
            f"for a clearer trend before entering. Trend is currently {trend}."
        )


def generate_recommendation(
    ticker: str,
    df: pd.DataFrame,
    predicted_price: float,
    profile: InvestorProfile,
    pe_ratio: float = None,
    sector_avg_pe: float = 20.0,
    directional_accuracy: float = 60.0,
    model_eval_metrics: dict = None
) -> Recommendation:
    """
    Main function to generate a personalised trading recommendation.

    Args:
        ticker: Stock symbol
        df: Feature-engineered DataFrame
        predicted_price: Model output (original price scale)
        profile: InvestorProfile instance
        pe_ratio: Company trailing P/E
        sector_avg_pe: Sector benchmark P/E
        directional_accuracy: From model evaluation (%)
        model_eval_metrics: Optional dict of evaluation metrics for display

    Returns:
        Recommendation dataclass instance
    """
    current_price = df['Close'].iloc[-1]
    predicted_change_pct = ((predicted_price - current_price) / current_price) * 100

    # Apply all financial filters
    filter_results = apply_all_filters(df, pe_ratio, sector_avg_pe)

    # Risk metrics (volatility-adjusted stop/take-profit, Sharpe, MDD)
    risk_metrics = portfolio_risk_assessment(
        df=df,
        predicted_price=predicted_price,
        entry_price=current_price,
        capital=profile.investment_amount,
        risk_tolerance=profile.risk_tolerance
    )

    # Decide action
    action = _determine_action(predicted_change_pct, filter_results, profile, risk_metrics)

    # Determine investment amount based on profile constraints
    pos = risk_metrics['position_size']
    suggested_investment = min(
        pos['investment'],
        profile.max_single_stock_investment
    )
    shares = int(suggested_investment / current_price)

    # Confidence level
    confidence = _determine_confidence(
        predicted_change_pct,
        filter_results.get('filter_score', 0),
        directional_accuracy
    )

    # Strategy summary
    strategy_summary = _build_strategy_summary(
        action, profile, filter_results, risk_metrics, predicted_change_pct
    )

    return Recommendation(
        ticker=ticker,
        action=action,
        current_price=round(current_price, 2),
        predicted_price=round(predicted_price, 2),
        predicted_change_pct=round(predicted_change_pct, 2),
        stop_loss=risk_metrics['stop_loss'],
        take_profit=risk_metrics['take_profit'],
        suggested_investment=round(suggested_investment, 2),
        shares=shares,
        risk_reward_ratio=risk_metrics['risk_reward_ratio'],
        confidence=confidence,
        strategy_summary=strategy_summary,
        filter_results=filter_results,
        risk_metrics=risk_metrics
    )


def generate_portfolio_recommendations(
    tickers: list,
    dataframes: dict,
    predicted_prices: dict,
    profile: InvestorProfile,
    company_info: dict = None,
    directional_accuracy: float = 60.0
) -> list:
    """
    Generate recommendations for multiple tickers and rank by confidence + action.

    Args:
        tickers: List of stock symbols
        dataframes: Dict mapping ticker → feature-engineered DataFrame
        predicted_prices: Dict mapping ticker → predicted price
        profile: InvestorProfile
        company_info: Dict mapping ticker → fundamentals (optional)
        directional_accuracy: Overall model directional accuracy

    Returns:
        List of Recommendation objects, sorted: BUY first, then HOLD, then SELL
    """
    recommendations = []
    action_order = {'BUY': 0, 'HOLD': 1, 'SELL': 2}

    for ticker in tickers:
        if ticker not in dataframes or ticker not in predicted_prices:
            continue
        df = dataframes[ticker]
        pred = predicted_prices[ticker]
        info = (company_info or {}).get(ticker, {})
        pe = info.get('pe_ratio', None)

        rec = generate_recommendation(
            ticker=ticker,
            df=df,
            predicted_price=pred,
            profile=profile,
            pe_ratio=pe,
            directional_accuracy=directional_accuracy
        )
        recommendations.append(rec)

    recommendations.sort(key=lambda r: (action_order.get(r.action, 3),
                                         -abs(r.predicted_change_pct)))
    return recommendations
