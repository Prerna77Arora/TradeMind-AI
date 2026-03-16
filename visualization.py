"""
visualization.py
----------------
Creates interactive Plotly charts for stock analysis:
  - Price history with technical indicators
  - Actual vs Predicted prices
  - Volume bar chart
  - RSI and MACD sub-plots
  - Buy/Sell signal markers
  - Training loss curves
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────
COLORS = {
    'price': '#00D4FF',
    'prediction': '#FF6B35',
    'sma20': '#FFD700',
    'sma50': '#7FFF00',
    'bb_upper': '#FF69B4',
    'bb_lower': '#FF69B4',
    'macd': '#00CED1',
    'signal': '#FF8C00',
    'buy': '#00FF7F',
    'sell': '#FF4500',
    'volume': '#4682B4',
    'bg': '#0d1117',
    'grid': '#21262d',
    'text': '#e6edf3',
}


def plot_price_with_indicators(
    df: pd.DataFrame,
    ticker: str,
    show_bollinger: bool = True,
    show_sma: bool = True,
    show_ema: bool = False,
    save_html: str = None
) -> go.Figure:
    """
    Interactive candlestick chart with optional overlays:
      - Bollinger Bands
      - SMA 20 and SMA 50
      - EMA 12 and EMA 26
      - Volume sub-chart

    Args:
        df: Feature-engineered DataFrame
        ticker: Stock symbol for chart title
        show_bollinger: Include Bollinger Bands
        show_sma: Include SMA lines
        show_ema: Include EMA lines
        save_html: Optional path to save chart as HTML

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} — Price & Indicators", "Volume")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # Bollinger Bands
    if show_bollinger and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            line=dict(color=COLORS['bb_upper'], width=1, dash='dot'),
            name='BB Upper', opacity=0.7
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            line=dict(color=COLORS['bb_lower'], width=1, dash='dot'),
            name='BB Lower', fill='tonexty',
            fillcolor='rgba(255,105,180,0.05)', opacity=0.7
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Middle'],
            line=dict(color='#888888', width=1),
            name='BB Middle', opacity=0.5
        ), row=1, col=1)

    # SMA lines
    if show_sma:
        for col, color in [('SMA_20', COLORS['sma20']), ('SMA_50', COLORS['sma50'])]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    line=dict(color=color, width=1.5),
                    name=col.replace('_', ' ')
                ), row=1, col=1)

    # EMA lines
    if show_ema:
        for col, color in [('EMA_12', '#FF6347'), ('EMA_26', '#9370DB')]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    line=dict(color=color, width=1.5, dash='dash'),
                    name=col.replace('_', ' ')
                ), row=1, col=1)

    # Volume bars
    colors_vol = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume', marker_color=colors_vol, opacity=0.7
    ), row=2, col=1)

    _apply_dark_theme(fig, f"{ticker} — Price Analysis")

    if save_html:
        fig.write_html(save_html)
        print(f"[Visualization] Chart saved to {save_html}")

    return fig


def plot_predictions(
    df: pd.DataFrame,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    ticker: str,
    sequence_length: int = 30,
    save_html: str = None
) -> go.Figure:
    """
    Compare actual vs predicted closing prices on the test set.

    Args:
        df: Original DataFrame (used for date alignment)
        y_actual: Actual prices (original scale)
        y_pred: Model-predicted prices (original scale)
        ticker: Stock symbol
        sequence_length: Used to align dates from the end of the DataFrame
        save_html: Optional save path

    Returns:
        Plotly Figure
    """
    # Align dates to the test set portion
    test_dates = df.index[-(len(y_actual)):]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_dates, y=y_actual,
        name='Actual Price',
        line=dict(color=COLORS['price'], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=y_pred,
        name='Predicted Price',
        line=dict(color=COLORS['prediction'], width=2, dash='dash')
    ))

    # Shade prediction error region
    fig.add_trace(go.Scatter(
        x=list(test_dates) + list(test_dates[::-1]),
        y=list(y_actual) + list(y_pred[::-1]),
        fill='toself',
        fillcolor='rgba(255,107,53,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Error Band',
        showlegend=True
    ))

    _apply_dark_theme(fig, f"{ticker} — Actual vs Predicted Price")

    if save_html:
        fig.write_html(save_html)
    return fig


def plot_rsi_macd(
    df: pd.DataFrame,
    ticker: str,
    save_html: str = None
) -> go.Figure:
    """
    Three-panel chart: Price | RSI | MACD histogram.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=(f"{ticker} — Close Price", "RSI (14)", "MACD")
    )

    # Price
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        line=dict(color=COLORS['price'], width=2), name='Close'
    ), row=1, col=1)

    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            line=dict(color='#9370DB', width=2), name='RSI'
        ), row=2, col=1)
        for level, color in [(70, 'rgba(255,69,0,0.3)'), (30, 'rgba(0,255,127,0.3)')]:
            fig.add_hline(y=level, line_dash='dot', line_color=color, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,69,0,0.05)',
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor='rgba(0,255,127,0.05)',
                      line_width=0, row=2, col=1)

    # MACD
    if 'MACD' in df.columns:
        hist_colors = [COLORS['buy'] if v >= 0 else COLORS['sell']
                       for v in df['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=df.index, y=df['MACD_Histogram'],
            name='MACD Hist', marker_color=hist_colors, opacity=0.6
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            line=dict(color=COLORS['macd'], width=1.5), name='MACD'
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'],
            line=dict(color=COLORS['signal'], width=1.5, dash='dash'),
            name='Signal'
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash='solid', line_color='#555', row=3, col=1)

    _apply_dark_theme(fig, f"{ticker} — RSI & MACD Analysis")

    if save_html:
        fig.write_html(save_html)
    return fig


def plot_buy_sell_signals(
    df: pd.DataFrame,
    ticker: str,
    save_html: str = None
) -> go.Figure:
    """
    Plot buy signals (RSI < 30 + MACD crossover) and sell signals (RSI > 70).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        line=dict(color=COLORS['price'], width=2), name='Close Price'
    ))

    # Buy signals: RSI oversold + MACD histogram turns positive
    if 'RSI' in df.columns and 'MACD_Histogram' in df.columns:
        buy_mask = (df['RSI'] < 35) & (df['MACD_Histogram'] > 0)
        sell_mask = (df['RSI'] > 65) & (df['MACD_Histogram'] < 0)

        if buy_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[buy_mask], y=df['Close'][buy_mask],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color=COLORS['buy'],
                            line=dict(color='white', width=1)),
                name='Buy Signal'
            ))

        if sell_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[sell_mask], y=df['Close'][sell_mask],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color=COLORS['sell'],
                            line=dict(color='white', width=1)),
                name='Sell Signal'
            ))

    _apply_dark_theme(fig, f"{ticker} — Buy/Sell Signals")

    if save_html:
        fig.write_html(save_html)
    return fig


def plot_training_history(history, save_html: str = None) -> go.Figure:
    """
    Plot training and validation loss over epochs.
    """
    epochs = list(range(1, len(history.history['loss']) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history.history['loss'],
        name='Training Loss', line=dict(color=COLORS['price'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history.history['val_loss'],
        name='Validation Loss', line=dict(color=COLORS['prediction'], width=2, dash='dash')
    ))

    _apply_dark_theme(fig, "LSTM Training History — Loss")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="MSE Loss")

    if save_html:
        fig.write_html(save_html)
    return fig


def plot_portfolio_performance(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series = None,
    save_html: str = None
) -> go.Figure:
    """
    Portfolio value over time vs optional benchmark.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_values.index, y=portfolio_values,
        name='Portfolio', line=dict(color=COLORS['price'], width=2),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.07)'
    ))

    if benchmark_values is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_values.index, y=benchmark_values,
            name='Benchmark (Nifty/SPY)', line=dict(color='#888', width=1.5, dash='dot')
        ))

    _apply_dark_theme(fig, "Portfolio Performance")
    fig.update_yaxes(title_text="Portfolio Value (₹)")

    if save_html:
        fig.write_html(save_html)
    return fig


# ─────────────────────────────────────────────
#  Shared theme helper
# ─────────────────────────────────────────────

def _apply_dark_theme(fig: go.Figure, title: str) -> None:
    """Apply a consistent dark theme to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS['text'], family='Courier New'),
            x=0.5
        ),
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], family='Courier New'),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            zeroline=False,
            showspikes=True,
            spikecolor=COLORS['text'],
            spikethickness=1
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            zeroline=False,
            showspikes=True,
            spikecolor=COLORS['text'],
            spikethickness=1
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=COLORS['grid'],
            borderwidth=1
        ),
        hovermode='x unified',
        height=700,
    )
