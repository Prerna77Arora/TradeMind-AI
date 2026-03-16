"""
main.py
-------
Master pipeline for the AI Stock Prediction & Portfolio Advisory Platform.

Execution order:
  1.  Collect historical OHLCV data via Yahoo Finance
  2.  Clean data (fill, deduplicate, outlier removal)
  3.  Engineer technical & statistical features
  4.  Preprocess: scale + create LSTM sequences
  5.  Train LSTM model (or load an existing one)
  6.  Evaluate model performance
  7.  Collect or create investor profile
  8.  Apply financial filters
  9.  Generate personalised recommendations
  10. Produce interactive Plotly visualisations
  11. Launch AI chatbot for natural-language advisory

Usage:
    python main.py

    # Skip training (use saved model):
    python main.py --no-train

    # Custom ticker list:
    python main.py --tickers AAPL MSFT GOOGL

    # Non-interactive demo mode:
    python main.py --demo
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Local module imports ──────────────────────────────────────────────────────
from data_collection import fetch_stock_data, get_company_info
from data_cleaning import clean_stock_data
from feature_engineering import engineer_features
from data_preprocessing import prepare_data
from model_training import train_model, load_trained_model, predict
from model_evaluation import evaluate_model
from investor_profile import (
    InvestorProfile,
    collect_investor_profile_interactive,
    create_default_profile
)
from recommendation_engine import generate_recommendation, generate_portfolio_recommendations
from visualization import (
    plot_price_with_indicators,
    plot_predictions,
    plot_rsi_macd,
    plot_buy_sell_signals,
    plot_training_history
)
from chatbot_interface import create_chatbot


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/'
CHARTS_DIR = 'charts/'
DATA_DIR = 'data/'

DEFAULT_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']  # NSE India
# For US markets use: ['AAPL', 'MSFT', 'GOOGL']

SEQUENCE_LENGTH = 30
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100


def parse_args():
    parser = argparse.ArgumentParser(description='AI Stock Advisor Platform')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                        help='List of stock tickers to analyse')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training and load existing model')
    parser.add_argument('--demo', action='store_true',
                        help='Use default demo investor profile (non-interactive)')
    parser.add_argument('--period', default='2y',
                        help='Data period for yfinance (e.g., 1y, 2y, 5y)')
    parser.add_argument('--no-chat', action='store_true',
                        help='Skip the chatbot session')
    parser.add_argument('--gemini-key', default=None,
                        help='Google Gemini API key (or set GOOGLE_API_KEY env var)')
    return parser.parse_args()


def ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [MODEL_PATH.split('/')[0], CHARTS_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)


def step_banner(step_num: int, title: str):
    print(f"\n{'━' * 55}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'━' * 55}")


def run_pipeline(args):
    """Main pipeline execution."""
    ensure_dirs()
    primary_ticker = args.tickers[0]

    print("\n" + "╔" + "═" * 53 + "╗")
    print("║     AI STOCK PREDICTION & ADVISORY PLATFORM      ║")
    print("╚" + "═" * 53 + "╝\n")

    # ── STEP 1: Data Collection ───────────────────────────────────────────────
    step_banner(1, "DATA COLLECTION")
    raw_data = {}
    for ticker in args.tickers:
        try:
            raw_data[ticker] = fetch_stock_data(ticker, period=args.period)
        except RuntimeError as e:
            print(f"  [Warning] {e}")

    if not raw_data:
        print("[Error] No data collected. Check ticker symbols and internet connection.")
        sys.exit(1)

    # Fetch company fundamentals for the primary ticker
    print(f"\n  Fetching fundamentals for {primary_ticker}...")
    try:
        company_info = {primary_ticker: get_company_info(primary_ticker)}
        pe_ratio = company_info[primary_ticker].get('pe_ratio')
        print(f"  Sector: {company_info[primary_ticker]['sector']} | "
              f"P/E: {pe_ratio} | "
              f"Market Cap: {company_info[primary_ticker]['market_cap']}")
    except Exception:
        company_info = {}
        pe_ratio = None

    # ── STEP 2: Data Cleaning ─────────────────────────────────────────────────
    step_banner(2, "DATA CLEANING")
    clean_data = {}
    for ticker, df in raw_data.items():
        clean_data[ticker] = clean_stock_data(df)
        # Save cleaned data
        clean_data[ticker].to_csv(os.path.join(DATA_DIR, f"{ticker}_clean.csv"))
        print(f"  Saved: {DATA_DIR}{ticker}_clean.csv")

    # ── STEP 3: Feature Engineering ───────────────────────────────────────────
    step_banner(3, "FEATURE ENGINEERING")
    feature_data = {}
    for ticker, df in clean_data.items():
        feature_data[ticker] = engineer_features(df)

    primary_df = feature_data[primary_ticker]

    # ── STEP 4: Data Preprocessing ────────────────────────────────────────────
    step_banner(4, "DATA PREPROCESSING")
    prep = prepare_data(
        df=primary_df,
        sequence_length=SEQUENCE_LENGTH,
        test_ratio=0.2,
        target_col='Close',
        scaler_save_path=SCALER_PATH
    )

    X_train = prep['X_train']
    X_test = prep['X_test']
    y_train = prep['y_train']
    y_test = prep['y_test']
    target_scaler = prep['target_scaler']
    n_features = prep['n_features']

    print(f"\n  Sequence length : {SEQUENCE_LENGTH}")
    print(f"  Features        : {n_features}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples    : {len(X_test)}")

    # ── STEP 5: Model Training / Loading ──────────────────────────────────────
    step_banner(5, "LSTM MODEL")

    if args.no_train and os.path.exists(MODEL_PATH):
        print(f"  Loading existing model from {MODEL_PATH}")
        model = load_trained_model(MODEL_PATH)
        history = None
    else:
        print("  Training new LSTM model...")
        model, history = train_model(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            sequence_length=SEQUENCE_LENGTH,
            n_features=n_features,
            lstm_units=LSTM_UNITS,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            model_save_path=MODEL_PATH
        )

    # ── STEP 6: Model Evaluation ──────────────────────────────────────────────
    step_banner(6, "MODEL EVALUATION")
    eval_results = evaluate_model(model, X_test, y_test, target_scaler)
    directional_accuracy = eval_results['Directional_Accuracy']

    y_actual = eval_results['y_actual']
    y_pred_test = eval_results['y_pred']

    # ── STEP 7: Investor Profile ──────────────────────────────────────────────
    step_banner(7, "INVESTOR PROFILE")
    if args.demo:
        print("  [Demo mode] Using default investor profile.")
        profile = create_default_profile(
            name="Demo Investor",
            amount=500000,
            risk='medium',
            horizon=3.0
        )
        print(profile.summary())
    else:
        profile = collect_investor_profile_interactive()

    # ── STEP 8 & 9: Predictions + Recommendations ─────────────────────────────
    step_banner(9, "PERSONALISED RECOMMENDATIONS")

    # Predict the next price for each ticker
    predicted_prices = {}
    for ticker in args.tickers:
        if ticker not in feature_data:
            continue
        fdf = feature_data[ticker]

        # Use the same feature scaler (trained on primary ticker) for simplicity
        # In production, train per-ticker or use normalised relative features
        from data_preprocessing import get_feature_columns, load_scalers
        feat_cols = get_feature_columns(fdf)
        feat_sc = prep['feature_scaler']
        tgt_sc = prep['target_scaler']

        scaled_f = feat_sc.transform(fdf[feat_cols].values[-SEQUENCE_LENGTH:])
        X_next = scaled_f.reshape(1, SEQUENCE_LENGTH, len(feat_cols))
        pred_scaled = model.predict(X_next, verbose=0).flatten()[0]
        predicted_prices[ticker] = float(
            tgt_sc.inverse_transform([[pred_scaled]])[0][0]
        )
        print(f"  {ticker}: Current ₹{fdf['Close'].iloc[-1]:,.2f} → "
              f"Predicted ₹{predicted_prices[ticker]:,.2f}")

    # Generate recommendations
    recs = generate_portfolio_recommendations(
        tickers=args.tickers,
        dataframes=feature_data,
        predicted_prices=predicted_prices,
        profile=profile,
        company_info=company_info,
        directional_accuracy=directional_accuracy
    )

    for rec in recs:
        print(rec.display(currency=profile.currency))

    # ── STEP 10: Visualisation ────────────────────────────────────────────────
    step_banner(10, "VISUALISATION")
    print("  Generating interactive charts...")

    fig1 = plot_price_with_indicators(
        primary_df, primary_ticker,
        save_html=os.path.join(CHARTS_DIR, f"{primary_ticker}_price.html")
    )
    fig2 = plot_predictions(
        primary_df, y_actual, y_pred_test, primary_ticker,
        save_html=os.path.join(CHARTS_DIR, f"{primary_ticker}_predictions.html")
    )
    fig3 = plot_rsi_macd(
        primary_df, primary_ticker,
        save_html=os.path.join(CHARTS_DIR, f"{primary_ticker}_rsi_macd.html")
    )
    fig4 = plot_buy_sell_signals(
        primary_df, primary_ticker,
        save_html=os.path.join(CHARTS_DIR, f"{primary_ticker}_signals.html")
    )

    if history is not None:
        fig5 = plot_training_history(
            history,
            save_html=os.path.join(CHARTS_DIR, "training_loss.html")
        )

    print(f"\n  Charts saved to '{CHARTS_DIR}'. Open HTML files in a browser.")

    # ── STEP 11: AI Chatbot ───────────────────────────────────────────────────
    if not args.no_chat:
        step_banner(11, "AI CHATBOT — FINACEGPT")
        chatbot = create_chatbot(api_key=args.gemini_key)
        if chatbot:
            chatbot.run_interactive_session(
                recommendations=recs,
                profile_summary=profile.summary()
            )
        else:
            print("  Chatbot unavailable. Set GOOGLE_API_KEY to enable it.")
            print("  Recommendations are printed above — review them manually.")

    print("\n" + "╔" + "═" * 53 + "╗")
    print("║       ANALYSIS COMPLETE — HAPPY INVESTING! 📈      ║")
    print("╚" + "═" * 53 + "╝\n")

    return {
        'model': model,
        'profile': profile,
        'recommendations': recs,
        'eval_metrics': eval_results,
    }


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
