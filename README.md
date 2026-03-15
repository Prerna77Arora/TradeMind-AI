# 🤖 AI Stock Prediction & Portfolio Advisory Platform

A production-grade, end-to-end AI investment system built in Python.  
It collects live stock data, trains an LSTM deep learning model, applies
financial filters, and generates **personalised buy/sell/hold recommendations**
— all explained by a **Google Gemini-powered AI chatbot**.

---

## 📁 Project Structure

```
ai_stock_advisor/
├── data_collection.py       # Yahoo Finance data fetching
├── data_cleaning.py         # Missing values, outliers, timestamps
├── feature_engineering.py   # SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV
├── data_preprocessing.py    # Min-Max scaling + sliding window sequences
├── model_training.py        # Stacked LSTM with dropout (TensorFlow/Keras)
├── model_evaluation.py      # MAE, RMSE, R², Directional Accuracy
├── financial_filters.py     # Trend, RSI, MACD, P/E, Volume, Bollinger filters
├── risk_management.py       # Stop Loss, Take Profit, Sharpe, MDD, position sizing
├── investor_profile.py      # Investor risk profile (low/medium/high)
├── recommendation_engine.py # BUY / SELL / HOLD decision engine
├── visualization.py         # Interactive Plotly charts (dark theme)
├── chatbot_interface.py     # Google Gemini AI chatbot
├── main.py                  # Master pipeline orchestrator
├── requirements.txt
├── models/                  # Saved LSTM weights + scalers
├── data/                    # Cleaned CSV exports
└── charts/                  # HTML interactive charts
```

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
git clone <repo-url>
cd ai_stock_advisor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Google Gemini API key (optional — for chatbot)

```bash
# Linux / Mac
export GOOGLE_API_KEY="your_gemini_api_key_here"

# Windows PowerShell
$env:GOOGLE_API_KEY="your_gemini_api_key_here"

# Or create a .env file:
echo "GOOGLE_API_KEY=your_key" > .env
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

---

## 🚀 Usage

### Full pipeline (interactive investor profile + chatbot)

```bash
python main.py
```

### Demo mode (no user input required)

```bash
python main.py --demo
```

### Custom tickers (NSE India)

```bash
python main.py --tickers RELIANCE.NS TCS.NS INFY.NS HDFC.NS
```

### Custom tickers (US markets)

```bash
python main.py --tickers AAPL MSFT GOOGL AMZN NVDA
```

### Load existing model (skip training)

```bash
python main.py --no-train --demo
```

### Skip chatbot

```bash
python main.py --demo --no-chat
```

### Pass Gemini key directly

```bash
python main.py --gemini-key YOUR_API_KEY
```

---

## 🧠 System Architecture

```
Yahoo Finance
     ↓
Data Cleaning  (ffill, IQR outliers, business-day reindex)
     ↓
Feature Engineering  (20+ technical + statistical features)
     ↓
MinMax Scaling + Sliding Window Sequences (seq_len=30)
     ↓
Stacked LSTM Model  (train/val split — temporal order preserved)
     ↓
Model Evaluation  (MAE, RMSE, R², Directional Accuracy ≥ 65%)
     ↓
Financial Filters  (Trend, RSI, MACD, Bollinger, Volume, P/E)
     ↓
Risk Management  (Stop Loss = Entry − ATR×multiplier, Sharpe Ratio, MDD)
     ↓
Investor Profile  (low / medium / high risk — adjusts position sizing)
     ↓
Recommendation Engine  (BUY / SELL / HOLD with confidence score)
     ↓
Plotly Charts  (interactive HTML — open in browser)
     ↓
Gemini AI Chatbot  (natural language strategy explanation)
```

---

## 📊 Model Architecture

```
Input  →  LSTM(128, return_sequences=True)
       →  Dropout(0.2)
       →  LSTM(64)
       →  Dropout(0.2)
       →  Dense(32, relu)
       →  Dense(1)          ← Predicted Close Price (scaled)
```

- **Optimiser:** Adam (lr=0.001, ReduceLROnPlateau)  
- **Loss:** Mean Squared Error  
- **Early Stopping:** patience=15, restore best weights  
- **Target Directional Accuracy:** ≥ 65%

---

## 📈 Features Generated

| Category      | Indicators |
|---------------|------------|
| Trend         | SMA (10, 20, 50), EMA (12, 26), MACD |
| Momentum      | RSI (14), Price Momentum (5, 10, 20d) |
| Volatility    | Bollinger Bands, ATR, Rolling Std, Annual Volatility |
| Volume        | Volume SMA, Volume Ratio, OBV |
| Statistical   | Daily Returns, Rolling Mean/Std, Candle Body/Shadows |

---

## 🛡️ Risk Management Parameters by Profile

| Parameter         | Conservative | Balanced | Aggressive |
|-------------------|-------------|----------|------------|
| Max stock alloc.  | 5%          | 10%      | 20%        |
| Risk per trade    | 0.5%        | 1.0%     | 2.0%       |
| SL multiplier     | 1.5× ATR    | 2.0× ATR | 3.0× ATR   |
| Min R:R ratio     | 3:1         | 2:1      | 1.5:1      |
| Max P/E           | 25          | 40       | 60         |

---

## 📋 Sample Recommendation Output

```
╔══════════════════════════════════════════════════╗
  🟢  RECOMMENDATION: BUY  —  TCS.NS
══════════════════════════════════════════════════
  Current Price     : ₹3,850.00
  Predicted Price   : ₹3,927.50  (+2.01%)
  Confidence        : HIGH

  Stop Loss         : ₹3,762.40
  Take Profit       : ₹4,025.20
  Risk:Reward       : 1 : 2.0

  Suggested Amount  : ₹50,000.00
  Shares            : 12

  Sharpe Ratio      : 1.42
  Max Drawdown      : -8.3%
  Annual Volatility : 22.5%

  STRATEGY SUMMARY
  ─────────────────
  Based on your Balanced profile and 3.0-year horizon,
  this is a medium term growth opportunity. The model
  forecasts a +2.01% price move. Trend is STRONG_UPTREND.
  RSI at 52. MACD is confirming bullish momentum.
  Risk:Reward of 2.0:1 meets your minimum of 2.0:1.
╚══════════════════════════════════════════════════╝
```

---

## ⚠️ Disclaimer

This platform is for **educational and research purposes only**.  
It does not constitute regulated financial advice.  
Past model performance does not guarantee future results.  
Always consult a SEBI-registered investment advisor before making large investments.

---

## 📄 License

MIT License — free to use, modify, and distribute.
