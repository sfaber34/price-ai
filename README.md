# 🚀 Crypto Price Prediction Bot

A free, local ML-powered bot that predicts Bitcoin and Ethereum prices using multiple data sources and XGBoost models.

## 🎯 Features

- **Multi-horizon predictions**: 15 minute, 1 hour, and 4 hour forecasts
- **Multiple data sources**: Crypto prices, traditional markets, economic indicators
- **Advanced ML models**: XGBoost ensemble with feature selection and cross-validation
- **Technical indicators**: 50+ features including RSI, MACD, Bollinger Bands
- **Market correlations**: Integrates stocks, bonds, commodities, and forex data
- **Real-time updates**: Configurable prediction intervals (5-10 minutes)
- **Local database**: SQLite storage for predictions and historical data
- **Free APIs**: Uses only free data sources (CoinGecko, Yahoo Finance, FRED)

## 📊 Sample Output

```
================================================================================
CRYPTO PRICE PREDICTIONS - 2024-01-15 14:30:15
================================================================================

📈 BITCOIN
----------------------------------------
   1H | Current: $43,250.00 | Predicted: $43,450.00 | Return:  +0.46% | 🟢 ⭐⭐⭐⭐
   1D | Current: $43,250.00 | Predicted: $44,100.00 | Return:  +1.96% | 🟢 ⭐⭐⭐
   1W | Current: $43,250.00 | Predicted: $41,800.00 | Return:  -3.35% | 🔴 ⭐⭐⭐

📈 ETHEREUM
----------------------------------------
   1H | Current:  $2,650.00 | Predicted:  $2,670.00 | Return:  +0.75% | 🟢 ⭐⭐⭐⭐
   1D | Current:  $2,650.00 | Predicted:  $2,720.00 | Return:  +2.64% | 🟢 ⭐⭐⭐
   1W | Current:  $2,650.00 | Predicted:  $2,580.00 | Return:  -2.64% | 🔴 ⭐⭐
================================================================================
```

### 📊 Output Explanation

**Column Meanings:**
- **Current**: Real-time price from live APIs
- **Predicted**: Model's price forecast for the time horizon
- **Return**: Expected percentage change from current to predicted price

**Direction Indicators:**
- **🟢 Green Circle**: Model predicts price will go **UP**
- **🔴 Red Circle**: Model predicts price will go **DOWN**

**Confidence Levels (Stars):**
- **⭐** = Very low confidence (50-60%)
- **⭐⭐** = Low confidence (60-70%)
- **⭐⭐⭐** = Medium confidence (70-80%)
- **⭐⭐⭐⭐** = High confidence (80-90%)
- **⭐⭐⭐⭐⭐** = Very high confidence (90-100%)

### 🔬 Model Components Explained

**Two XGBoost Models Work Together:**

1. **Price Prediction (Return %)**: 
   - **XGBoost Regressor** trained on 89 features (technical indicators, market correlations, economic data)
   - Predicts exact price target and calculates return percentage vs current price
   - Used for: "Predicted: $43,450.00" and "Return: +0.46%"

2. **Direction Prediction (🔴🟢)**: 
   - **XGBoost Classifier** trained on the same 89 features
   - Binary classification: Will price go UP or DOWN from historical baseline?
   - Used for: 🟢 (UP) or 🔴 (DOWN) circles

3. **Confidence Score (⭐)**: 
   - Based on the direction classifier's probability output
   - Higher confidence = model is more certain about the direction
   - Calculated as distance from 50% probability (closer to 0% or 100% = higher confidence)

**Data Sources (No News Sentiment):**
- Technical indicators: RSI, MACD, Bollinger Bands, moving averages
- Market correlations: S&P 500, Gold, Dollar Index, VIX relationships  
- Economic indicators: Fed rates, inflation, GDP, unemployment (via FRED API)
- Price history: Lagged features and momentum indicators

**Why Direction and Return % Can Differ:**
The regressor and classifier are trained independently on the same data but optimize for different objectives. The regressor predicts exact prices, while the classifier focuses on directional movement patterns. This can lead to scenarios where the direction shows 🟢 (UP) but return % is negative, meaning the model expects upward movement from the historical training baseline but the current real-time price is already higher than the predicted target.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd crypto-price-prediction-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys (optional but recommended)
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Train optimal models (first time setup)
```bash
# Quick test (1 run per training window - just test script)
python train_optimal_models.py --quick

# Full training optimization (30 runs per window - recommended)  
python train_optimal_models.py
```

### 5. Run the bot
```bash
# Test run (single prediction using pre-trained models)
python crypto_prediction_bot.py --once

# Continuous mode (predictions every 10 minutes)
python crypto_prediction_bot.py
```

## 🔑 API Keys (Free Tier)

While the bot works without API keys using CoinGecko and Yahoo Finance, adding these free APIs will enhance predictions:

1. **Alpha Vantage** (Free: 5 calls/min, 500/day)
   - Sign up: https://www.alphavantage.co/support/#api-key
   - Used for: Stock market data

2. **FRED** (Free: 120 calls/min)
   - Sign up: https://fred.stlouisfed.org/docs/api/api_key.html
   - Used for: Economic indicators (Fed rates, inflation, GDP)

3. **News API** (Free: 1000 calls/day)
   - Sign up: https://newsapi.org/register
   - Used for: News sentiment analysis

Add these to your `.env` file:
```
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

## 🧠 How It Works

### Two-Script Architecture

This bot uses an optimized two-script workflow:

#### 1. **train_optimal_models.py** - The Heavy Lifter 🏋️
- **Purpose**: Finds optimal training windows and trains production models
- **Process**: 
  - Tests 6 different training window sizes (1 month → 4 months)
  - Runs 30 backtests per window to find the best performing setup
  - Trains final production models using optimal windows
  - Saves trained models to `models/` directory
- **Run**: Once initially, then weekly/monthly for reoptimization
- **Output**: Pre-trained models ready for immediate use

#### 2. **crypto_prediction_bot.py** - The Speed Demon ⚡
- **Purpose**: Makes real-time predictions using pre-trained models  
- **Process**:
  - Loads optimal models from `models/` directory (instant startup)
  - Collects recent data and generates predictions
  - No training delays - immediate predictions
- **Run**: Continuously for live predictions
- **Fallback**: Can still train from scratch if no pre-trained models exist

### Workflow Benefits
- **🚀 Instant startup**: Bot loads pre-trained models immediately
- **🎯 Optimized accuracy**: Models use scientifically determined optimal training windows
- **⚡ Efficiency**: No wasted compute retraining every time
- **🔄 Flexible**: Can retrain/reoptimize models as needed

### Data Collection
- **Crypto prices**: CoinGecko API (free, no registration)
- **Traditional markets**: Yahoo Finance (S&P 500, Gold, Dollar Index, VIX)
- **Economic data**: FRED API (Federal Reserve economic indicators)
- **Rate limiting**: Respects free API limits automatically

### Feature Engineering
- **Technical indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Market correlations**: Rolling correlations with traditional assets
- **Time features**: Hour, day, weekend effects, market sessions
- **Lagged features**: Price history and momentum indicators
- **Volume analysis**: Price-volume relationships

### Machine Learning
- **XGBoost models**: Separate regression and classification models
- **Feature selection**: Selects top 50 most predictive features
- **Time series validation**: Prevents data leakage with proper splits
- **Multi-horizon**: Different models for 15m, 1h, and 4h predictions
- **Ensemble approach**: Combines price and direction predictions

### Prediction Types
- **Price targets**: Absolute price predictions
- **Return percentages**: Expected percentage change
- **Direction**: Up/down movement with confidence scores
- **Confidence levels**: Model certainty (0-100%)

## 📁 Project Structure

```
crypto-price-prediction-bot/
├── config.py                    # Configuration and API settings
├── data_collector.py            # Data collection from APIs
├── feature_engineering.py       # Technical indicators and features
├── ml_predictor.py              # Machine learning models
├── train_optimal_models.py      # Training script - finds optimal windows & trains models
├── crypto_prediction_bot.py     # Main bot - loads models & makes predictions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .env.example                 # Environment variables template
├── crypto_predictions.db       # SQLite database (created automatically)
├── crypto_bot.log              # Log file (created automatically)
├── optimal_training_results.json # Training optimization results
└── models/                      # Trained models & metadata
    ├── production_models.json   # Model metadata & paths
    ├── bitcoin_15m_production.pkl
    ├── bitcoin_1h_production.pkl
    ├── bitcoin_4h_production.pkl
    ├── ethereum_15m_production.pkl
    ├── ethereum_1h_production.pkl
    └── ethereum_4h_production.pkl
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Update frequency**: How often to generate predictions (default: 10 minutes)
- **Cryptocurrencies**: Add/remove coins to predict
- **Technical indicators**: Modify indicator parameters
- **Model settings**: Adjust XGBoost parameters
- **Traditional markets**: Add/remove correlated assets

## 🔍 Monitoring & Analysis

### Database Tables
- `crypto_data`: Historical price data
- `traditional_markets`: Stock/commodity data
- `economic_indicators`: Fed data
- `predictions`: All generated predictions

### Logs
- `crypto_bot.log`: Detailed execution logs
- Console output: Real-time status updates

### Performance Monitoring
The bot tracks prediction accuracy and model performance over time.

## 🚨 Important Notes

### Limitations
- **Free API limits**: Some features may be limited by API rate limits
- **Market volatility**: Crypto markets are highly unpredictable
- **Model accuracy**: No guarantee of profitable predictions
- **Data quality**: Depends on external API availability

### Risk Disclaimer
This bot is for educational purposes only. Cryptocurrency trading involves significant risk. Do not use these predictions for financial decisions without proper risk management.

### Best Practices
- Run for several days to collect sufficient training data
- Monitor prediction accuracy before trusting results
- Use ensemble results rather than single predictions
- Consider market context and external factors

## 🛡️ Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```
   Solution: Wait for rate limit reset or upgrade to paid API tiers
   ```

2. **No Predictions Generated**
   ```
   Solution: Check internet connection and API availability
   ```

3. **Low Model Accuracy**
   ```
   Solution: Increase training data history (config.py)
   ```

4. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Debug Mode
```bash
# Run with verbose logging
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python crypto_prediction_bot.py --once
```

## 📈 Advanced Usage

### Custom Analysis
```python
from crypto_prediction_bot import CryptoPredictionBot

bot = CryptoPredictionBot()
performance = bot.get_model_performance()
print(performance)
```

### Model Training Options

#### Optimal Training Script
```bash
# Quick test (1 run per window - for testing)
python train_optimal_models.py --quick

# Standard optimization (30 runs per window - recommended)
python train_optimal_models.py --runs 30

# Extended data collection (default 180 days)
python train_optimal_models.py --days 365

# Use this:
python3 train_optimal_models.py --skip-backtest
```

#### Manual Bot Training
```python
# Force model retraining in the bot
bot.train_models(force_retrain=True)
```

### Database Queries
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('crypto_predictions.db')
df = pd.read_sql_query('SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10', conn)
print(df)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- CoinGecko for free crypto data
- Yahoo Finance for market data
- Federal Reserve (FRED) for economic indicators
- XGBoost team for the ML framework
- Technical Analysis Library (ta) for indicators

---

⚠️ **Disclaimer**: This software is for educational purposes only. Cryptocurrency trading involves significant financial risk. The developers are not responsible for any financial losses incurred through the use of this software. 