# 🚀 Crypto Price Prediction Bot

A free, local ML-powered bot that predicts Bitcoin and Ethereum prices using multiple data sources and XGBoost models.

## 🎯 Features

- **Multi-horizon predictions**: 1 hour, 1 day, and 1 week forecasts
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

### 4. Run the bot
```bash
# Test run (single prediction)
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
- **Multi-horizon**: Different models for 1h, 1d, and 1w predictions
- **Ensemble approach**: Combines price and direction predictions

### Prediction Types
- **Price targets**: Absolute price predictions
- **Return percentages**: Expected percentage change
- **Direction**: Up/down movement with confidence scores
- **Confidence levels**: Model certainty (0-100%)

## 📁 Project Structure

```
crypto-price-prediction-bot/
├── config.py                 # Configuration and API settings
├── data_collector.py         # Data collection from APIs
├── feature_engineering.py    # Technical indicators and features
├── ml_predictor.py           # Machine learning models
├── crypto_prediction_bot.py  # Main bot orchestrator
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env.example             # Environment variables template
├── crypto_predictions.db    # SQLite database (created automatically)
├── crypto_bot.log          # Log file (created automatically)
└── models/                 # Trained models (created automatically)
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

### Manual Training
```python
# Force model retraining
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