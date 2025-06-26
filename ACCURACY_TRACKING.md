# Prediction Accuracy Tracking System

This system automatically tracks the accuracy of crypto price predictions and provides comprehensive analysis tools.

## How It Works

### 1. Automatic Data Collection (crypto_prediction_bot.py)

The main prediction bot automatically:
- Evaluates mature predictions when they reach their target times
- Stores actual vs predicted prices in the database
- Tracks error metrics (absolute error, percent error, direction accuracy)
- Updates timeseries data for analysis
- Runs accuracy evaluation every hour

**Database Tables Created:**
- `prediction_evaluations` - Individual prediction accuracy results
- `actual_prices_at_targets` - Actual prices at prediction target times  
- `accuracy_metrics_summary` - Aggregated accuracy metrics by time period
- `prediction_timeseries` - Time series of prices, predictions, and errors

### 2. On-Demand Plotting (plot_accuracy.py)

The separate plotting script allows you to generate visualizations whenever needed:

```bash
# Generate all plots for all cryptos (last 30 days)
python plot_accuracy.py

# Generate plots for Bitcoin only (last 7 days)
python plot_accuracy.py --crypto bitcoin --days 7

# Show quick accuracy summary without plots
python plot_accuracy.py --summary

# Generate plots in specific directory
python plot_accuracy.py --output my_plots

# Show available options
python plot_accuracy.py --help
```

## Generated Visualizations

### 1. Timeseries Plots
- **Actual vs Predicted Prices**: Shows how well predictions track actual price movements
- **Absolute Errors Over Time**: Shows prediction errors in dollar amounts
- **Percent Errors Over Time**: Shows prediction errors as percentages
- **Error Distribution**: Histogram of prediction errors

### 2. Error Analysis
- **Error Histograms**: Distribution of errors by prediction horizon (1h, 1d, 1w)
- **Accuracy Metrics**: Mean/median errors, direction accuracy, RMSE

### 3. Reports
- **Text Report**: Comprehensive accuracy statistics saved as `accuracy_report.txt`
- **README**: Summary of generated files and parameters

## Key Metrics Tracked

- **Absolute Error**: Dollar difference between predicted and actual price
- **Percent Error**: Percentage difference between predicted and actual price  
- **Squared Error**: For calculating RMSE
- **Direction Accuracy**: Whether prediction correctly predicted price direction
- **Confidence**: Model confidence in the prediction

## Usage Examples

### Running the Bot with Accuracy Tracking
```bash
# Run once with accuracy evaluation
python crypto_prediction_bot.py --once

# Run continuously (evaluates accuracy every hour)
python crypto_prediction_bot.py
```

### Generating Plots

```bash
# Quick summary
python plot_accuracy.py -s

# All cryptos, last 30 days
python plot_accuracy.py

# Bitcoin only, last 14 days  
python plot_accuracy.py -c bitcoin -d 14

# Ethereum only, custom output directory
python plot_accuracy.py -c ethereum -o ethereum_analysis
```

## File Structure

```
plots/
├── 20241201_143022/              # Timestamp directory
│   ├── bitcoin_accuracy_timeseries.png
│   ├── bitcoin_error_histograms.png
│   ├── ethereum_accuracy_timeseries.png
│   ├── ethereum_error_histograms.png
│   ├── accuracy_report.txt
│   └── README.txt
```

## Database Schema

### prediction_evaluations
```sql
CREATE TABLE prediction_evaluations (
    id INTEGER PRIMARY KEY,
    prediction_id INTEGER,
    crypto TEXT,
    prediction_horizon TEXT,
    predicted_price REAL,
    actual_price REAL,
    absolute_error REAL,
    percent_error REAL,
    squared_error REAL,
    direction_predicted INTEGER,
    direction_actual INTEGER,
    direction_correct INTEGER,
    prediction_timestamp TIMESTAMP,
    evaluation_timestamp TIMESTAMP,
    confidence REAL,
    target_timestamp TIMESTAMP
);
```

### prediction_timeseries
```sql
CREATE TABLE prediction_timeseries (
    id INTEGER PRIMARY KEY,
    crypto TEXT,
    timestamp TIMESTAMP,
    actual_price REAL,
    predicted_price_1h REAL,
    predicted_price_1d REAL,
    predicted_price_1w REAL,
    error_1h REAL,
    error_1d REAL,
    error_1w REAL,
    percent_error_1h REAL,
    percent_error_1d REAL,
    percent_error_1w REAL
);
```

## Benefits

1. **Automatic Tracking**: No manual intervention needed for data collection
2. **Comprehensive Metrics**: Multiple error types and accuracy measures
3. **Flexible Analysis**: Generate plots for any time period or crypto
4. **Historical Data**: Build up accuracy trends over time
5. **Performance Insights**: Identify which horizons and conditions work best

## Tips

- Let the bot run for several days to accumulate meaningful accuracy data
- Use `--summary` for quick checks without generating large plot files
- Combine different time periods to see accuracy trends
- Check the text reports for detailed statistical analysis 