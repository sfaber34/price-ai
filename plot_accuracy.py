#!/usr/bin/env python3
"""
Prediction Accuracy Plotting Script
Generates comprehensive visualizations of prediction accuracy data
"""
import argparse
import os
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from prediction_accuracy_tracker import PredictionAccuracyTracker
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_plots_directory(base_dir="plots"):
    """Create plots directory with timestamp subdirectory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(base_dir, timestamp)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def generate_all_plots(crypto=None, days_back=30, save_dir=None):
    """Generate all accuracy plots for specified crypto(s)"""
    tracker = PredictionAccuracyTracker()
    
    if save_dir is None:
        save_dir = create_plots_directory()
    
    # Determine which cryptos to plot
    if crypto:
        cryptos = [crypto] if crypto in config.CRYPTOCURRENCIES else []
        if not cryptos:
            logger.error(f"Invalid crypto: {crypto}. Available: {config.CRYPTOCURRENCIES}")
            return
    else:
        cryptos = config.CRYPTOCURRENCIES
    
    logger.info(f"Generating plots for: {cryptos}")
    logger.info(f"Looking back {days_back} days")
    logger.info(f"Saving plots to: {save_dir}")
    
    plots_generated = 0
    
    for crypto in cryptos:
        try:
            # Check if we have data for this crypto
            metrics = tracker.calculate_accuracy_metrics(crypto=crypto, days_back=days_back)
            if not metrics:
                logger.warning(f"No accuracy data available for {crypto}")
                continue
            
            logger.info(f"Generating plots for {crypto}...")
            
            # Generate timeseries plot
            timeseries_path = os.path.join(save_dir, f"{crypto}_accuracy_timeseries.png")
            tracker.plot_prediction_timeseries(
                crypto=crypto,
                days_back=days_back,
                save_path=timeseries_path
            )
            plots_generated += 1
            
            # Generate error histograms
            histogram_path = os.path.join(save_dir, f"{crypto}_error_histograms.png")
            tracker.plot_error_histograms(
                crypto=crypto,
                days_back=days_back,
                save_path=histogram_path
            )
            plots_generated += 1
            
            logger.info(f"✅ Generated plots for {crypto}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate plots for {crypto}: {e}")
    
    # Generate combined accuracy report
    try:
        report_path = os.path.join(save_dir, "accuracy_report.txt")
        report = tracker.generate_accuracy_report(crypto=None, days_back=days_back)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Also print to console
        print(report)
        
        logger.info(f"📊 Accuracy report saved to: {report_path}")
        plots_generated += 1
        
    except Exception as e:
        logger.error(f"Failed to generate accuracy report: {e}")
    
    if plots_generated > 0:
        logger.info(f"🎉 Successfully generated {plots_generated} files in {save_dir}")
        
        # Create a summary file
        summary_path = os.path.join(save_dir, "README.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Prediction Accuracy Plots\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: Last {days_back} days\n")
            f.write(f"Cryptos: {', '.join(cryptos)}\n\n")
            f.write(f"Files:\n")
            for file in os.listdir(save_dir):
                if file.endswith(('.png', '.txt')):
                    f.write(f"- {file}\n")
        
        return save_dir
    else:
        logger.warning("No plots were generated")
        return None

def show_accuracy_summary():
    """Show a quick accuracy summary without generating plots"""
    tracker = PredictionAccuracyTracker()
    
    print("\n" + "="*80)
    print("PREDICTION ACCURACY SUMMARY")
    print("="*80)
    print("Note: Predictions can only be evaluated after their target time:")
    print("  • 15M predictions: evaluated 15 minutes after prediction")
    print("  • 1H predictions: evaluated 1 hour after prediction")
    print("  • 4H predictions: evaluated 4 hours after prediction")
    print("-" * 80)
    
    overall_has_data = False
    
    for crypto in config.CRYPTOCURRENCIES:
        print(f"\n🔸 {crypto.upper()}")
        print("-" * 40)
        
        crypto_has_data = False
        for horizon in config.PREDICTION_INTERVALS:
            metrics = tracker.calculate_accuracy_metrics(crypto=crypto, horizon=horizon, days_back=7)
            
            if metrics and metrics.get('total_predictions', 0) > 0:
                crypto_has_data = True
                overall_has_data = True
                print(f"  {horizon.upper()}: {metrics['total_predictions']} evaluations")
                print(f"    Mean Error: {metrics['mean_percent_error']:.2f}%")
                print(f"    Direction Accuracy: {metrics['direction_accuracy']:.1%}")
            else:
                # Check if we have any predictions stored but not yet mature for evaluation
                # This gives more helpful feedback
                if horizon == '15m':
                    time_needed = "15 minutes"
                elif horizon == '1h':
                    time_needed = "1 hour"
                elif horizon == '4h':
                    time_needed = "4 hours"
                else:
                    time_needed = "target time"
                    
                print(f"  {horizon.upper()}: No mature evaluations yet")
                print(f"    (Requires {time_needed} to pass after prediction)")
        
        if not crypto_has_data:
            print(f"    📍 No mature evaluations available for {crypto} yet")
    
    if not overall_has_data:
        print(f"\n⏳ IMPORTANT: No mature prediction evaluations are available yet.")
        print(f"   This is normal if the model has been running for less than:")
        print(f"   • 15 minutes (for 15M predictions)")
        print(f"   • 1 hour (for 1H predictions)")
        print(f"   • 4 hours (for 4H predictions)")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Generate prediction accuracy plots')
    parser.add_argument('--crypto', '-c', choices=config.CRYPTOCURRENCIES + ['all'], 
                       default='all', help='Crypto to plot (default: all)')
    parser.add_argument('--days', '-d', type=int, default=30, 
                       help='Days to look back (default: 30)')
    parser.add_argument('--output', '-o', help='Output directory (default: plots/timestamp)')
    parser.add_argument('--summary', '-s', action='store_true', 
                       help='Show accuracy summary only (no plots)')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots instead of saving them')
    
    args = parser.parse_args()
    
    # Validate crypto argument
    if args.crypto == 'all':
        crypto = None
    elif args.crypto in config.CRYPTOCURRENCIES:
        crypto = args.crypto
    else:
        logger.error(f"Invalid crypto: {args.crypto}")
        return 1
    
    try:
        if args.summary:
            show_accuracy_summary()
        else:
            # Set up matplotlib backend
            if not args.show:
                plt.switch_backend('Agg')  # Non-interactive backend for saving
            
            save_dir = generate_all_plots(
                crypto=crypto,
                days_back=args.days,
                save_dir=args.output
            )
            
            if save_dir and args.show:
                # If user wants to show plots, open the directory
                import subprocess
                import platform
                
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', save_dir])
                elif platform.system() == 'Windows':
                    subprocess.run(['explorer', save_dir])
                elif platform.system() == 'Linux':
                    subprocess.run(['xdg-open', save_dir])
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 