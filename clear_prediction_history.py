#!/usr/bin/env python3
"""
Clear Prediction History Database

This script clears all prediction and evaluation data from the database
while preserving the historical market data (crypto_data, traditional_markets, etc.)

Use this after retraining models with bug fixes to start fresh with accurate evaluations.
"""

import sqlite3
import config
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_prediction_history():
    """
    Clear all prediction-related tables while preserving market data
    """
    
    # Tables to clear (prediction and evaluation data)
    prediction_tables = [
        'predictions',
        'prediction_evaluations', 
        'actual_prices_at_targets',
        'accuracy_metrics_summary',
        'prediction_timeseries'
    ]
    
    # Tables to preserve (market data)
    preserved_tables = [
        'crypto_data',
        'traditional_markets', 
        'economic_indicators'
    ]
    
    try:
        logger.info("Connecting to database...")
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get list of existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"Found tables in database: {existing_tables}")
        
        # Clear prediction tables
        cleared_count = 0
        for table in prediction_tables:
            if table in existing_tables:
                # Get row count before clearing
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                if row_count > 0:
                    cursor.execute(f"DELETE FROM {table}")
                    logger.info(f"Cleared {row_count} rows from {table}")
                    cleared_count += 1
                else:
                    logger.info(f"Table {table} was already empty")
            else:
                logger.info(f"Table {table} does not exist (will be created when needed)")
        
        # Commit the changes
        conn.commit()
        
        # Show preserved tables status
        logger.info("\nPreserved tables (market data):")
        for table in preserved_tables:
            if table in existing_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                logger.info(f"  {table}: {row_count} rows preserved")
            else:
                logger.info(f"  {table}: table does not exist")
        
        conn.close()
        
        if cleared_count > 0:
            logger.info(f"\n✅ Successfully cleared {cleared_count} prediction tables")
            logger.info("The database is now ready for fresh prediction evaluations")
        else:
            logger.info("\n📊 No prediction data found to clear")
            
        logger.info("Historical market data has been preserved")
        
    except Exception as e:
        logger.error(f"❌ Failed to clear prediction history: {e}")
        raise

def confirm_clear():
    """
    Ask for user confirmation before clearing data
    """
    print("🗑️  CLEAR PREDICTION HISTORY DATABASE")
    print("="*50)
    print("This will permanently delete:")
    print("  • All predictions")
    print("  • All prediction evaluations") 
    print("  • All accuracy tracking data")
    print("  • All prediction timeseries data")
    print()
    print("This will PRESERVE:")
    print("  • Historical crypto price data")
    print("  • Traditional market data")
    print("  • Economic indicator data")
    print()
    
    response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        return True
    elif response in ['no', 'n']:
        print("Operation cancelled.")
        return False
    else:
        print("Please enter 'yes' or 'no'")
        return confirm_clear()

if __name__ == "__main__":
    print(f"Database path: {config.DATABASE_PATH}")
    print(f"Timestamp: {datetime.now()}")
    print()
    
    if confirm_clear():
        clear_prediction_history()
    else:
        logger.info("Operation cancelled by user") 