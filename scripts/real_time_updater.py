#!/usr/bin/env python3
"""
Real-time Data Updater
Continuously updates stock data and predictions in the background
"""
import time
import schedule
import logging
from datetime import datetime, timedelta
from typing import List
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from etl.etl_pipeline import ETLPipeline
from etl.extractors.stock_extractor import StockDataExtractor
from models.ml_prediction_model import StockPredictionModel
from config.database import get_db_session
from etl.loaders.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_time_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealTimeUpdater:
    """Real-time data updater for continuous operation"""
    
    def __init__(self):
        self.etl_pipeline = ETLPipeline()
        self.stock_extractor = StockDataExtractor()
        self.prediction_model = StockPredictionModel()
        self.data_loader = DataLoader()
        self.symbols = Config.DEFAULT_STOCKS
        
    def update_stock_prices(self):
        """Update current stock prices"""
        logger.info("Updating real-time stock prices...")
        
        try:
            # Get real-time data
            realtime_data = self.stock_extractor.extract_realtime_data(self.symbols)
            
            if not realtime_data.empty:
                logger.info(f"Updated prices for {len(realtime_data)} stocks")
                
                # Store in database (you might want a separate table for real-time data)
                # For now, we'll just log the current prices
                for _, row in realtime_data.iterrows():
                    logger.info(f"{row['symbol']}: ${row['current_price']:.2f} "
                              f"(Vol: {row['volume']/1e6:.1f}M)")
            
        except Exception as e:
            logger.error(f"Error updating stock prices: {e}")
    
    def update_news_sentiment(self):
        """Update news sentiment data"""
        logger.info("Updating news sentiment...")
        
        try:
            # Run incremental news update
            result = self.etl_pipeline.run_incremental_update()
            
            if result['success']:
                logger.info("News sentiment updated successfully")
            else:
                logger.error("News sentiment update failed")
                
        except Exception as e:
            logger.error(f"Error updating news sentiment: {e}")
    
    def generate_predictions(self):
        """Generate fresh predictions"""
        logger.info("Generating fresh predictions...")
        
        try:
            # Load existing model if available
            model_file = os.path.join(Config.MODEL_DIR, 'latest_model.pkl')
            
            if os.path.exists(model_file):
                self.prediction_model.load_models(model_file)
                
                predictions = {}
                
                # Generate predictions for each symbol
                for symbol in self.symbols:
                    # Get latest features for prediction
                    features = self._get_latest_features(symbol)
                    
                    if not features.empty:
                        prediction = self.prediction_model.predict(symbol, features)
                        
                        if prediction:
                            predictions[symbol] = prediction
                            
                            # Log prediction
                            logger.info(f"{symbol} prediction: "
                                      f"${prediction['predicted_price']:.2f} "
                                      f"({prediction['predicted_direction']}, "
                                      f"{prediction['confidence']*100:.1f}% confidence)")
                
                # Store predictions in database
                if predictions:
                    self._store_predictions(predictions)
                    
            else:
                logger.warning("No trained model found. Run model training first.")
                
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    def _get_latest_features(self, symbol: str):
        """Get latest features for a symbol"""
        try:
            from config.database import execute_raw_query
            
            query = f"""
            SELECT s.*, t.sma_20, t.sma_50, t.ema_12, t.ema_26, t.macd, 
                   t.macd_signal, t.rsi, t.bb_upper, t.bb_middle, t.bb_lower
            FROM stock_data s
            LEFT JOIN technical_indicators t ON s.symbol = t.symbol AND s.date = t.date
            WHERE s.symbol = '{symbol}'
            ORDER BY s.date DESC
            LIMIT 1
            """
            
            data = execute_raw_query(query)
            
            if not data.empty:
                # Add required features
                data['price_change'] = 0
                data['day_of_week'] = datetime.now().weekday()
                data['month'] = datetime.now().month
                
            return data
            
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return pd.DataFrame()
    
    def _store_predictions(self, predictions: dict):
        """Store predictions in database"""
        try:
            prediction_records = []
            
            for symbol, pred in predictions.items():
                record = {
                    'symbol': symbol,
                    'prediction_date': datetime.now(),
                    'target_date': datetime.now() + timedelta(days=1),
                    'predicted_price': pred['predicted_price'],
                    'predicted_direction': pred['predicted_direction'],
                    'confidence_score': pred['confidence'],
                    'model_name': pred['model_type'],
                    'model_version': '1.0'
                }
                prediction_records.append(record)
            
            # Load predictions into database
            success = self.data_loader.load_predictions(prediction_records)
            
            if success:
                logger.info(f"Stored {len(prediction_records)} predictions in database")
            else:
                logger.error("Failed to store predictions in database")
                
        except Exception as e:
            logger.error(f"Error storing predictions: {e}")
    
    def health_check(self):
        """Perform system health check"""
        logger.info("Performing system health check...")
        
        try:
            # Check database connection
            session = get_db_session()
            if session:
                session.close()
                logger.info("✅ Database connection OK")
            else:
                logger.error("❌ Database connection failed")
                
            # Check data freshness
            from config.database import execute_raw_query
            
            latest_query = "SELECT MAX(date) as latest_date FROM stock_data"
            result = execute_raw_query(latest_query)
            
            if not result.empty:
                latest_date = result.iloc[0]['latest_date']
                days_old = (datetime.now().date() - latest_date.date()).days
                
                if days_old <= 1:
                    logger.info(f"✅ Data is fresh (latest: {latest_date.date()})")
                elif days_old <= 7:
                    logger.warning(f"⚠️ Data is {days_old} days old")
                else:
                    logger.error(f"❌ Data is stale ({days_old} days old)")
            
            # Check disk space
            import shutil
            
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb > 1:
                logger.info(f"✅ Disk space OK ({free_gb:.1f}GB free)")
            else:
                logger.warning(f"⚠️ Low disk space ({free_gb:.1f}GB free)")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def start_scheduler(self):
        """Start the real-time update scheduler"""
        logger.info("Starting real-time updater scheduler...")
        
        # Schedule updates
        schedule.every(15).minutes.do(self.update_stock_prices)
        schedule.every(1).hours.do(self.update_news_sentiment)
        schedule.every(2).hours.do(self.generate_predictions)
        schedule.every(6).hours.do(self.health_check)
        
        # Initial health check
        self.health_check()
        
        logger.info("Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")

def main():
    """Main function"""
    updater = RealTimeUpdater()
    
    import argparse
    parser = argparse.ArgumentParser(description='Real-time Stock Market Updater')
    parser.add_argument('--mode', choices=['prices', 'news', 'predictions', 'health', 'continuous'], 
                       default='continuous', help='Update mode')
    
    args = parser.parse_args()
    
    if args.mode == 'prices':
        updater.update_stock_prices()
    elif args.mode == 'news':
        updater.update_news_sentiment()
    elif args.mode == 'predictions':
        updater.generate_predictions()
    elif args.mode == 'health':
        updater.health_check()
    elif args.mode == 'continuous':
        updater.start_scheduler()

if __name__ == "__main__":
    main()
