"""
ETL Pipeline Orchestrator
Coordinates extraction, transformation, and loading of stock market data
"""
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from config.database import db_manager
from etl.extractors.stock_extractor import StockDataExtractor
from etl.extractors.news_extractor import NewsExtractor
from etl.transformers.technical_indicators import DataTransformer
from etl.loaders.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ETLPipeline:
    """Main ETL Pipeline for Stock Market Data"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or Config.DEFAULT_STOCKS
        
        # Initialize components
        self.stock_extractor = StockDataExtractor()
        self.news_extractor = NewsExtractor()
        self.data_transformer = DataTransformer()
        self.data_loader = DataLoader()
        
        # Pipeline status
        self.pipeline_status = {
            'start_time': None,
            'end_time': None,
            'success': False,
            'errors': [],
            'data_summary': {}
        }
    
    def run_full_pipeline(self, 
                         period: str = "1y",
                         include_news: bool = True,
                         news_days_back: int = 7) -> Dict:
        """
        Run the complete ETL pipeline
        
        Args:
            period: Time period for stock data extraction
            include_news: Whether to include news data extraction
            news_days_back: Number of days to look back for news
            
        Returns:
            Pipeline execution summary
        """
        logger.info("Starting ETL Pipeline execution")
        self.pipeline_status['start_time'] = datetime.now()
        
        try:
            # Step 1: Initialize database
            if not self._initialize_database():
                raise Exception("Database initialization failed")
            
            # Step 2: Extract stock data
            logger.info("Step 1: Extracting stock data")
            stock_data = self._extract_stock_data(period)
            if stock_data.empty:
                raise Exception("No stock data extracted")
            
            # Step 3: Extract news data (if enabled)
            news_data = pd.DataFrame()
            if include_news:
                logger.info("Step 2: Extracting news data")
                news_data = self._extract_news_data(news_days_back)
            
            # Step 4: Transform stock data
            logger.info("Step 3: Transforming stock data")
            transformed_stock_data = self._transform_stock_data(stock_data)
            
            # Step 5: Transform news data
            if not news_data.empty:
                logger.info("Step 4: Transforming news data")
                transformed_news_data = self._transform_news_data(news_data)
            else:
                transformed_news_data = pd.DataFrame()
            
            # Step 6: Load data into database
            logger.info("Step 5: Loading data into database")
            loading_success = self._load_data(transformed_stock_data, transformed_news_data)
            if not loading_success:
                raise Exception("Data loading failed")
            
            # Step 7: Generate pipeline summary
            self._generate_pipeline_summary()
            
            self.pipeline_status['success'] = True
            logger.info("ETL Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            self.pipeline_status['errors'].append(str(e))
            self.pipeline_status['success'] = False
        
        finally:
            self.pipeline_status['end_time'] = datetime.now()
            self._cleanup()
        
        return self.pipeline_status
    
    def run_incremental_update(self) -> Dict:
        """
        Run incremental data update (only new data since last update)
        
        Returns:
            Pipeline execution summary
        """
        logger.info("Starting incremental ETL update")
        self.pipeline_status['start_time'] = datetime.now()
        
        try:
            # Initialize database
            if not self._initialize_database():
                raise Exception("Database initialization failed")
            
            # Get latest data dates
            latest_stock_date = self.data_loader.get_latest_data_date('stock_data')
            latest_news_date = self.data_loader.get_latest_data_date('news_data')
            
            # Determine update period
            if latest_stock_date:
                start_date = (latest_stock_date + timedelta(days=1)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                logger.info(f"Updating stock data from {start_date} to {end_date}")
                
                # Extract incremental stock data
                stock_data = self.stock_extractor.extract_historical_range(
                    self.symbols, start_date, end_date
                )
                
                if not stock_data.empty:
                    # Transform and load stock data
                    transformed_stock_data = self._transform_stock_data(stock_data)
                    self._load_stock_data_only(transformed_stock_data)
            
            # Update news data (always get recent news)
            logger.info("Updating news data")
            news_data = self._extract_news_data(days_back=1)  # Just yesterday's news
            
            if not news_data.empty:
                transformed_news_data = self._transform_news_data(news_data)
                self._load_news_data_only(transformed_news_data)
            
            self._generate_pipeline_summary()
            self.pipeline_status['success'] = True
            logger.info("Incremental update completed successfully")
            
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            self.pipeline_status['errors'].append(str(e))
            self.pipeline_status['success'] = False
        
        finally:
            self.pipeline_status['end_time'] = datetime.now()
            self._cleanup()
        
        return self.pipeline_status
    
    def _initialize_database(self) -> bool:
        """Initialize database connection and create tables"""
        try:
            if db_manager.connect():
                db_manager.create_tables()
                logger.info("Database initialized successfully")
                return True
            else:
                logger.error("Failed to connect to database")
                return False
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False
    
    def _extract_stock_data(self, period: str) -> pd.DataFrame:
        """Extract stock data from various sources"""
        all_stock_data = []
        
        # Primary source: Yahoo Finance
        yahoo_data = self.stock_extractor.extract_yahoo_finance(self.symbols, period)
        if not yahoo_data.empty:
            all_stock_data.append(yahoo_data)
            logger.info(f"Extracted {len(yahoo_data)} records from Yahoo Finance")
        
        # Secondary source: Alpha Vantage (if API key available)
        if Config.ALPHA_VANTAGE_API_KEY:
            alpha_vantage_data = self.stock_extractor.extract_alpha_vantage(self.symbols)
            if not alpha_vantage_data.empty:
                all_stock_data.append(alpha_vantage_data)
                logger.info(f"Extracted {len(alpha_vantage_data)} records from Alpha Vantage")
        
        # Combine data from all sources
        if all_stock_data:
            combined_data = pd.concat(all_stock_data, ignore_index=True)
            
            # Remove duplicates (prefer Yahoo Finance data)
            combined_data = combined_data.drop_duplicates(
                subset=['symbol', 'date'], 
                keep='first'
            )
            
            logger.info(f"Total stock data records after deduplication: {len(combined_data)}")
            return combined_data
        else:
            logger.warning("No stock data extracted from any source")
            return pd.DataFrame()
    
    def _extract_news_data(self, days_back: int) -> pd.DataFrame:
        """Extract news data from various sources"""
        if not Config.NEWS_API_KEY:
            logger.warning("News API key not configured, skipping news extraction")
            return pd.DataFrame()
        
        # Extract combined news from all available sources
        news_data = self.news_extractor.extract_combined_news(
            self.symbols, 
            days_back=days_back
        )
        
        if not news_data.empty:
            logger.info(f"Extracted {len(news_data)} news articles")
        else:
            logger.warning("No news data extracted")
        
        return news_data
    
    def _transform_stock_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Transform stock data with technical indicators"""
        if stock_data.empty:
            return stock_data
        
        # Apply complete transformation pipeline
        transformed_data = self.data_transformer.transform_stock_data(stock_data)
        
        logger.info(f"Stock data transformation completed: {len(transformed_data)} records")
        return transformed_data
    
    def _transform_news_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Transform news data"""
        if news_data.empty:
            return news_data
        
        # Apply news data transformation
        transformed_data = self.data_transformer.transform_news_data(news_data)
        
        logger.info(f"News data transformation completed: {len(transformed_data)} records")
        return transformed_data
    
    def _load_data(self, stock_data: pd.DataFrame, news_data: pd.DataFrame) -> bool:
        """Load all transformed data into database"""
        try:
            success_count = 0
            
            # Load stock data
            if not stock_data.empty:
                # Split stock data and technical indicators
                stock_df, indicators_df = self.data_transformer.indicator_calculator.prepare_data_for_loading(stock_data)
                
                if self.data_loader.load_stock_data(stock_df):
                    success_count += 1
                    logger.info("Stock data loaded successfully")
                
                if self.data_loader.load_technical_indicators(indicators_df):
                    success_count += 1
                    logger.info("Technical indicators loaded successfully")
            
            # Load news data
            if not news_data.empty:
                if self.data_loader.load_news_data(news_data):
                    success_count += 1
                    logger.info("News data loaded successfully")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return False
    
    def _load_stock_data_only(self, stock_data: pd.DataFrame) -> bool:
        """Load only stock data (for incremental updates)"""
        try:
            if stock_data.empty:
                return True
            
            stock_df, indicators_df = self.data_transformer.indicator_calculator.prepare_data_for_loading(stock_data)
            
            stock_success = self.data_loader.load_stock_data(stock_df)
            indicators_success = self.data_loader.load_technical_indicators(indicators_df)
            
            return stock_success and indicators_success
            
        except Exception as e:
            logger.error(f"Stock data loading error: {e}")
            return False
    
    def _load_news_data_only(self, news_data: pd.DataFrame) -> bool:
        """Load only news data (for incremental updates)"""
        try:
            if news_data.empty:
                return True
            
            return self.data_loader.load_news_data(news_data)
            
        except Exception as e:
            logger.error(f"News data loading error: {e}")
            return False
    
    def _generate_pipeline_summary(self):
        """Generate summary of pipeline execution"""
        try:
            self.pipeline_status['data_summary'] = self.data_loader.get_data_summary()
            
            # Calculate execution time
            if self.pipeline_status['start_time'] and self.pipeline_status['end_time']:
                execution_time = self.pipeline_status['end_time'] - self.pipeline_status['start_time']
                self.pipeline_status['execution_time_minutes'] = execution_time.total_seconds() / 60
            
            logger.info(f"Pipeline summary: {self.pipeline_status['data_summary']}")
            
        except Exception as e:
            logger.error(f"Error generating pipeline summary: {e}")
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            # Close database connections
            if self.data_loader:
                self.data_loader.close()
            
            # Close Spark session if used
            if hasattr(self.data_transformer.indicator_calculator, 'spark'):
                self.data_transformer.indicator_calculator.close_spark_session()
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return self.pipeline_status
    
    def validate_configuration(self) -> Dict:
        """Validate pipeline configuration"""
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check required configurations
        if not Config.POSTGRES_PASSWORD or Config.POSTGRES_PASSWORD == 'password':
            validation_result['errors'].append("PostgreSQL password not configured")
            validation_result['valid'] = False
        
        if not Config.NEWS_API_KEY:
            validation_result['warnings'].append("News API key not configured - news extraction will be skipped")
        
        if not Config.ALPHA_VANTAGE_API_KEY:
            validation_result['warnings'].append("Alpha Vantage API key not configured - only Yahoo Finance will be used")
        
        # Test database connection
        try:
            db_test = db_manager.connect()
            if not db_test:
                validation_result['errors'].append("Cannot connect to PostgreSQL database")
                validation_result['valid'] = False
        except Exception as e:
            validation_result['errors'].append(f"Database connection test failed: {e}")
            validation_result['valid'] = False
        
        return validation_result


def main():
    """Main function for running ETL pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Market ETL Pipeline')
    parser.add_argument('--mode', choices=['full', 'incremental'], default='full',
                       help='Pipeline execution mode')
    parser.add_argument('--symbols', nargs='*', default=Config.DEFAULT_STOCKS,
                       help='Stock symbols to process')
    parser.add_argument('--period', default='1y',
                       help='Time period for stock data (full mode only)')
    parser.add_argument('--skip-news', action='store_true',
                       help='Skip news data extraction')
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration only')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ETLPipeline(symbols=args.symbols)
    
    # Validate configuration if requested
    if args.validate:
        validation = pipeline.validate_configuration()
        print("\n=== Configuration Validation ===")
        print(f"Valid: {validation['valid']}")
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        if validation['errors']:
            print("\nErrors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        return 0 if validation['valid'] else 1
    
    # Run pipeline
    if args.mode == 'full':
        result = pipeline.run_full_pipeline(
            period=args.period,
            include_news=not args.skip_news
        )
    else:
        result = pipeline.run_incremental_update()
    
    # Print results
    print("\n=== ETL Pipeline Results ===")
    print(f"Success: {result['success']}")
    print(f"Start Time: {result['start_time']}")
    print(f"End Time: {result['end_time']}")
    
    if result.get('execution_time_minutes'):
        print(f"Execution Time: {result['execution_time_minutes']:.2f} minutes")
    
    if result.get('errors'):
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result.get('data_summary'):
        print("\nData Summary:")
        for table, summary in result['data_summary'].items():
            print(f"  {table}: {summary}")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    exit(main())
