"""
Data Loader Module
Loads transformed data into PostgreSQL database
"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from config.database import (
    DatabaseManager, StockData, NewsData, TechnicalIndicators, 
    Predictions, ModelPerformance, bulk_insert_data
)
from config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load data into PostgreSQL database"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.session = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        if self.db_manager.connect():
            self.session = self.db_manager.get_session()
            logger.info("Database connection established for data loading")
        else:
            logger.error("Failed to establish database connection")
    
    def load_stock_data(self, df: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        Load stock price data into database
        
        Args:
            df: DataFrame with stock data
            batch_size: Number of records to insert in each batch
            
        Returns:
            Success status
        """
        if df.empty:
            logger.warning("No stock data to load")
            return True
        
        logger.info(f"Loading {len(df)} stock data records")
        
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Process in batches
            total_loaded = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Handle duplicates by updating existing records
                success = self._upsert_stock_data(batch)
                if success:
                    total_loaded += len(batch)
                    logger.info(f"Loaded batch: {total_loaded}/{len(records)} records")
                else:
                    logger.error(f"Failed to load batch starting at index {i}")
                    return False
            
            logger.info(f"Successfully loaded {total_loaded} stock data records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            return False
    
    def _upsert_stock_data(self, records: List[Dict]) -> bool:
        """
        Insert or update stock data records
        
        Args:
            records: List of stock data records
            
        Returns:
            Success status
        """
        try:
            for record in records:
                # Check if record already exists
                existing = self.session.query(StockData).filter(
                    StockData.symbol == record['symbol'],
                    StockData.date == record['date']
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if key not in ['symbol', 'date']:  # Don't update primary key components
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Insert new record
                    new_record = StockData(**record)
                    self.session.add(new_record)
            
            self.session.commit()
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error upserting stock data: {e}")
            return False
    
    def load_news_data(self, df: pd.DataFrame, batch_size: int = 500) -> bool:
        """
        Load news and sentiment data into database
        
        Args:
            df: DataFrame with news data
            batch_size: Number of records to insert in each batch
            
        Returns:
            Success status
        """
        if df.empty:
            logger.warning("No news data to load")
            return True
        
        logger.info(f"Loading {len(df)} news records")
        
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Process in batches
            total_loaded = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Clean and validate batch data
                cleaned_batch = self._clean_news_batch(batch)
                
                if cleaned_batch:
                    success = bulk_insert_data(NewsData, cleaned_batch, self.session)
                    if success:
                        total_loaded += len(cleaned_batch)
                        logger.info(f"Loaded batch: {total_loaded}/{len(records)} records")
                    else:
                        logger.error(f"Failed to load batch starting at index {i}")
                        return False
            
            logger.info(f"Successfully loaded {total_loaded} news records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            return False
    
    def _clean_news_batch(self, records: List[Dict]) -> List[Dict]:
        """
        Clean and validate news records before insertion
        
        Args:
            records: List of news records
            
        Returns:
            Cleaned list of records
        """
        cleaned_records = []
        
        for record in records:
            try:
                # Skip records without essential fields
                if not record.get('title') or not record.get('symbol'):
                    continue
                
                # Clean and truncate text fields
                record['title'] = str(record.get('title', ''))[:500]
                record['description'] = str(record.get('description', ''))[:2000]
                record['content'] = str(record.get('content', ''))[:5000]
                record['source'] = str(record.get('source', ''))[:100]
                record['author'] = str(record.get('author', ''))[:200]
                record['url'] = str(record.get('url', ''))[:500]
                
                # Handle sentiment values
                record['sentiment_score'] = float(record.get('sentiment_score', 0.0))
                record['sentiment_score'] = max(-1.0, min(1.0, record['sentiment_score']))
                
                record['sentiment_confidence'] = float(record.get('sentiment_confidence', 0.0))
                record['sentiment_confidence'] = max(0.0, min(1.0, record['sentiment_confidence']))
                
                record['sentiment_label'] = str(record.get('sentiment_label', 'neutral'))
                
                # Ensure published_at is datetime
                if pd.isna(record.get('published_at')):
                    record['published_at'] = datetime.utcnow()
                
                cleaned_records.append(record)
                
            except Exception as e:
                logger.warning(f"Skipping invalid news record: {e}")
                continue
        
        return cleaned_records
    
    def load_technical_indicators(self, df: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        Load technical indicators into database
        
        Args:
            df: DataFrame with technical indicators
            batch_size: Number of records to insert in each batch
            
        Returns:
            Success status
        """
        if df.empty:
            logger.warning("No technical indicators to load")
            return True
        
        logger.info(f"Loading {len(df)} technical indicator records")
        
        try:
            # Select only columns that exist in the TechnicalIndicators table
            indicator_columns = ['symbol', 'date', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                               'macd', 'macd_signal', 'macd_histogram', 'rsi',
                               'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma']
            
            # Filter dataframe to only include available columns
            available_columns = [col for col in indicator_columns if col in df.columns]
            df_filtered = df[available_columns].copy()
            
            # Handle NaN values
            df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan)
            df_filtered = df_filtered.fillna(0.0)  # Replace NaN with 0 for indicators
            
            # Convert to records
            records = df_filtered.to_dict('records')
            
            # Process in batches
            total_loaded = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                success = self._upsert_technical_indicators(batch)
                if success:
                    total_loaded += len(batch)
                    logger.info(f"Loaded batch: {total_loaded}/{len(records)} records")
                else:
                    logger.error(f"Failed to load batch starting at index {i}")
                    return False
            
            logger.info(f"Successfully loaded {total_loaded} technical indicator records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading technical indicators: {e}")
            return False
    
    def _upsert_technical_indicators(self, records: List[Dict]) -> bool:
        """
        Insert or update technical indicator records
        
        Args:
            records: List of technical indicator records
            
        Returns:
            Success status
        """
        try:
            for record in records:
                # Check if record already exists
                existing = self.session.query(TechnicalIndicators).filter(
                    TechnicalIndicators.symbol == record['symbol'],
                    TechnicalIndicators.date == record['date']
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if key not in ['symbol', 'date'] and hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    # Insert new record
                    new_record = TechnicalIndicators(**record)
                    self.session.add(new_record)
            
            self.session.commit()
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error upserting technical indicators: {e}")
            return False
    
    def load_predictions(self, predictions: List[Dict]) -> bool:
        """
        Load model predictions into database
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Success status
        """
        if not predictions:
            logger.warning("No predictions to load")
            return True
        
        logger.info(f"Loading {len(predictions)} predictions")
        
        try:
            cleaned_predictions = []
            for pred in predictions:
                cleaned_pred = {
                    'symbol': pred['symbol'],
                    'prediction_date': pred.get('prediction_date', datetime.utcnow()),
                    'target_date': pred['target_date'],
                    'predicted_price': float(pred.get('predicted_price', 0.0)),
                    'predicted_direction': str(pred.get('predicted_direction', 'stable')),
                    'confidence_score': float(pred.get('confidence_score', 0.0)),
                    'model_name': str(pred.get('model_name', 'unknown')),
                    'model_version': str(pred.get('model_version', '1.0')),
                    'features_used': str(pred.get('features_used', ''))
                }
                cleaned_predictions.append(cleaned_pred)
            
            success = bulk_insert_data(Predictions, cleaned_predictions, self.session)
            if success:
                logger.info(f"Successfully loaded {len(cleaned_predictions)} predictions")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return False
    
    def load_model_performance(self, performance_data: Dict) -> bool:
        """
        Load model performance metrics into database
        
        Args:
            performance_data: Dictionary with performance metrics
            
        Returns:
            Success status
        """
        logger.info("Loading model performance data")
        
        try:
            performance_record = ModelPerformance(**performance_data)
            self.session.add(performance_record)
            self.session.commit()
            
            logger.info("Successfully loaded model performance data")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error loading model performance: {e}")
            return False
    
    def get_latest_data_date(self, table_name: str, symbol: str = None) -> Optional[datetime]:
        """
        Get the latest date for which we have data
        
        Args:
            table_name: Name of the table ('stock_data', 'news_data', etc.)
            symbol: Optional symbol to filter by
            
        Returns:
            Latest date or None
        """
        try:
            if table_name == 'stock_data':
                query = self.session.query(StockData.date)
                if symbol:
                    query = query.filter(StockData.symbol == symbol)
                latest = query.order_by(StockData.date.desc()).first()
                
            elif table_name == 'news_data':
                query = self.session.query(NewsData.published_at)
                if symbol:
                    query = query.filter(NewsData.symbol == symbol)
                latest = query.order_by(NewsData.published_at.desc()).first()
                
            elif table_name == 'technical_indicators':
                query = self.session.query(TechnicalIndicators.date)
                if symbol:
                    query = query.filter(TechnicalIndicators.symbol == symbol)
                latest = query.order_by(TechnicalIndicators.date.desc()).first()
                
            else:
                return None
            
            return latest[0] if latest else None
            
        except Exception as e:
            logger.error(f"Error getting latest data date: {e}")
            return None
    
    def clear_old_data(self, table_name: str, days_to_keep: int = 365) -> bool:
        """
        Clear old data to manage database size
        
        Args:
            table_name: Name of the table to clean
            days_to_keep: Number of days of data to keep
            
        Returns:
            Success status
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        logger.info(f"Clearing {table_name} data older than {cutoff_date}")
        
        try:
            if table_name == 'news_data':
                deleted = self.session.query(NewsData).filter(
                    NewsData.published_at < cutoff_date
                ).delete(synchronize_session=False)
                
            elif table_name == 'predictions':
                deleted = self.session.query(Predictions).filter(
                    Predictions.prediction_date < cutoff_date
                ).delete(synchronize_session=False)
                
            else:
                logger.warning(f"Clear operation not implemented for table: {table_name}")
                return True
            
            self.session.commit()
            logger.info(f"Cleared {deleted} old records from {table_name}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error clearing old data from {table_name}: {e}")
            return False
    
    def get_data_summary(self) -> Dict:
        """
        Get summary of data in database
        
        Returns:
            Dictionary with data summary
        """
        try:
            summary = {}
            
            # Stock data summary
            stock_count = self.session.query(StockData).count()
            stock_symbols = self.session.query(StockData.symbol.distinct()).count()
            latest_stock = self.session.query(StockData.date).order_by(StockData.date.desc()).first()
            
            summary['stock_data'] = {
                'total_records': stock_count,
                'symbols_count': stock_symbols,
                'latest_date': latest_stock[0] if latest_stock else None
            }
            
            # News data summary
            news_count = self.session.query(NewsData).count()
            latest_news = self.session.query(NewsData.published_at).order_by(NewsData.published_at.desc()).first()
            
            summary['news_data'] = {
                'total_records': news_count,
                'latest_date': latest_news[0] if latest_news else None
            }
            
            # Technical indicators summary
            indicators_count = self.session.query(TechnicalIndicators).count()
            latest_indicators = self.session.query(TechnicalIndicators.date).order_by(TechnicalIndicators.date.desc()).first()
            
            summary['technical_indicators'] = {
                'total_records': indicators_count,
                'latest_date': latest_indicators[0] if latest_indicators else None
            }
            
            # Predictions summary
            predictions_count = self.session.query(Predictions).count()
            
            summary['predictions'] = {
                'total_records': predictions_count
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()
            logger.info("Database session closed")
