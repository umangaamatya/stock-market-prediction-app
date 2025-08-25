"""
Database utilities and schema management for Stock Market Prediction Application
"""
import os
import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from config.config import Config
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or Config.get_database_uri()
        self.engine = None
        self.SessionLocal = None
        self.metadata = MetaData()
        
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.engine = create_engine(
                self.database_url,
                echo=Config.FLASK_DEBUG,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("Database connection established successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self._create_indexes()
            logger.info("Database tables created successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            with self.engine.connect() as conn:
                # Create indexes for better query performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_date ON stock_data (symbol, date DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_news_data_date ON news_data (published_at DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions (symbol, prediction_date DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date ON technical_indicators (symbol, date DESC);"
                ]
                
                for index in indexes:
                    conn.execute(text(index))
                conn.commit()
                
        except SQLAlchemyError as e:
            logger.warning(f"Index creation warning: {e}")

# Database Models
class StockData(Base):
    """Stock price data table"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NewsData(Base):
    """News and sentiment data table"""
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text)
    content = Column(Text)
    source = Column(String(100))
    author = Column(String(200))
    url = Column(String(500))
    published_at = Column(DateTime, nullable=False)
    sentiment_score = Column(Float)  # -1 to 1, negative to positive
    sentiment_label = Column(String(20))  # negative, neutral, positive
    sentiment_confidence = Column(Float)  # 0 to 1
    created_at = Column(DateTime, default=datetime.utcnow)

class TechnicalIndicators(Base):
    """Technical analysis indicators table"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Moving averages
    sma_20 = Column(Float)  # Simple Moving Average 20 days
    sma_50 = Column(Float)  # Simple Moving Average 50 days
    ema_12 = Column(Float)  # Exponential Moving Average 12 days
    ema_26 = Column(Float)  # Exponential Moving Average 26 days
    
    # MACD indicators
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    
    # RSI
    rsi = Column(Float)  # Relative Strength Index
    
    # Bollinger Bands
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    
    # Volume indicators
    volume_sma = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Predictions(Base):
    """Stock price predictions table"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)  # Date being predicted
    
    # Predictions
    predicted_price = Column(Float)
    predicted_direction = Column(String(10))  # up, down, stable
    confidence_score = Column(Float)  # 0 to 1
    
    # Model information
    model_name = Column(String(50))
    model_version = Column(String(20))
    features_used = Column(Text)  # JSON string of features used
    
    # Actual values (filled in later for model evaluation)
    actual_price = Column(Float)
    actual_direction = Column(String(10))
    prediction_accuracy = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelPerformance(Base):
    """Model performance metrics table"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    symbol = Column(String(10))
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Square Error
    
    # Evaluation period
    evaluation_start = Column(DateTime)
    evaluation_end = Column(DateTime)
    total_predictions = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Utility functions
def get_db_session() -> Session:
    """Get database session - utility function"""
    db = DatabaseManager()
    if db.connect():
        return db.get_session()
    return None

def execute_raw_query(query: str, params: dict = None) -> pd.DataFrame:
    """Execute raw SQL query and return DataFrame"""
    db = DatabaseManager()
    if db.connect():
        try:
            return pd.read_sql_query(query, db.engine, params=params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Global database manager instance for pipeline modules
# Note: This is initialized lazily on first connect()
db_manager = DatabaseManager()

def bulk_insert_data(table_class, data: list, session: Session = None):
    """Bulk insert data into specified table"""
    if session is None:
        db = DatabaseManager()
        if not db.connect():
            return False
        session = db.get_session()
        close_session = True
    else:
        close_session = False
    
    try:
        session.bulk_insert_mappings(table_class, data)
        session.commit()
        logger.info(f"Successfully inserted {len(data)} records into {table_class.__tablename__}")
        return True
        
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Bulk insert failed: {e}")
        return False
        
    finally:
        if close_session:
            session.close()

# Initialize database manager instance
db_manager = DatabaseManager()
