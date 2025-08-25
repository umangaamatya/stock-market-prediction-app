"""
Technical Indicators and Data Transformation Module
Uses PySpark and pandas for efficient calculation of technical indicators
"""
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import talib
from typing import Dict, List, Tuple
import logging
from config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicatorCalculator:
    """Calculate technical indicators for stock data"""
    
    def __init__(self):
        self.spark = None
        self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session"""
        try:
            self.spark = SparkSession.builder \
                .appName("StockMarketETL") \
                .config("spark.driver.memory", Config.SPARK_DRIVER_MEMORY) \
                .config("spark.executor.memory", Config.SPARK_EXECUTOR_MEMORY) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            logger.info("Spark session initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {e}")
            self.spark = None
    
    def calculate_indicators_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using pandas and TA-Lib
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Calculating technical indicators using pandas")
        
        if df.empty:
            return df
        
        # Ensure we have required columns
        required_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return df
        
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate indicators for each symbol
        result_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = self._calculate_symbol_indicators(symbol_df)
            result_dfs.append(symbol_df)
        
        # Combine all symbols
        final_df = pd.concat(result_dfs, ignore_index=True)
        logger.info(f"Technical indicators calculated for {len(final_df)} records")
        
        return final_df
    
    def _calculate_symbol_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators for a single symbol
        
        Args:
            df: DataFrame for single symbol
            
        Returns:
            DataFrame with indicators added
        """
        # Convert to numpy arrays for TA-Lib (more efficient)
        open_prices = df['open_price'].values.astype(float)
        high_prices = df['high_price'].values.astype(float)
        low_prices = df['low_price'].values.astype(float)
        close_prices = df['close_price'].values.astype(float)
        volumes = df['volume'].values.astype(float)
        
        # Simple Moving Averages
        df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
        df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
        df['sma_200'] = talib.SMA(close_prices, timeperiod=200)
        
        # Exponential Moving Averages
        df['ema_12'] = talib.EMA(close_prices, timeperiod=12)
        df['ema_26'] = talib.EMA(close_prices, timeperiod=26)
        df['ema_50'] = talib.EMA(close_prices, timeperiod=50)
        
        # MACD
        macd, macd_signal, macd_histogram = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # RSI
        df['rsi'] = talib.RSI(close_prices, timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # Williams %R
        df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Average True Range (ATR)
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Commodity Channel Index (CCI)
        df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Money Flow Index (MFI)
        df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=14)
        
        # Volume indicators
        df['volume_sma'] = talib.SMA(volumes, timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # On-Balance Volume (OBV)
        df['obv'] = talib.OBV(close_prices, volumes)
        
        # Accumulation/Distribution Line
        df['ad_line'] = talib.AD(high_prices, low_prices, close_prices, volumes)
        
        # Price-based indicators
        df['price_change'] = df['close_price'].pct_change()
        df['price_change_5d'] = df['close_price'].pct_change(periods=5)
        df['high_low_pct'] = (df['high_price'] - df['low_price']) / df['close_price'] * 100
        
        # Volatility indicators
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # Support and resistance levels (simplified)
        df['resistance_level'] = df['high_price'].rolling(window=20).max()
        df['support_level'] = df['low_price'].rolling(window=20).min()
        
        return df
    
    def calculate_indicators_spark(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using PySpark for large datasets
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        if not self.spark:
            logger.warning("Spark session not available, falling back to pandas")
            return self.calculate_indicators_pandas(df)
        
        logger.info("Calculating technical indicators using PySpark")
        
        try:
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Define window specifications
            window_spec = Window.partitionBy("symbol").orderBy("date")
            
            # Calculate moving averages using Spark SQL functions
            spark_df = spark_df.withColumn("sma_20", 
                                         avg("close_price").over(window_spec.rowsBetween(-19, 0)))
            
            spark_df = spark_df.withColumn("sma_50", 
                                         avg("close_price").over(window_spec.rowsBetween(-49, 0)))
            
            # Calculate price changes
            spark_df = spark_df.withColumn("prev_close", 
                                         lag("close_price").over(window_spec))
            
            spark_df = spark_df.withColumn("price_change", 
                                         (col("close_price") - col("prev_close")) / col("prev_close"))
            
            # Calculate volatility
            spark_df = spark_df.withColumn("volatility_20", 
                                         stddev("price_change").over(window_spec.rowsBetween(-19, 0)))
            
            # Convert back to pandas for more complex indicators
            result_df = spark_df.toPandas()
            
            # Calculate remaining indicators with pandas/TA-Lib
            result_df = self.calculate_indicators_pandas(result_df)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating indicators with Spark: {e}")
            logger.info("Falling back to pandas calculation")
            return self.calculate_indicators_pandas(df)
    
    def create_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for machine learning models
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with ML features
        """
        logger.info("Creating features for machine learning")
        
        # Create lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df.groupby('symbol')['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df.groupby('symbol')['volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df.groupby('symbol')['rsi'].shift(lag)
        
        # Create rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df.groupby('symbol')['close_price'].transform(lambda x: x.rolling(window).mean())
            df[f'close_std_{window}'] = df.groupby('symbol')['close_price'].transform(lambda x: x.rolling(window).std())
            df[f'volume_mean_{window}'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).mean())
        
        # Create cross-indicator features
        df['macd_signal_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
        df['price_above_sma20'] = np.where(df['close_price'] > df['sma_20'], 1, 0)
        df['price_above_sma50'] = np.where(df['close_price'] > df['sma_50'], 1, 0)
        
        # RSI signals
        df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
        df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
        
        # Bollinger Bands position
        df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume analysis
        df['volume_spike'] = np.where(df['volume_ratio'] > 2, 1, 0)
        
        # Price momentum
        df['momentum_5'] = df['close_price'] / df['close_lag_5'] - 1
        df['momentum_10'] = df['close_price'] / df['close_lag_10'] - 1
        
        return df
    
    def add_target_variables(self, df: pd.DataFrame, prediction_days: int = 1) -> pd.DataFrame:
        """
        Add target variables for prediction
        
        Args:
            df: DataFrame with features
            prediction_days: Number of days ahead to predict
            
        Returns:
            DataFrame with target variables
        """
        logger.info(f"Adding target variables for {prediction_days} days prediction")
        
        # Future price (target for regression)
        df['target_price'] = df.groupby('symbol')['close_price'].shift(-prediction_days)
        
        # Price direction (target for classification)
        df['target_return'] = (df['target_price'] - df['close_price']) / df['close_price']
        
        # Classification targets
        df['target_direction'] = np.where(df['target_return'] > 0.02, 'up',
                                        np.where(df['target_return'] < -0.02, 'down', 'stable'))
        
        df['target_binary'] = np.where(df['target_return'] > 0, 1, 0)
        
        return df
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning and validating data")
        
        initial_rows = len(df)
        
        # Remove rows with invalid prices
        df = df[(df['open_price'] > 0) & (df['high_price'] > 0) & 
                (df['low_price'] > 0) & (df['close_price'] > 0)]
        
        # Remove rows where high < low (data quality issues)
        df = df[df['high_price'] >= df['low_price']]
        
        # Remove extreme outliers (price changes > 50% in a day) - only if price_change exists
        if 'price_change' in df.columns:
            df = df[df['price_change'].abs() <= 0.5]
        
        # Fill NaN values for technical indicators (forward fill then backward fill)
        indicator_columns = [col for col in df.columns if col not in 
                           ['symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adj_close']]
        
        for col in indicator_columns:
            if col in df.columns:
                df[col] = df.groupby('symbol')[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows that still have NaN in critical columns
        critical_columns = ['close_price', 'volume']
        df = df.dropna(subset=critical_columns)
        
        final_rows = len(df)
        logger.info(f"Data cleaning completed: {initial_rows} -> {final_rows} rows ({final_rows/initial_rows*100:.1f}% retained)")
        
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict:
        """
        Prepare data for feature importance analysis
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            Dictionary with feature categories
        """
        
        feature_categories = {
            'price_features': [col for col in df.columns if 'close' in col or 'open' in col or 'high' in col or 'low' in col],
            'volume_features': [col for col in df.columns if 'volume' in col or 'obv' in col],
            'momentum_features': [col for col in df.columns if 'rsi' in col or 'macd' in col or 'momentum' in col],
            'volatility_features': [col for col in df.columns if 'volatility' in col or 'atr' in col or 'bb_' in col],
            'trend_features': [col for col in df.columns if 'sma' in col or 'ema' in col],
            'oscillator_features': [col for col in df.columns if 'stoch' in col or 'williams' in col or 'cci' in col],
        }
        
        return feature_categories
    
    def prepare_data_for_loading(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for loading into database tables
        
        Args:
            df: DataFrame with all indicators
            
        Returns:
            Tuple of (stock_data, technical_indicators) DataFrames
        """
        logger.info("Preparing data for database loading")
        
        # Stock data table columns
        stock_data_columns = ['symbol', 'date', 'open_price', 'high_price', 
                             'low_price', 'close_price', 'volume', 'adj_close']
        
        stock_data = df[stock_data_columns].copy()
        
        # Technical indicators table columns
        indicator_columns = [col for col in df.columns if col not in stock_data_columns]
        technical_indicators = df[['symbol', 'date'] + indicator_columns].copy()
        
        # Remove columns that are all NaN
        technical_indicators = technical_indicators.dropna(axis=1, how='all')
        
        logger.info(f"Prepared {len(stock_data)} stock data records and {len(technical_indicators)} technical indicator records")
        
        return stock_data, technical_indicators
    
    def close_spark_session(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session closed")


class DataTransformer:
    """Transform and clean raw stock market data"""
    
    def __init__(self):
        self.indicator_calculator = TechnicalIndicatorCalculator()
    
    def transform_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete transformation pipeline for stock data
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            Transformed DataFrame with technical indicators
        """
        logger.info("Starting complete data transformation pipeline")
        
        # Step 1: Basic cleaning and validation
        df = self.indicator_calculator.clean_and_validate_data(df)
        
        # Step 2: Calculate technical indicators
        df = self.indicator_calculator.calculate_indicators_pandas(df)
        
        # Step 3: Create ML features
        df = self.indicator_calculator.create_features_for_ml(df)
        
        # Step 4: Add target variables
        df = self.indicator_calculator.add_target_variables(df, prediction_days=1)
        
        # Step 5: Final cleaning
        df = self.indicator_calculator.clean_and_validate_data(df)
        
        logger.info("Data transformation pipeline completed")
        return df
    
    def transform_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform news data for loading
        
        Args:
            df: Raw news data DataFrame
            
        Returns:
            Transformed news DataFrame
        """
        logger.info("Transforming news data")
        
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title', 'url', 'symbol'])
        
        # Clean text fields
        text_columns = ['title', 'description', 'content']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
                # Truncate very long text
                df[col] = df[col].str[:2000]
        
        # Validate sentiment scores
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
            df['sentiment_score'] = df['sentiment_score'].clip(-1.0, 1.0)
        
        if 'sentiment_confidence' in df.columns:
            df['sentiment_confidence'] = df['sentiment_confidence'].fillna(0.0)
            df['sentiment_confidence'] = df['sentiment_confidence'].clip(0.0, 1.0)
        
        # Ensure proper date format
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            df = df.dropna(subset=['published_at'])
        
        logger.info(f"News data transformation completed: {len(df)} articles")
        return df
