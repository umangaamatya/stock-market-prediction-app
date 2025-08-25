"""
Stock Market Prediction Models
Combines technical indicators with sentiment analysis for predictions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from config.database import execute_raw_query

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictionModel:
    """Advanced stock prediction model using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.label_encoders = {}
        self.performance_metrics = {}
        self.is_trained = False
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'classifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            },
            'gradient_boosting': {
                'regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'classifier': None
            },
            'linear': {
                'regressor': LinearRegression(),
                'classifier': LogisticRegression(random_state=42, max_iter=1000)
            }
        }
    
    def prepare_training_data(self, 
                            symbols: List[str] = None, 
                            days_back: int = 365,
                            include_sentiment: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare training data by combining stock data, technical indicators, and sentiment
        """
        logger.info("Preparing training data")
        
        if symbols is None:
            symbols = Config.DEFAULT_STOCKS
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Build symbol list for SQL query
        symbol_list = "','".join(symbols)
        
        # Query stock data with technical indicators
        stock_query = f"""
        SELECT s.*, t.sma_20, t.sma_50, t.ema_12, t.ema_26, t.macd, t.macd_signal, 
               t.macd_histogram, t.rsi, t.bb_upper, t.bb_middle, t.bb_lower, t.volume_sma
        FROM stock_data s
        LEFT JOIN technical_indicators t ON s.symbol = t.symbol AND s.date = t.date
        WHERE s.symbol IN ('{symbol_list}')
        AND s.date >= '{start_date.strftime('%Y-%m-%d')}'
        AND s.date <= '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY s.symbol, s.date
        """
        
        stock_data = execute_raw_query(stock_query)
        
        if stock_data.empty:
            logger.error("No stock data found for training")
            return pd.DataFrame(), {}
        
        logger.info(f"Loaded {len(stock_data)} stock data records")
        
        # Add sentiment features if requested
        if include_sentiment:
            sentiment_data = self._get_sentiment_features(symbols, start_date, end_date)
            if not sentiment_data.empty:
                stock_data = self._merge_sentiment_data(stock_data, sentiment_data)
                logger.info("Sentiment features added")
        
        # Create additional features
        stock_data = self._create_advanced_features(stock_data)
        
        # Create target variables
        stock_data = self._create_targets(stock_data)
        
        # Remove rows with missing targets
        stock_data = stock_data.dropna(subset=['target_price', 'target_direction'])
        
        metadata = {
            'symbols': symbols,
            'date_range': (start_date, end_date),
            'total_records': len(stock_data),
            'features_count': len([col for col in stock_data.columns if col not in 
                                 ['symbol', 'date', 'target_price', 'target_direction', 'target_return']]),
            'include_sentiment': include_sentiment
        }
        
        logger.info(f"Training data prepared: {len(stock_data)} records with {metadata['features_count']} features")
        return stock_data, metadata
    
    def _get_sentiment_features(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get sentiment features from news data"""
        symbol_list = "','".join(symbols)
        
        sentiment_query = f"""
        SELECT 
            symbol,
            DATE(published_at) as date,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as news_count,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_news,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_news,
            MAX(sentiment_score) as max_sentiment,
            MIN(sentiment_score) as min_sentiment
        FROM news_data
        WHERE symbol IN ('{symbol_list}')
        AND DATE(published_at) >= '{start_date.strftime('%Y-%m-%d')}'
        AND DATE(published_at) <= '{end_date.strftime('%Y-%m-%d')}'
        GROUP BY symbol, DATE(published_at)
        """
        
        return execute_raw_query(sentiment_query)
    
    def _merge_sentiment_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Merge sentiment data with stock data"""
        # Ensure date columns are datetime
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Merge on symbol and date
        merged_data = stock_data.merge(sentiment_data, on=['symbol', 'date'], how='left')
        
        # Fill missing sentiment values
        sentiment_columns = ['avg_sentiment', 'news_count', 'positive_news', 'negative_news',
                           'max_sentiment', 'min_sentiment']
        
        for col in sentiment_columns:
            if col in merged_data.columns:
                if col in ['avg_sentiment', 'max_sentiment', 'min_sentiment']:
                    merged_data[col] = merged_data[col].fillna(0.0)
                else:
                    merged_data[col] = merged_data[col].fillna(0)
        
        return merged_data
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better predictions"""
        logger.info("Creating advanced features")
        
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Price-based features
        df['price_change'] = df.groupby('symbol')['close_price'].pct_change()
        df['price_volatility_5d'] = df.groupby('symbol')['close_price'].transform(
            lambda x: x.rolling(5).std()
        )
        df['price_momentum_3d'] = df.groupby('symbol')['close_price'].transform(
            lambda x: x.pct_change(3)
        )
        
        # Volume features
        if 'volume_sma' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
        else:
            df['volume_ratio'] = 1.0
        
        # Technical indicator combinations
        if 'rsi' in df.columns:
            df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
            df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_signal_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
        
        # Market context features
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['date']).dt.month
        
        # Sentiment-price interaction (if sentiment data available)
        if 'avg_sentiment' in df.columns:
            df['sentiment_price_interaction'] = df['avg_sentiment'] * df['price_change']
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        # Price prediction target (next day's closing price)
        df['target_price'] = df.groupby('symbol')['close_price'].shift(-1)
        
        # Return prediction target
        df['target_return'] = (df['target_price'] - df['close_price']) / df['close_price']
        
        # Direction prediction target
        df['target_direction'] = np.where(df['target_return'] > 0.01, 'up',
                                        np.where(df['target_return'] < -0.01, 'down', 'stable'))
        
        return df
    
    def train_models(self, 
                    training_data: pd.DataFrame, 
                    test_size: float = 0.2) -> Dict:
        """Train multiple prediction models"""
        logger.info("Starting model training")
        
        if training_data.empty:
            logger.error("No training data provided")
            return {}
        
        # Prepare features and targets
        feature_columns = self._get_feature_columns(training_data)
        X = training_data[feature_columns].copy()
        y_price = training_data['target_price'].copy()
        y_direction = training_data['target_direction'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Split data by symbols
        symbols = training_data['symbol'].unique()
        train_results = {}
        
        for symbol in symbols:
            logger.info(f"Training models for {symbol}")
            
            # Get symbol-specific data
            symbol_mask = training_data['symbol'] == symbol
            X_symbol = X[symbol_mask]
            y_price_symbol = y_price[symbol_mask]
            y_direction_symbol = y_direction[symbol_mask]
            
            # Remove samples with missing targets
            valid_mask = ~(y_price_symbol.isna() | y_direction_symbol.isna())
            X_symbol = X_symbol[valid_mask]
            y_price_symbol = y_price_symbol[valid_mask]
            y_direction_symbol = y_direction_symbol[valid_mask]
            
            if len(X_symbol) < 50:
                logger.warning(f"Not enough data for {symbol}, skipping")
                continue
            
            # Time series split
            split_idx = int(len(X_symbol) * (1 - test_size))
            X_train = X_symbol.iloc[:split_idx]
            X_test = X_symbol.iloc[split_idx:]
            y_price_train = y_price_symbol.iloc[:split_idx]
            y_price_test = y_price_symbol.iloc[split_idx:]
            y_direction_train = y_direction_symbol.iloc[:split_idx]
            y_direction_test = y_direction_symbol.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[symbol] = scaler
            
            symbol_results = {
                'regression': {},
                'classification': {}
            }
            
            # Train regression models
            for model_name, config in self.model_configs.items():
                if config['regressor'] is not None:
                    try:
                        model = config['regressor']
                        model.fit(X_train_scaled, y_price_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Metrics
                        mae = mean_absolute_error(y_price_test, y_pred)
                        mse = mean_squared_error(y_price_test, y_pred)
                        r2 = r2_score(y_price_test, y_pred)
                        
                        symbol_results['regression'][model_name] = {
                            'mae': mae,
                            'mse': mse,
                            'rmse': np.sqrt(mse),
                            'r2': r2
                        }
                        
                        # Store model
                        self.models[f'{symbol}_{model_name}_reg'] = model
                        
                        logger.info(f"{symbol} {model_name} regression - MAE: {mae:.4f}, R²: {r2:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error training {model_name} regression for {symbol}: {e}")
            
            # Train classification models
            for model_name, config in self.model_configs.items():
                if config['classifier'] is not None:
                    try:
                        model = config['classifier']
                        
                        label_encoder = LabelEncoder()
                        y_train_encoded = label_encoder.fit_transform(y_direction_train)
                        y_test_encoded = label_encoder.transform(y_direction_test)
                        
                        model.fit(X_train_scaled, y_train_encoded)
                        
                        # Predictions
                        y_pred_encoded = model.predict(X_test_scaled)
                        
                        # Metrics
                        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
                        
                        symbol_results['classification'][model_name] = {
                            'accuracy': accuracy
                        }
                        
                        # Store model and encoder
                        self.models[f'{symbol}_{model_name}_clf'] = model
                        self.label_encoders[symbol] = label_encoder
                        
                        logger.info(f"{symbol} {model_name} classification - Accuracy: {accuracy:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error training {model_name} classification for {symbol}: {e}")
            
            train_results[symbol] = symbol_results
        
        self.performance_metrics = train_results
        self.is_trained = True
        
        logger.info("Model training completed")
        return train_results
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns"""
        exclude_columns = [
            'symbol', 'date', 'target_price', 'target_direction', 'target_return',
            'created_at', 'updated_at', 'id'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns
    
    def predict(self, 
               symbol: str, 
               features: pd.DataFrame, 
               model_type: str = 'random_forest') -> Dict:
        """Make predictions for a symbol"""
        if not self.is_trained:
            logger.error("Models not trained. Call train_models() first.")
            return {}
        
        # Get models for the symbol
        reg_model_key = f'{symbol}_{model_type}_reg'
        clf_model_key = f'{symbol}_{model_type}_clf'
        
        if reg_model_key not in self.models or clf_model_key not in self.models:
            logger.error(f"No trained models found for {symbol} with {model_type}")
            return {}
        
        try:
            # Prepare features
            feature_columns = self._get_feature_columns(features)
            X = features[feature_columns].copy()
            X = X.fillna(X.median())
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Scale features
            scaler = self.scalers.get(symbol)
            if scaler is None:
                logger.error(f"No scaler found for {symbol}")
                return {}
            
            X_scaled = scaler.transform(X)
            
            # Price prediction
            reg_model = self.models[reg_model_key]
            predicted_price = reg_model.predict(X_scaled)[0]
            
            # Direction prediction
            clf_model = self.models[clf_model_key]
            predicted_direction_encoded = clf_model.predict(X_scaled)[0]
            label_encoder = self.label_encoders[symbol]
            predicted_direction = label_encoder.inverse_transform([predicted_direction_encoded])[0]
            
            # Confidence scores
            if hasattr(clf_model, 'predict_proba'):
                confidence_scores = clf_model.predict_proba(X_scaled)[0]
                confidence = np.max(confidence_scores)
            else:
                confidence = 0.5
            
            current_price = features['close_price'].iloc[0] if 'close_price' in features.columns else 0
            predicted_return = (predicted_price - current_price) / current_price if current_price > 0 else 0
            
            return {
                'symbol': symbol,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'predicted_return': predicted_return,
                'predicted_direction': predicted_direction,
                'confidence': confidence,
                'model_type': model_type,
                'prediction_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return {}
    
    def save_models(self, filepath: str = None):
        """Save trained models to disk"""
        if not self.is_trained:
            logger.warning("No trained models to save")
            return
        
        if filepath is None:
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            filepath = os.path.join(Config.MODEL_DIR, f'stock_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'saved_at': datetime.now()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.label_encoders = model_data['label_encoders']
            self.performance_metrics = model_data['performance_metrics']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all trained models"""
        return self.performance_metrics
