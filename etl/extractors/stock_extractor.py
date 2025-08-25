"""
Stock Data Extractor Module
Extracts stock price data from various APIs (Yahoo Finance, Alpha Vantage)
"""
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
from config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataExtractor:
    """Extract stock price data from multiple sources"""
    
    def __init__(self):
        self.alpha_vantage_key = Config.ALPHA_VANTAGE_API_KEY
        self.rate_limit_delay = 12  # Alpha Vantage allows 5 calls per minute
        
    def extract_yahoo_finance(self, 
                            symbols: List[str], 
                            period: str = "1y", 
                            interval: str = "1d") -> pd.DataFrame:
        """
        Extract stock data from Yahoo Finance using yfinance
        
        Args:
            symbols: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with stock data
        """
        logger.info(f"Extracting Yahoo Finance data for symbols: {symbols}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Download stock data
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period, interval=interval)
                
                if hist.empty:
                    logger.warning(f"No data found for symbol: {symbol}")
                    continue
                
                # Reset index to get date as column
                hist.reset_index(inplace=True)
                hist['symbol'] = symbol
                
                # Rename columns to match our database schema
                hist.rename(columns={
                    'Date': 'date',
                    'Open': 'open_price',
                    'High': 'high_price',
                    'Low': 'low_price',
                    'Close': 'close_price',
                    'Volume': 'volume'
                }, inplace=True)
                
                # Add adjusted close if not present
                if 'Adj Close' in hist.columns:
                    hist.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
                else:
                    hist['adj_close'] = hist['close_price']
                
                # Select only required columns
                columns_needed = ['symbol', 'date', 'open_price', 'high_price', 
                                'low_price', 'close_price', 'volume', 'adj_close']
                hist = hist[columns_needed]
                
                all_data.append(hist)
                logger.info(f"Successfully extracted {len(hist)} records for {symbol}")
                
                # Small delay to be respectful to the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error extracting data for {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total records extracted: {len(combined_data)}")
            return combined_data
        else:
            logger.warning("No data extracted from Yahoo Finance")
            return pd.DataFrame()
    
    def extract_alpha_vantage(self, 
                            symbols: List[str], 
                            function: str = "TIME_SERIES_DAILY_ADJUSTED",
                            outputsize: str = "full") -> pd.DataFrame:
        """
        Extract stock data from Alpha Vantage API
        
        Args:
            symbols: List of stock symbols
            function: API function (TIME_SERIES_DAILY_ADJUSTED, etc.)
            outputsize: compact (100 data points) or full (20+ years)
        
        Returns:
            DataFrame with stock data
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not provided, skipping")
            return pd.DataFrame()
        
        logger.info(f"Extracting Alpha Vantage data for symbols: {symbols}")
        
        all_data = []
        base_url = "https://www.alphavantage.co/query"
        
        for symbol in symbols:
            try:
                params = {
                    'function': function,
                    'symbol': symbol,
                    'outputsize': outputsize,
                    'apikey': self.alpha_vantage_key
                }
                
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                    continue
                
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage note for {symbol}: {data['Note']}")
                    continue
                
                # Extract time series data
                time_series_key = [key for key in data.keys() if 'Time Series' in key]
                if not time_series_key:
                    logger.error(f"No time series data found for {symbol}")
                    continue
                
                time_series = data[time_series_key[0]]
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.reset_index(inplace=True)
                df.columns = ['date'] + list(df.columns[1:])
                
                # Convert date column
                df['date'] = pd.to_datetime(df['date'])
                df['symbol'] = symbol
                
                # Rename columns based on function used
                if function == "TIME_SERIES_DAILY_ADJUSTED":
                    df.rename(columns={
                        '1. open': 'open_price',
                        '2. high': 'high_price',
                        '3. low': 'low_price',
                        '4. close': 'close_price',
                        '5. adjusted close': 'adj_close',
                        '6. volume': 'volume'
                    }, inplace=True)
                
                # Convert price columns to float
                price_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'adj_close']
                for col in price_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Convert volume to integer
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')
                
                # Select only required columns
                columns_needed = ['symbol', 'date', 'open_price', 'high_price', 
                                'low_price', 'close_price', 'volume', 'adj_close']
                available_columns = [col for col in columns_needed if col in df.columns]
                df = df[available_columns]
                
                all_data.append(df)
                logger.info(f"Successfully extracted {len(df)} records for {symbol} from Alpha Vantage")
                
                # Rate limiting - Alpha Vantage allows 5 calls per minute
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error extracting Alpha Vantage data for {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total Alpha Vantage records extracted: {len(combined_data)}")
            return combined_data
        else:
            logger.warning("No data extracted from Alpha Vantage")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get additional stock information (company name, sector, etc.)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract relevant information
            stock_info = {
                'symbol': symbol,
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'country': info.get('country', ''),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'website': info.get('website', '')
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            return {'symbol': symbol}
    
    def extract_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Extract real-time/current stock data
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with current stock data
        """
        logger.info(f"Extracting real-time data for symbols: {symbols}")
        
        try:
            # Create tickers object
            tickers = yf.Tickers(' '.join(symbols))
            
            current_data = []
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    
                    # Get current price and basic info
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    
                    if current_price:
                        data_point = {
                            'symbol': symbol,
                            'date': datetime.now(),
                            'current_price': current_price,
                            'previous_close': info.get('previousClose', 0),
                            'open_price': info.get('regularMarketOpen', 0),
                            'high_price': info.get('dayHigh', 0),
                            'low_price': info.get('dayLow', 0),
                            'volume': info.get('regularMarketVolume', 0),
                            'market_cap': info.get('marketCap', 0)
                        }
                        
                        current_data.append(data_point)
                        
                except Exception as e:
                    logger.error(f"Error getting real-time data for {symbol}: {e}")
                    continue
            
            if current_data:
                df = pd.DataFrame(current_data)
                logger.info(f"Successfully extracted real-time data for {len(df)} symbols")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting real-time data: {e}")
            return pd.DataFrame()
    
    def extract_historical_range(self, 
                               symbols: List[str],
                               start_date: str,
                               end_date: str) -> pd.DataFrame:
        """
        Extract historical data for a specific date range
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical stock data
        """
        logger.info(f"Extracting historical data from {start_date} to {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date, end=end_date)
                
                if hist.empty:
                    logger.warning(f"No historical data found for {symbol}")
                    continue
                
                # Process the data similar to extract_yahoo_finance method
                hist.reset_index(inplace=True)
                hist['symbol'] = symbol
                
                hist.rename(columns={
                    'Date': 'date',
                    'Open': 'open_price',
                    'High': 'high_price',
                    'Low': 'low_price',
                    'Close': 'close_price',
                    'Volume': 'volume'
                }, inplace=True)
                
                if 'Adj Close' in hist.columns:
                    hist.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
                else:
                    hist['adj_close'] = hist['close_price']
                
                columns_needed = ['symbol', 'date', 'open_price', 'high_price', 
                                'low_price', 'close_price', 'volume', 'adj_close']
                hist = hist[columns_needed]
                
                all_data.append(hist)
                
            except Exception as e:
                logger.error(f"Error extracting historical data for {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total historical records extracted: {len(combined_data)}")
            return combined_data
        else:
            return pd.DataFrame()
