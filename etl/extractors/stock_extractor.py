"""
Stock Data Extractor Module
Extracts stock price data from various APIs (Yahoo Finance, Alpha Vantage)
"""
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from pandas.tseries.offsets import DateOffset
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
import random
from config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataExtractor:
    """Extract stock price data from multiple sources"""
    
    def __init__(self):
        self.alpha_vantage_key = Config.ALPHA_VANTAGE_API_KEY
        # Treat placeholder keys as missing
        if self.alpha_vantage_key and str(self.alpha_vantage_key).lower().startswith("your_"):
            logger.warning("Alpha Vantage API key appears to be a placeholder; disabling Alpha Vantage extraction.")
            self.alpha_vantage_key = None
        self.rate_limit_delay = 12  # Alpha Vantage allows 5 calls per minute
        
    def _make_http_session(self) -> requests.Session:
        """Create a resilient HTTP session with retries and browser-like headers."""
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"])  # urllib3 v2 requires frozenset
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        })
        return session

    def _period_to_dates(self, period: str) -> (datetime, datetime):
        """Convert a yfinance-style period string into start/end datetimes."""
        now = datetime.utcnow()
        period = (period or "1y").lower()
        if period.endswith('y'):
            years = int(period[:-1] or 1)
            start = now - DateOffset(years=years)
        elif period.endswith('mo'):
            months = int(period[:-2] or 1)
            start = now - DateOffset(months=months)
        elif period.endswith('m'):
            # minutes not supported for stooq; default to 30 days
            start = now - timedelta(days=30)
        elif period.endswith('d'):
            days = int(period[:-1] or 1)
            start = now - timedelta(days=days)
        else:
            # default 1 year
            start = now - DateOffset(years=1)
        return start.to_pydatetime() if hasattr(start, 'to_pydatetime') else start, now

    def extract_stooq(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Fallback extractor using Stooq via pandas-datareader (daily bars)."""
        try:
            from pandas_datareader import data as pdr_data
        except ImportError:
            logger.warning("pandas-datareader not installed; cannot use Stooq fallback")
            return pd.DataFrame()
        
        start, end = self._period_to_dates(period)
        all_data: List[pd.DataFrame] = []
        
        for symbol in symbols:
            try:
                df = pdr_data.DataReader(symbol, 'stooq', start, end)
                if df is None or df.empty:
                    logger.warning(f"No Stooq data for {symbol}")
                    continue
                # Stooq returns newest-first; sort ascending
                df = df.sort_index().reset_index()
                df['symbol'] = symbol
                df.rename(columns={
                    'Date': 'date',
                    'Open': 'open_price',
                    'High': 'high_price',
                    'Low': 'low_price',
                    'Close': 'close_price',
                    'Volume': 'volume'
                }, inplace=True)
                df['adj_close'] = df.get('close_price', df['Close'] if 'Close' in df.columns else None)
                cols = ['symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adj_close']
                df = df[cols]
                all_data.append(df)
                logger.info(f"Stooq: extracted {len(df)} records for {symbol}")
            except Exception as e:
                logger.warning(f"Stooq fallback failed for {symbol}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def extract_yahoo_finance(self, 
                            symbols: List[str], 
                            period: str = "1y", 
                            interval: str = "1d") -> pd.DataFrame:
        """
        Extract stock data from Yahoo Finance using yfinance (robust with batch + retries)
        
        Args:
            symbols: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with stock data
        """
        logger.info(f"Extracting Yahoo Finance data for symbols: {symbols}")
        
        # We'll fetch in small chunks to avoid Yahoo rate-limits (HTTP 429)
        session = self._make_http_session()
        all_data: List[pd.DataFrame] = []

        def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
            return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

        def normalize_single_symbol(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            df = df.copy().reset_index()
            df['symbol'] = symbol
            rename_map = {
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)
            if 'adj_close' not in df.columns and 'close_price' in df.columns:
                df['adj_close'] = df['close_price']
            cols = ['symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adj_close']
            existing = [c for c in cols if c in df.columns]
            out = df[existing]
            return out

        def sleep_with_jitter(base: float):
            time.sleep(base + random.uniform(0, base * 0.3))

        # Try small batches (size 2) first; fallback to per-symbol as needed
        chunk_size = 2
        base_delay = 1.0

        rate_limited = False
        for group in chunk_list(symbols, chunk_size):
            attempts = 2
            delay = base_delay
            batch_df = pd.DataFrame()
            for attempt in range(1, attempts + 1):
                try:
                    batch_df = yf.download(
                        tickers=group,
                        period=period,
                        interval=interval,
                        group_by='ticker',
                        auto_adjust=False,
                        threads=False,
                        progress=False,
                        session=session
                    )
                    break
                except Exception as e:
                    msg = str(e)
                    logger.warning(f"Batch download failed for {group} (attempt {attempt}/{attempts}): {msg}")
                    # If rate-limited, short-circuit to Stooq fallback
                    if '429' in msg or 'Too Many Requests' in msg:
                        rate_limited = True
                        break
                    # Otherwise back off and retry
                    delay *= 1.5
                    sleep_with_jitter(delay)
                    continue
            if rate_limited:
                logger.warning("Yahoo Finance rate-limited (HTTP 429). Falling back to Stooq for this run.")
                stooq_df = self.extract_stooq(symbols, period=period)
                return stooq_df if not stooq_df.empty else pd.DataFrame()

            # If we got data, normalize it by symbol
            if isinstance(batch_df.columns, pd.MultiIndex):
                for sym in group:
                    try:
                        if sym in batch_df.columns.get_level_values(0):
                            sym_df = batch_df[sym]
                            norm = normalize_single_symbol(sym_df, sym)
                            if norm is not None and not norm.empty:
                                all_data.append(norm)
                            else:
                                logger.warning(f"No batch data for symbol: {sym}")
                    except Exception as e:
                        logger.warning(f"Failed to process batch data for {sym}: {e}")
            elif not batch_df.empty and len(group) == 1:
                norm = normalize_single_symbol(batch_df, group[0])
                if norm is not None and not norm.empty:
                    all_data.append(norm)

            # Fallback per-symbol for any missing symbols in group
            got_syms = set([df['symbol'].iloc[0] for df in all_data])
            missing_in_group = [s for s in group if s not in got_syms]
            for sym in missing_in_group:
                try:
                    t = yf.Ticker(sym, session=session)
                    hist = t.history(period=period, interval=interval, timeout=30)
                    if hist.empty:
                        logger.warning(f"No data found for symbol via fallback: {sym}")
                        continue
                    norm = normalize_single_symbol(hist, sym)
                    if norm is not None and not norm.empty:
                        all_data.append(norm)
                        logger.info(f"Fetched {len(norm)} records for {sym} (fallback)")
                    sleep_with_jitter(0.8)
                except Exception as e:
                    msg = str(e)
                    logger.error(f"Error extracting data for {sym} (fallback): {e}")
                    if '429' in msg or 'Too Many Requests' in msg:
                        logger.warning("Yahoo fallback also rate-limited. Switching to Stooq fallback for remaining symbols.")
                        # Fetch remaining symbols via Stooq
                        stooq_df = self.extract_stooq(missing_in_group, period=period)
                        if not stooq_df.empty:
                            all_data.append(stooq_df)
                        # Stop further Yahoo calls
                        break
                    sleep_with_jitter(1.5)
                    continue
        
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total Yahoo Finance records extracted: {len(combined_data)} (symbols: {sorted(set(combined_data['symbol']))})")
            return combined_data
        else:
            logger.warning("No data extracted from Yahoo Finance (after batch and fallback) — trying Stooq fallback")
            stooq_df = self.extract_stooq(symbols, period=period)
            if not stooq_df.empty:
                logger.info(f"Using Stooq fallback: {len(stooq_df)} records")
                return stooq_df
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
