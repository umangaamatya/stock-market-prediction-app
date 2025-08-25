"""
News and Sentiment Data Extractor Module
Extracts news articles and performs sentiment analysis
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
from newsapi import NewsApiClient
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config.config import Config
import re
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsExtractor:
    """Extract news articles and perform sentiment analysis"""
    
    def __init__(self):
        self.news_api_key = Config.NEWS_API_KEY
        self.newsapi = NewsApiClient(api_key=self.news_api_key) if self.news_api_key else None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def extract_news_api(self, 
                        symbols: List[str], 
                        days_back: int = 7,
                        language: str = 'en',
                        sort_by: str = 'publishedAt') -> pd.DataFrame:
        """
        Extract news articles from News API
        
        Args:
            symbols: List of stock symbols to search for
            days_back: Number of days to look back for news
            language: Language of articles
            sort_by: Sort articles by (publishedAt, relevancy, popularity)
            
        Returns:
            DataFrame with news articles
        """
        if not self.newsapi:
            logger.warning("News API key not provided, skipping News API extraction")
            return pd.DataFrame()
        
        logger.info(f"Extracting news for symbols: {symbols}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_articles = []
        
        for symbol in symbols:
            try:
                # Search for news articles about the stock
                # Create search query
                company_name = self._get_company_name(symbol)
                query = f'{symbol} OR "{company_name}" stock'
                
                # Get articles
                articles = self.newsapi.get_everything(
                    q=query,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language=language,
                    sort_by=sort_by,
                    page_size=100  # Maximum allowed by free tier
                )
                
                if articles['status'] == 'ok' and articles['articles']:
                    for article in articles['articles']:
                        # Clean and process article
                        processed_article = self._process_article(article, symbol)
                        if processed_article:
                            all_articles.append(processed_article)
                
                logger.info(f"Extracted {len([a for a in all_articles if a['symbol'] == symbol])} articles for {symbol}")
                
                # Rate limiting - free tier allows 1000 requests per day
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error extracting news for {symbol}: {e}")
                continue
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            logger.info(f"Total articles extracted: {len(df)}")
            return df
        else:
            logger.warning("No news articles extracted")
            return pd.DataFrame()
    
    def extract_financial_news_rss(self, symbols: List[str]) -> pd.DataFrame:
        """
        Extract financial news from RSS feeds
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with news articles
        """
        logger.info("Extracting news from financial RSS feeds")
        
        # Popular financial news RSS feeds
        rss_feeds = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.reuters.com/business/finance',
            'https://rss.cnn.com/rss/money_latest.rss',
            'https://feeds.marketwatch.com/marketwatch/MarketPulse/',
        ]
        
        all_articles = []
        
        for feed_url in rss_feeds:
            try:
                response = requests.get(feed_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                for item in items:
                    # Extract article information
                    title = item.find('title').text if item.find('title') else ''
                    description = item.find('description').text if item.find('description') else ''
                    link = item.find('link').text if item.find('link') else ''
                    pub_date = item.find('pubDate').text if item.find('pubDate') else ''
                    
                    # Check if article mentions any of our symbols
                    content = f"{title} {description}".lower()
                    mentioned_symbols = [symbol for symbol in symbols if symbol.lower() in content]
                    
                    if mentioned_symbols:
                        try:
                            published_at = pd.to_datetime(pub_date)
                        except:
                            published_at = datetime.now()
                        
                        for symbol in mentioned_symbols:
                            article_data = {
                                'symbol': symbol,
                                'title': title,
                                'description': description,
                                'content': description,  # RSS usually only has description
                                'source': feed_url.split('/')[2],  # Extract domain name
                                'author': '',
                                'url': link,
                                'published_at': published_at
                            }
                            
                            all_articles.append(article_data)
                
            except Exception as e:
                logger.error(f"Error extracting from RSS feed {feed_url}: {e}")
                continue
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df = df.drop_duplicates(subset=['title', 'symbol'])  # Remove duplicates
            logger.info(f"Extracted {len(df)} articles from RSS feeds")
            return df
        else:
            return pd.DataFrame()
    
    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of news articles
        
        Args:
            df: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment analysis added
        """
        logger.info("Analyzing sentiment of news articles")
        
        if df.empty:
            return df
        
        sentiment_scores = []
        sentiment_labels = []
        sentiment_confidences = []
        
        for _, row in df.iterrows():
            # Combine title and description for sentiment analysis
            text = f"{row.get('title', '')} {row.get('description', '')}"
            
            if not text.strip():
                sentiment_scores.append(0.0)
                sentiment_labels.append('neutral')
                sentiment_confidences.append(0.0)
                continue
            
            # Use VADER sentiment analyzer (better for financial text)
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # Use TextBlob as secondary analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # Combine both methods for more robust sentiment
            combined_score = (vader_scores['compound'] + textblob_polarity) / 2
            
            # Determine label and confidence
            if combined_score > 0.1:
                label = 'positive'
                confidence = abs(combined_score)
            elif combined_score < -0.1:
                label = 'negative'
                confidence = abs(combined_score)
            else:
                label = 'neutral'
                confidence = 1 - abs(combined_score)
            
            sentiment_scores.append(combined_score)
            sentiment_labels.append(label)
            sentiment_confidences.append(min(confidence, 1.0))
        
        # Add sentiment columns to dataframe
        df = df.copy()
        df['sentiment_score'] = sentiment_scores
        df['sentiment_label'] = sentiment_labels
        df['sentiment_confidence'] = sentiment_confidences
        
        logger.info(f"Sentiment analysis completed for {len(df)} articles")
        return df
    
    def _process_article(self, article: Dict, symbol: str) -> Optional[Dict]:
        """
        Process and clean individual article
        
        Args:
            article: Article dictionary from News API
            symbol: Stock symbol
            
        Returns:
            Processed article dictionary or None
        """
        try:
            # Skip articles without content
            if not article.get('title') and not article.get('description'):
                return None
            
            # Parse published date
            published_at = pd.to_datetime(article.get('publishedAt'))
            
            # Clean text
            title = self._clean_text(article.get('title', ''))
            description = self._clean_text(article.get('description', ''))
            content = self._clean_text(article.get('content', ''))
            
            processed_article = {
                'symbol': symbol,
                'title': title,
                'description': description,
                'content': content,
                'source': article.get('source', {}).get('name', ''),
                'author': article.get('author', ''),
                'url': article.get('url', ''),
                'published_at': published_at
            }
            
            return processed_article
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _get_company_name(self, symbol: str) -> str:
        """
        Get company name for better news search
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company name or empty string
        """
        # Simple mapping for common stocks - in production, this could be more sophisticated
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'META': 'Meta Facebook',
            'NVDA': 'Nvidia',
            'NFLX': 'Netflix',
            'AMD': 'AMD',
            'INTC': 'Intel'
        }
        
        return company_names.get(symbol, symbol)
    
    def get_sentiment_summary(self, df: pd.DataFrame, symbol: str = None) -> Dict:
        """
        Get sentiment summary for articles
        
        Args:
            df: DataFrame with sentiment analysis
            symbol: Optional symbol to filter by
            
        Returns:
            Dictionary with sentiment summary
        """
        if df.empty:
            return {}
        
        # Filter by symbol if provided
        if symbol:
            df = df[df['symbol'] == symbol]
        
        if df.empty:
            return {}
        
        # Calculate summary statistics
        total_articles = len(df)
        avg_sentiment = df['sentiment_score'].mean()
        positive_articles = len(df[df['sentiment_label'] == 'positive'])
        negative_articles = len(df[df['sentiment_label'] == 'negative'])
        neutral_articles = len(df[df['sentiment_label'] == 'neutral'])
        
        # Most recent articles sentiment trend
        df_recent = df.sort_values('published_at', ascending=False).head(10)
        recent_sentiment = df_recent['sentiment_score'].mean() if len(df_recent) > 0 else 0
        
        summary = {
            'total_articles': total_articles,
            'average_sentiment': avg_sentiment,
            'recent_sentiment': recent_sentiment,
            'positive_count': positive_articles,
            'negative_count': negative_articles,
            'neutral_count': neutral_articles,
            'positive_percentage': (positive_articles / total_articles) * 100,
            'negative_percentage': (negative_articles / total_articles) * 100,
            'neutral_percentage': (neutral_articles / total_articles) * 100
        }
        
        return summary
    
    def extract_social_sentiment(self, symbols: List[str]) -> pd.DataFrame:
        """
        Extract social media sentiment (placeholder for Twitter/Reddit APIs)
        This would require Twitter API v2 or Reddit API access
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with social sentiment data
        """
        logger.info("Social sentiment extraction not implemented yet")
        
        # Placeholder implementation
        # In a real implementation, you would:
        # 1. Connect to Twitter API v2
        # 2. Search for tweets about the stock symbols
        # 3. Analyze sentiment of tweets
        # 4. Return aggregated sentiment data
        
        return pd.DataFrame()
    
    def extract_combined_news(self, symbols: List[str], days_back: int = 7) -> pd.DataFrame:
        """
        Extract news from multiple sources and combine
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
            
        Returns:
            Combined DataFrame with all news sources
        """
        logger.info("Extracting news from multiple sources")
        
        all_news = []
        
        # Extract from News API
        news_api_data = self.extract_news_api(symbols, days_back)
        if not news_api_data.empty:
            all_news.append(news_api_data)
        
        # Extract from RSS feeds
        rss_data = self.extract_financial_news_rss(symbols)
        if not rss_data.empty:
            all_news.append(rss_data)
        
        # Combine all sources
        if all_news:
            combined_df = pd.concat(all_news, ignore_index=True)
            
            # Remove duplicates based on title and URL
            combined_df = combined_df.drop_duplicates(subset=['title', 'url'])
            
            # Analyze sentiment
            combined_df = self.analyze_sentiment(combined_df)
            
            logger.info(f"Total combined news articles: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No news data extracted from any source")
            return pd.DataFrame()
