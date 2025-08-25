"""
Stock Market Prediction Web Application
Built with Streamlit for interactive data visualization and predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from config.database import execute_raw_query
from models.ml_prediction_model import StockPredictionModel
from etl.etl_pipeline import ETLPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Market Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockMarketApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.prediction_model = StockPredictionModel()
        self.etl_pipeline = ETLPipeline()
        self.available_symbols = Config.DEFAULT_STOCKS
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}
    
    def run(self):
        """Main application entry point"""
        st.title("🏦 Stock Market Prediction Platform")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.get('current_page', 'Dashboard')
        
        if page == 'Dashboard':
            self.render_dashboard()
        elif page == 'Data Pipeline':
            self.render_data_pipeline()
        elif page == 'Predictions':
            self.render_predictions()
        elif page == 'Analysis':
            self.render_analysis()
        elif page == 'Settings':
            self.render_settings()
    
    def render_sidebar(self):
        """Render sidebar navigation and controls"""
        st.sidebar.title("Navigation")
        
        pages = ['Dashboard', 'Data Pipeline', 'Predictions', 'Analysis', 'Settings']
        current_page = st.sidebar.selectbox("Select Page", pages)
        st.session_state.current_page = current_page
        
        st.sidebar.markdown("---")
        
        # Stock selection
        st.sidebar.subheader("Stock Selection")
        selected_stocks = st.sidebar.multiselect(
            "Choose stocks to analyze:",
            self.available_symbols,
            default=self.available_symbols[:5]
        )
        st.session_state.selected_stocks = selected_stocks
        
        # Time range selection
        st.sidebar.subheader("Time Range")
        time_range = st.sidebar.selectbox(
            "Select time period:",
            ['1M', '3M', '6M', '1Y', '2Y', '5Y'],
            index=3
        )
        st.session_state.time_range = time_range
        
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.subheader("System Status")
        self.render_system_status()
    
    def render_system_status(self):
        """Render system status in sidebar"""
        # Database connection status
        try:
            test_query = "SELECT COUNT(*) FROM stock_data LIMIT 1"
            result = execute_raw_query(test_query)
            if not result.empty:
                st.sidebar.success("🟢 Database Connected")
                db_status = True
            else:
                st.sidebar.error("🔴 Database Empty")
                db_status = False
        except:
            st.sidebar.error("🔴 Database Error")
            db_status = False
        
        # Data status
        if db_status:
            try:
                data_summary = self.get_data_summary()
                if data_summary:
                    st.sidebar.info(f"📊 {data_summary.get('total_records', 0)} records")
                    latest_date = data_summary.get('latest_date')
                    if latest_date:
                        st.sidebar.info(f"📅 Latest: {latest_date.strftime('%Y-%m-%d')}")
            except:
                st.sidebar.warning("⚠️ Data Summary Unavailable")
        
        # Model status
        model_status = st.session_state.get('models_trained', False)
        if model_status:
            st.sidebar.success("🤖 Models Ready")
        else:
            st.sidebar.warning("🤖 Models Not Trained")
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.header("📊 Dashboard Overview")
        
        # Key metrics
        self.render_key_metrics()
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_stock_price_chart()
        
        with col2:
            self.render_sentiment_chart()
        
        # Recent predictions
        st.subheader("🔮 Recent Predictions")
        self.render_recent_predictions()
        
        # Market summary
        st.subheader("📈 Market Summary")
        self.render_market_summary()
    
    def render_key_metrics(self):
        """Render key performance metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            # Get data for selected stocks
            selected_stocks = st.session_state.get('selected_stocks', [])
            if not selected_stocks:
                st.warning("Please select stocks from the sidebar")
                return
            
            # Stock data metrics
            stock_data = self.get_stock_data(selected_stocks, days_back=30)
            
            if not stock_data.empty:
                # Total return
                total_return = stock_data.groupby('symbol')['close_price'].apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100 if len(x) > 1 else 0
                ).mean()
                
                # Volatility
                volatility = stock_data.groupby('symbol')['close_price'].apply(
                    lambda x: x.pct_change().std() * np.sqrt(252) * 100
                ).mean()
                
                # Volume
                avg_volume = stock_data['volume'].mean()
                
                # News sentiment
                sentiment_data = self.get_sentiment_data(selected_stocks, days_back=7)
                avg_sentiment = sentiment_data['avg_sentiment'].mean() if not sentiment_data.empty else 0
                
                with col1:
                    st.metric(
                        "Avg 30D Return",
                        f"{total_return:.2f}%",
                        delta=f"{total_return:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Avg Volatility",
                        f"{volatility:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "Avg Volume",
                        f"{avg_volume/1e6:.1f}M"
                    )
                
                with col4:
                    sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                    st.metric(
                        "News Sentiment",
                        sentiment_label,
                        delta=f"{avg_sentiment:.3f}"
                    )
            
        except Exception as e:
            logger.error(f"Error rendering key metrics: {e}")
            st.error("Error loading key metrics")
    
    def render_stock_price_chart(self):
        """Render interactive stock price chart"""
        st.subheader("📈 Stock Price Trends")
        
        try:
            selected_stocks = st.session_state.get('selected_stocks', [])
            if not selected_stocks:
                st.info("Select stocks from sidebar to view charts")
                return
            
            # Get stock data
            days_back = self.get_days_from_range(st.session_state.get('time_range', '1Y'))
            stock_data = self.get_stock_data(selected_stocks, days_back=days_back)
            
            if stock_data.empty:
                st.warning("No stock data available")
                return
            
            # Create interactive chart
            fig = go.Figure()
            
            for symbol in selected_stocks:
                symbol_data = stock_data[stock_data['symbol'] == symbol]
                if not symbol_data.empty:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['date'],
                        y=symbol_data['close_price'],
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Stock Price Trends",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering stock price chart: {e}")
            st.error("Error loading stock price chart")
    
    def render_sentiment_chart(self):
        """Render sentiment analysis chart"""
        st.subheader("😊 News Sentiment")
        
        try:
            selected_stocks = st.session_state.get('selected_stocks', [])
            if not selected_stocks:
                st.info("Select stocks from sidebar to view sentiment")
                return
            
            # Get sentiment data
            sentiment_data = self.get_sentiment_data(selected_stocks, days_back=30)
            
            if sentiment_data.empty:
                st.warning("No sentiment data available")
                return
            
            # Create sentiment chart
            fig = px.bar(
                sentiment_data.groupby('symbol').agg({
                    'positive_news': 'sum',
                    'negative_news': 'sum',
                    'news_count': 'sum'
                }).reset_index(),
                x='symbol',
                y=['positive_news', 'negative_news'],
                title='News Sentiment Distribution',
                labels={'value': 'Number of Articles', 'symbol': 'Stock Symbol'}
            )
            
            fig.update_layout(
                xaxis_title="Stock Symbol",
                yaxis_title="Number of Articles",
                legend_title="Sentiment"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering sentiment chart: {e}")
            st.error("Error loading sentiment chart")
    
    def render_recent_predictions(self):
        """Render recent predictions table"""
        try:
            predictions = st.session_state.get('predictions', {})
            
            if not predictions:
                st.info("No predictions available. Train models in the Predictions tab.")
                return
            
            # Convert predictions to DataFrame
            pred_data = []
            for symbol, pred in predictions.items():
                pred_data.append({
                    'Symbol': symbol,
                    'Current Price': f"${pred.get('current_price', 0):.2f}",
                    'Predicted Price': f"${pred.get('predicted_price', 0):.2f}",
                    'Expected Return': f"{pred.get('predicted_return', 0)*100:.2f}%",
                    'Direction': pred.get('predicted_direction', 'unknown'),
                    'Confidence': f"{pred.get('confidence', 0)*100:.1f}%"
                })
            
            if pred_data:
                df = pd.DataFrame(pred_data)
                st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering predictions: {e}")
            st.error("Error loading predictions")
    
    def render_market_summary(self):
        """Render market summary table"""
        try:
            selected_stocks = st.session_state.get('selected_stocks', [])
            if not selected_stocks:
                return
            
            # Get latest stock data
            stock_data = self.get_latest_stock_data(selected_stocks)
            
            if not stock_data.empty:
                # Calculate additional metrics
                stock_data['Change %'] = stock_data.groupby('symbol')['close_price'].pct_change() * 100
                
                # Format for display
                display_data = stock_data[['symbol', 'close_price', 'volume']].copy()
                display_data.columns = ['Symbol', 'Price ($)', 'Volume']
                display_data['Price ($)'] = display_data['Price ($)'].apply(lambda x: f"${x:.2f}")
                display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x/1e6:.1f}M")
                
                st.dataframe(display_data, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering market summary: {e}")
            st.error("Error loading market summary")
    
    def render_data_pipeline(self):
        """Render data pipeline management page"""
        st.header("🔄 Data Pipeline")
        
        # Pipeline status
        st.subheader("Pipeline Status")
        data_summary = self.get_data_summary()
        
        if data_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", data_summary.get('total_records', 0))
            with col2:
                st.metric("Symbols", data_summary.get('symbols_count', 0))
            with col3:
                latest_date = data_summary.get('latest_date')
                if latest_date:
                    st.metric("Latest Date", latest_date.strftime('%Y-%m-%d'))
        
        st.markdown("---")
        
        # Pipeline controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Run ETL Pipeline")
            
            pipeline_mode = st.selectbox(
                "Pipeline Mode",
                ['Incremental Update', 'Full Refresh']
            )
            
            include_news = st.checkbox("Include News Data", value=True)
            
            if st.button("🚀 Run Pipeline", type="primary"):
                self.run_etl_pipeline(pipeline_mode, include_news)
        
        with col2:
            st.subheader("Data Sources")
            
            # API status
            st.write("**API Configuration:**")
            st.write(f"📊 Yahoo Finance: ✅ Available")
            st.write(f"📰 News API: {'✅' if Config.NEWS_API_KEY else '❌'} {'Configured' if Config.NEWS_API_KEY else 'Not Configured'}")
            st.write(f"💹 Alpha Vantage: {'✅' if Config.ALPHA_VANTAGE_API_KEY else '❌'} {'Configured' if Config.ALPHA_VANTAGE_API_KEY else 'Not Configured'}")
        
        # Data quality
        st.markdown("---")
        st.subheader("📊 Data Quality")
        self.render_data_quality_metrics()
    
    def render_predictions(self):
        """Render predictions page"""
        st.header("🔮 Stock Predictions")
        
        # Model training section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Training")
            
            training_period = st.selectbox(
                "Training Period",
                ['6 Months', '1 Year', '2 Years'],
                index=1
            )
            
            include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
            
            if st.button("🤖 Train Models", type="primary"):
                self.train_prediction_models(training_period, include_sentiment)
        
        with col2:
            st.subheader("Generate Predictions")
            
            selected_stocks = st.session_state.get('selected_stocks', [])
            prediction_stocks = st.multiselect(
                "Stocks to Predict",
                selected_stocks,
                default=selected_stocks[:3] if len(selected_stocks) >= 3 else selected_stocks
            )
            
            model_type = st.selectbox(
                "Model Type",
                ['random_forest', 'linear', 'gradient_boosting']
            )
            
            if st.button("📈 Generate Predictions"):
                if st.session_state.get('models_trained', False):
                    self.generate_predictions(prediction_stocks, model_type)
                else:
                    st.error("Please train models first!")
        
        # Display predictions
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        predictions = st.session_state.get('predictions', {})
        if predictions:
            self.render_prediction_results(predictions)
        else:
            st.info("No predictions available. Train models and generate predictions above.")
    
    def render_analysis(self):
        """Render analysis page"""
        st.header("📊 Advanced Analysis")
        
        # Technical analysis
        st.subheader("Technical Analysis")
        self.render_technical_analysis()
        
        # Correlation analysis
        st.markdown("---")
        st.subheader("📈 Correlation Analysis")
        self.render_correlation_analysis()
        
        # Feature importance
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        self.render_feature_importance()
    
    def render_settings(self):
        """Render settings page"""
        st.header("⚙️ Settings")
        
        # Configuration settings
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Database Settings:**")
            st.code(f"Host: {Config.POSTGRES_HOST}")
            st.code(f"Database: {Config.POSTGRES_DB}")
            st.code(f"User: {Config.POSTGRES_USER}")
            
            st.write("**Data Settings:**")
            st.code(f"Default Stocks: {', '.join(Config.DEFAULT_STOCKS)}")
            st.code(f"Update Interval: {Config.UPDATE_INTERVAL_HOURS}h")
        
        with col2:
            st.write("**API Settings:**")
            st.code(f"News API: {'Configured' if Config.NEWS_API_KEY else 'Not Configured'}")
            st.code(f"Alpha Vantage: {'Configured' if Config.ALPHA_VANTAGE_API_KEY else 'Not Configured'}")
            
            st.write("**Model Settings:**")
            st.code(f"Prediction Days: {Config.PREDICTION_DAYS}")
        
        # System information
        st.markdown("---")
        st.subheader("System Information")
        
        # Show system status
        self.render_detailed_system_status()
    
    # Helper methods
    
    def get_stock_data(self, symbols: List[str], days_back: int = 365) -> pd.DataFrame:
        """Get stock data for specified symbols"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            symbol_list = "','".join(symbols)
            query = f"""
            SELECT symbol, date, open_price, high_price, low_price, close_price, volume
            FROM stock_data
            WHERE symbol IN ('{symbol_list}')
            AND date >= '{start_date.strftime('%Y-%m-%d')}'
            ORDER BY symbol, date
            """
            
            return execute_raw_query(query)
            
        except Exception as e:
            logger.error(f"Error getting stock data: {e}")
            return pd.DataFrame()
    
    def get_sentiment_data(self, symbols: List[str], days_back: int = 30) -> pd.DataFrame:
        """Get sentiment data for specified symbols"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            symbol_list = "','".join(symbols)
            query = f"""
            SELECT symbol, DATE(published_at) as date, 
                   AVG(sentiment_score) as avg_sentiment,
                   COUNT(*) as news_count,
                   SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_news,
                   SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_news
            FROM news_data
            WHERE symbol IN ('{symbol_list}')
            AND DATE(published_at) >= '{start_date.strftime('%Y-%m-%d')}'
            GROUP BY symbol, DATE(published_at)
            ORDER BY symbol, date
            """
            
            return execute_raw_query(query)
            
        except Exception as e:
            logger.error(f"Error getting sentiment data: {e}")
            return pd.DataFrame()
    
    def get_latest_stock_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get latest stock data for symbols"""
        try:
            symbol_list = "','".join(symbols)
            query = f"""
            SELECT DISTINCT ON (symbol) symbol, date, close_price, volume
            FROM stock_data
            WHERE symbol IN ('{symbol_list}')
            ORDER BY symbol, date DESC
            """
            
            return execute_raw_query(query)
            
        except Exception as e:
            logger.error(f"Error getting latest stock data: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self) -> Dict:
        """Get data summary from database"""
        try:
            # Stock data summary
            stock_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as symbols_count,
                MAX(date) as latest_date
            FROM stock_data
            """
            
            result = execute_raw_query(stock_query)
            if not result.empty:
                return result.iloc[0].to_dict()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    def get_days_from_range(self, time_range: str) -> int:
        """Convert time range string to days"""
        range_map = {
            '1M': 30,
            '3M': 90,
            '6M': 180,
            '1Y': 365,
            '2Y': 730,
            '5Y': 1825
        }
        return range_map.get(time_range, 365)
    
    def run_etl_pipeline(self, mode: str, include_news: bool):
        """Run ETL pipeline"""
        with st.spinner("Running ETL pipeline..."):
            try:
                if mode == 'Full Refresh':
                    result = self.etl_pipeline.run_full_pipeline(
                        period="2y",
                        include_news=include_news
                    )
                else:
                    result = self.etl_pipeline.run_incremental_update()
                
                if result['success']:
                    st.success("✅ Pipeline completed successfully!")
                    st.json(result['data_summary'])
                else:
                    st.error("❌ Pipeline failed!")
                    if result.get('errors'):
                        for error in result['errors']:
                            st.error(error)
                            
            except Exception as e:
                st.error(f"Pipeline error: {e}")
    
    def train_prediction_models(self, period: str, include_sentiment: bool):
        """Train prediction models"""
        with st.spinner("Training models..."):
            try:
                days_back = 365 if period == '1 Year' else 730 if period == '2 Years' else 180
                
                # Prepare training data
                training_data, metadata = self.prediction_model.prepare_training_data(
                    symbols=st.session_state.get('selected_stocks'),
                    days_back=days_back,
                    include_sentiment=include_sentiment
                )
                
                if training_data.empty:
                    st.error("No training data available!")
                    return
                
                # Train models
                results = self.prediction_model.train_models(training_data)
                
                if results:
                    st.session_state.models_trained = True
                    st.success("✅ Models trained successfully!")
                    
                    # Show training results
                    for symbol, metrics in results.items():
                        with st.expander(f"📊 {symbol} Model Performance"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Regression Metrics:**")
                                for model, perf in metrics.get('regression', {}).items():
                                    st.write(f"• {model}: R² = {perf.get('r2', 0):.3f}, MAE = {perf.get('mae', 0):.3f}")
                            with col2:
                                st.write("**Classification Metrics:**")
                                for model, perf in metrics.get('classification', {}).items():
                                    st.write(f"• {model}: Accuracy = {perf.get('accuracy', 0):.3f}")
                else:
                    st.error("❌ Model training failed!")
                    
            except Exception as e:
                st.error(f"Training error: {e}")
    
    def generate_predictions(self, symbols: List[str], model_type: str):
        """Generate predictions for selected stocks"""
        with st.spinner("Generating predictions..."):
            try:
                predictions = {}
                
                for symbol in symbols:
                    # Get latest data for prediction
                    latest_data = self.get_prediction_features(symbol)
                    
                    if not latest_data.empty:
                        prediction = self.prediction_model.predict(
                            symbol, latest_data, model_type
                        )
                        
                        if prediction:
                            predictions[symbol] = prediction
                
                if predictions:
                    st.session_state.predictions = predictions
                    st.success(f"✅ Generated predictions for {len(predictions)} stocks!")
                else:
                    st.error("❌ No predictions generated!")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    def get_prediction_features(self, symbol: str) -> pd.DataFrame:
        """Get latest features for prediction"""
        try:
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
                # Add basic features that the model expects
                data['price_change'] = 0  # Will be calculated properly in a real scenario
                data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
                data['month'] = pd.to_datetime(data['date']).dt.month
                
            return data
            
        except Exception as e:
            logger.error(f"Error getting prediction features: {e}")
            return pd.DataFrame()
    
    def render_prediction_results(self, predictions: Dict):
        """Render prediction results in detail"""
        for symbol, pred in predictions.items():
            with st.expander(f"📈 {symbol} - Prediction Details", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_price = pred.get('current_price', 0)
                    predicted_price = pred.get('predicted_price', 0)
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}"
                    )
                    st.metric(
                        "Predicted Price",
                        f"${predicted_price:.2f}",
                        delta=f"${predicted_price - current_price:.2f}"
                    )
                
                with col2:
                    predicted_return = pred.get('predicted_return', 0)
                    st.metric(
                        "Expected Return",
                        f"{predicted_return*100:.2f}%",
                        delta=f"{predicted_return*100:.2f}%"
                    )
                    st.metric(
                        "Direction",
                        pred.get('predicted_direction', 'unknown').upper()
                    )
                
                with col3:
                    confidence = pred.get('confidence', 0)
                    st.metric(
                        "Confidence",
                        f"{confidence*100:.1f}%"
                    )
                    st.metric(
                        "Model Used",
                        pred.get('model_type', 'unknown')
                    )
    
    def render_technical_analysis(self):
        """Render technical analysis charts"""
        try:
            selected_stocks = st.session_state.get('selected_stocks', [])
            if not selected_stocks:
                st.info("Select stocks to view technical analysis")
                return
            
            symbol = st.selectbox("Select stock for technical analysis:", selected_stocks)
            
            # Get data with technical indicators
            query = f"""
            SELECT s.date, s.close_price, s.volume, t.sma_20, t.sma_50, 
                   t.rsi, t.macd, t.macd_signal, t.bb_upper, t.bb_lower
            FROM stock_data s
            LEFT JOIN technical_indicators t ON s.symbol = t.symbol AND s.date = t.date
            WHERE s.symbol = '{symbol}'
            AND s.date >= NOW() - INTERVAL '6 months'
            ORDER BY s.date
            """
            
            data = execute_raw_query(query)
            
            if not data.empty:
                # Create subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                    vertical_spacing=0.1
                )
                
                # Price and moving averages
                fig.add_trace(go.Scatter(x=data['date'], y=data['close_price'], name='Close Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=data['date'], y=data['sma_20'], name='SMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=data['date'], y=data['sma_50'], name='SMA 50'), row=1, col=1)
                
                # RSI
                if 'rsi' in data.columns:
                    fig.add_trace(go.Scatter(x=data['date'], y=data['rsi'], name='RSI'), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                if 'macd' in data.columns:
                    fig.add_trace(go.Scatter(x=data['date'], y=data['macd'], name='MACD'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=data['date'], y=data['macd_signal'], name='Signal'), row=3, col=1)
                
                fig.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering technical analysis: {e}")
    
    def render_correlation_analysis(self):
        """Render correlation analysis"""
        try:
            selected_stocks = st.session_state.get('selected_stocks', [])
            if len(selected_stocks) < 2:
                st.info("Select at least 2 stocks to view correlations")
                return
            
            # Get stock data
            stock_data = self.get_stock_data(selected_stocks, days_back=180)
            
            if not stock_data.empty:
                # Pivot data for correlation
                price_data = stock_data.pivot(index='date', columns='symbol', values='close_price')
                correlation_matrix = price_data.corr()
                
                # Create heatmap
                fig = px.imshow(
                    correlation_matrix,
                    title="Stock Price Correlations",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering correlation analysis: {e}")
    
    def render_feature_importance(self):
        """Render feature importance if models are trained"""
        if not st.session_state.get('models_trained', False):
            st.info("Train models first to see feature importance")
            return
        
        try:
            selected_stocks = st.session_state.get('selected_stocks', [])
            if not selected_stocks:
                return
            
            symbol = st.selectbox("Select stock for feature importance:", selected_stocks, key="feature_importance_symbol")
            
            # Get feature importance from trained model
            importance = self.prediction_model.get_feature_importance(symbol)
            
            if importance:
                st.write(f"Feature importance for {symbol} (Random Forest model)")
                st.write(f"Number of features: {importance['feature_count']}")
                
                # Note: In a real implementation, you'd want to map feature indices to names
                importance_values = importance['importance_values'][:10]  # Top 10
                feature_names = [f'Feature {i+1}' for i in range(len(importance_values))]
                
                fig = px.bar(
                    x=importance_values,
                    y=feature_names,
                    orientation='h',
                    title=f'Top 10 Features - {symbol}'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance available for selected stock")
                
        except Exception as e:
            st.error(f"Error rendering feature importance: {e}")
    
    def render_data_quality_metrics(self):
        """Render data quality metrics"""
        try:
            # Data completeness
            quality_query = """
            SELECT 
                symbol,
                COUNT(*) as total_records,
                COUNT(CASE WHEN close_price IS NOT NULL THEN 1 END) as price_records,
                COUNT(CASE WHEN volume IS NOT NULL AND volume > 0 THEN 1 END) as volume_records,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM stock_data
            WHERE date >= NOW() - INTERVAL '30 days'
            GROUP BY symbol
            ORDER BY symbol
            """
            
            quality_data = execute_raw_query(quality_query)
            
            if not quality_data.empty:
                # Calculate completeness percentages
                quality_data['price_completeness'] = (quality_data['price_records'] / quality_data['total_records'] * 100).round(1)
                quality_data['volume_completeness'] = (quality_data['volume_records'] / quality_data['total_records'] * 100).round(1)
                
                display_data = quality_data[['symbol', 'total_records', 'price_completeness', 'volume_completeness']].copy()
                display_data.columns = ['Symbol', 'Records', 'Price Complete (%)', 'Volume Complete (%)']
                
                st.dataframe(display_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering data quality metrics: {e}")
    
    def render_detailed_system_status(self):
        """Render detailed system status"""
        try:
            # Database statistics
            stats_query = """
            SELECT 
                'Stock Data' as table_name,
                COUNT(*) as record_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM stock_data
            UNION ALL
            SELECT 
                'News Data' as table_name,
                COUNT(*) as record_count,
                MIN(published_at) as earliest_date,
                MAX(published_at) as latest_date
            FROM news_data
            UNION ALL
            SELECT 
                'Technical Indicators' as table_name,
                COUNT(*) as record_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM technical_indicators
            """
            
            stats_data = execute_raw_query(stats_query)
            
            if not stats_data.empty:
                st.write("**Database Statistics:**")
                st.dataframe(stats_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering system status: {e}")

def main():
    """Main application entry point"""
    app = StockMarketApp()
    app.run()

if __name__ == "__main__":
    main()
