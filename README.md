# 🏦 Stock Market Prediction Application

A comprehensive stock market prediction platform that combines technical analysis, sentiment analysis, and machine learning to forecast stock price movements. Built with Python, PostgreSQL, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://www.postgresql.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Dashboard Screenshot](docs/images/dashboard_preview.png)

## ✨ Features

### 📊 **Comprehensive Data Pipeline**
- **Multi-source data extraction**: Yahoo Finance, Alpha Vantage, News API
- **Real-time data updates**: Automated ETL pipeline with PySpark
- **Sentiment analysis**: AI-powered news sentiment analysis
- **Technical indicators**: 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)

### 🤖 **Advanced Machine Learning**
- **Multiple algorithms**: Random Forest, Gradient Boosting, Linear Regression
- **Feature engineering**: Technical + sentiment features
- **Model evaluation**: Comprehensive performance metrics
- **Ensemble predictions**: Multiple model consensus

### 📈 **Interactive Web Interface**
- **Real-time dashboard**: Live stock data and predictions
- **Technical analysis charts**: Interactive charts with indicators
- **Correlation analysis**: Portfolio correlation heatmaps
- **Prediction visualization**: Price targets with confidence intervals

### 🔄 **Production-Ready**
- **Automated updates**: Scheduled data refreshes
- **Error handling**: Robust error handling and logging
- **Scalable architecture**: Docker support and cloud deployment ready
- **Data quality monitoring**: Automated data validation

## 🚀 Quick Start

### Prerequisites
- macOS (tested), Linux and Windows should also work
- Python 3.10+ (recommended)
- Conda (Anaconda/Miniconda) OR Docker Desktop
- PostgreSQL 12+
- 4GB RAM (8GB recommended)

You can run the project either with Docker (easiest) or locally with Conda.

### Option A: Docker (recommended)
```bash
# 1) Clone the repository
git clone https://github.com/yourusername/stock-market-prediction-app.git
cd stock-market-prediction-app

# 2) Copy and edit environment variables
cp .env.template .env
# Edit .env and set at least:
#   POSTGRES_PASSWORD=your_secure_password
# Optional API keys:
#   NEWS_API_KEY=...
#   ALPHA_VANTAGE_API_KEY=...

# 3) Start services (DB, cache, app, and ETL)
# First bring up Postgres and Redis so DB init succeeds
docker compose up -d postgres redis
# Then build and start the app and the continuous ETL service
docker compose up --build -d stock_app etl_service

# 4) Open the app
# App will be on:
#   http://localhost:8501
```
Notes:
- The nginx service is only for production and not required locally.
- If you add API keys later, restart the affected containers: `docker compose restart stock_app etl_service`.

### Option B: Local (Conda on macOS)
```bash
# 1) Clone the repository
git clone https://github.com/yourusername/stock-market-prediction-app.git
cd stock-market-prediction-app

# 2) Create and activate a clean Python 3.10 environment (recommended)
conda create -y -n stockapp python=3.10
conda activate stockapp

# 3) Install TA-Lib via conda-forge (works reliably on macOS/Apple Silicon)
conda install -y -c conda-forge ta-lib

# 4) Install Python dependencies (pinned)
# If the pinned set fails on your platform, you can fall back to requirements_simple.txt
pip install -r requirements.txt || pip install -r requirements_simple.txt

# 5) Install the Stooq data reader (used as fallback when Yahoo rate-limits)
pip install pandas-datareader

# 6) Configure environment variables
cp .env.template .env
# Edit .env to match your local DB and optional API keys
# Key settings:
#   POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT
#   NEWS_API_KEY (optional), ALPHA_VANTAGE_API_KEY (optional)

# 7) Create PostgreSQL database and user (example)
# Adjust commands for your setup. On macOS (Homebrew):
#   brew services start postgresql@14
# Then connect and run:
#   psql -d postgres
# Inside psql:
#   CREATE DATABASE stock_market_db;
#   CREATE USER stock_user WITH PASSWORD 'your_secure_password';
#   GRANT ALL PRIVILEGES ON DATABASE stock_market_db TO stock_user;
#   \q

# 8) Load initial data (full load, ~1y)
python scripts/run_etl.py --mode full --period 1y
# Tip: If Yahoo Finance rate-limits (HTTP 429), the extractor will automatically
#      fall back to Stooq daily data to complete the load.

# 9) Start the Streamlit app
streamlit run web_app/app.py
# App will be on:
#   http://localhost:8501
```

Alternative installer (optional):
- There is a helper script at `scripts/install.py` that creates a venv and installs dependencies.
  On macOS with Apple Silicon, Conda + conda-forge TA-Lib tends to be more reliable than compiling TA-Lib from pip.

### Notes on data providers
- Yahoo Finance (via yfinance) is the primary free source used. It can temporarily rate-limit (HTTP 429). In that case, the pipeline falls back to Stooq (via pandas-datareader) for daily data.
- Alpha Vantage is optional. If you set `ALPHA_VANTAGE_API_KEY`, the pipeline can blend that data as well.

Visit http://localhost:8501 🎉

## 📚 Documentation

- **[Complete User Manual](docs/USER_MANUAL.md)** - Step-by-step guide for non-technical users
- **[API Documentation](docs/API.md)** - Developer reference
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and components

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  ETL Pipeline   │    │   PostgreSQL    │
│                 │    │                 │    │                 │
│ • Yahoo Finance │───▶│ • Extraction    │───▶│ • Stock Data    │
│ • News API      │    │ • Transform     │    │ • News Data     │
│ • Alpha Vantage │    │ • Load          │    │ • Indicators    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐              │
│  Web Interface  │    │  ML Pipeline    │◀─────────────┘
│                 │    │                 │
│ • Streamlit     │◀───│ • Training      │
│ • Interactive   │    │ • Prediction    │
│ • Visualizations│    │ • Evaluation    │
└─────────────────┘    └─────────────────┘
```

## 📊 Supported Stocks

**Default Portfolio:**
- 🍎 AAPL (Apple)
- 🔍 GOOGL (Alphabet)
- 💻 MSFT (Microsoft)
- 🚗 TSLA (Tesla)
- 📦 AMZN (Amazon)
- 📘 META (Meta)
- 🎮 NVDA (Nvidia)
- 🎬 NFLX (Netflix)
- 💾 AMD (Advanced Micro Devices)
- 💻 INTC (Intel)

*Easily configurable to analyze any stock symbol*

## 🤖 Machine Learning Models

### Prediction Models
| Model | Purpose | Accuracy Range | Speed |
|-------|---------|----------------|--------|
| **Random Forest** | Price & Direction | 65-75% | Medium |
| **Gradient Boosting** | Price Prediction | 60-70% | Slow |
| **Linear Regression** | Trend Analysis | 55-65% | Fast |

### Features Used
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Price Patterns**: Momentum, Volatility, Support/Resistance
- **Volume Analysis**: Volume trends and anomalies
- **Sentiment Scores**: News sentiment and social media buzz
- **Market Context**: Day of week, month, seasonal patterns

## 📈 Sample Predictions

```
Symbol: AAPL
Current Price: $175.43
Predicted Price: $178.20 (+1.58%)
Direction: UP ↗️
Confidence: 73.2%
Model: Random Forest
```

## 🛠️ Configuration

### Environment Variables
```env
# Database
POSTGRES_USER=stock_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=stock_market_db

# APIs (Optional)
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_API_KEY=your_av_key

# Settings
DEFAULT_STOCKS=AAPL,GOOGL,MSFT,TSLA,AMZN
UPDATE_INTERVAL_HOURS=6
PREDICTION_DAYS=1
```

### Customization
- **Stock Selection**: Edit `config/config.py`
- **Model Parameters**: Modify `models/ml_prediction_model.py`
- **Technical Indicators**: Update `etl/transformers/technical_indicators.py`
- **UI Themes**: Customize `web_app/app.py`

## 🔧 Advanced Usage

### Real-time Updates
```bash
# Start real-time updater
python scripts/real_time_updater.py --mode continuous

# Manual updates
python scripts/real_time_updater.py --mode prices
python scripts/real_time_updater.py --mode news
python scripts/real_time_updater.py --mode predictions
```

### Model Training
```python
from models.ml_prediction_model import StockPredictionModel

# Initialize model
model = StockPredictionModel()

# Prepare training data
data, metadata = model.prepare_training_data(
    symbols=['AAPL', 'GOOGL'],
    days_back=365,
    include_sentiment=True
)

# Train models
results = model.train_models(data)

# Save trained models
model.save_models('my_model.pkl')
```

### API Usage
```python
from etl.extractors.stock_extractor import StockDataExtractor

# Extract stock data
extractor = StockDataExtractor()
data = extractor.extract_yahoo_finance(['AAPL'], period='1y')
```

## 📊 Performance Metrics

### Model Performance
- **Prediction Accuracy**: 55-75% (varies by stock and market conditions)
- **Processing Speed**: ~1000 records/second
- **Memory Usage**: ~500MB for full dataset
- **Storage**: ~100MB/year per stock

### System Performance
- **Data Loading**: 10-15 minutes for full refresh
- **Model Training**: 5-10 minutes per stock
- **Prediction Generation**: <1 second per stock
- **Web Interface**: <2 second page loads

## 🐛 Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U stock_user -d stock_market_db
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Install TA-Lib (may require system packages)
pip install TA-Lib
```

**API Limits Exceeded**
- News API: 1000 requests/month (free tier)
- Alpha Vantage: 500 requests/day (free tier)
- Yahoo Finance: No explicit limits

For more troubleshooting, see the [User Manual](docs/USER_MANUAL.md#troubleshooting).

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** your changes: `git commit -am 'Add feature'`
4. **Push** to the branch: `git push origin feature-name`
5. **Submit** a pull request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/stock-market-prediction-app.git

# Create development branch
git checkout -b feature-branch

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Code formatting
black .
flake8 .
```

## 📋 Roadmap

### v1.1 - Enhanced ML
- [ ] Deep Learning models (LSTM, Transformer)
- [ ] Cryptocurrency support
- [ ] Options pricing models
- [ ] Portfolio optimization

### v1.2 - Enterprise Features
- [ ] Multi-user support
- [ ] Role-based access control
- [ ] Advanced backtesting
- [ ] API endpoints

### v1.3 - Cloud & Mobile
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Mobile app (React Native)
- [ ] Cloud provider integrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. **Do not use this as the sole basis for investment decisions.** Always consult with qualified financial advisors and do your own research before making investment decisions. The authors are not responsible for any financial losses incurred from using this software.

## 🙏 Acknowledgments

- **Yahoo Finance** for free stock data API
- **News API** for news data
- **Alpha Vantage** for additional financial data
- **Streamlit** for the amazing web framework
- **scikit-learn** for machine learning tools
- **TA-Lib** for technical analysis indicators

## 📞 Support

- **Documentation**: [User Manual](docs/USER_MANUAL.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/stock-market-prediction-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/stock-market-prediction-app/discussions)
- **Email**: support@stockpredictionapp.com

---

**Made with ❤️ for the financial analysis community**

*Star ⭐ this repository if you find it useful!*
