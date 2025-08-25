# 📈 Stock Market Prediction Application - User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Configuration](#configuration)
5. [Getting Started](#getting-started)
6. [Using the Application](#using-the-application)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)
9. [FAQ](#faq)
10. [Support](#support)

---

## Introduction

Welcome to the **Stock Market Prediction Application**! This comprehensive platform combines:
- 📊 **Real-time stock data** from Yahoo Finance and Alpha Vantage
- 📰 **News sentiment analysis** using AI-powered natural language processing
- 🤖 **Machine learning predictions** with multiple algorithms
- 📈 **Interactive visualizations** and technical analysis
- 🔄 **Automated ETL pipeline** for data processing

**What this application does:**
- Predicts stock price movements using technical indicators and sentiment analysis
- Provides interactive charts and visualizations
- Analyzes news sentiment impact on stock prices
- Offers multiple prediction models (Random Forest, Linear Regression, etc.)
- Maintains historical data for trend analysis

**Who this is for:**
- Individual investors and traders
- Financial analysts and researchers
- Students learning about financial markets
- Anyone interested in stock market predictions

---

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Stable internet connection for data retrieval

### Required Software
1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
2. **PostgreSQL 12+** - [Download here](https://www.postgresql.org/download/)
3. **Git** (optional but recommended) - [Download here](https://git-scm.com/downloads/)

### Optional API Keys (Free)
- **News API** - For news sentiment analysis
- **Alpha Vantage** - For additional stock data sources

---

## Installation Guide

### Step 1: Download the Application

**Option A: Download ZIP File**
1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to your desired location

**Option B: Clone with Git**
```bash
git clone https://github.com/yourusername/stock-market-prediction-app.git
cd stock-market-prediction-app
```

### Step 2: Run the Automated Installer

**For Windows:**
1. Open Command Prompt as Administrator
2. Navigate to the application folder:
   ```cmd
   cd path\to\stock-market-prediction-app
   ```
3. Run the installer:
   ```cmd
   python scripts/install.py
   ```

**For macOS/Linux:**
1. Open Terminal
2. Navigate to the application folder:
   ```bash
   cd /path/to/stock-market-prediction-app
   ```
3. Make the installer executable and run it:
   ```bash
   chmod +x scripts/install.py
   python3 scripts/install.py
   ```

The installer will:
- ✅ Check system requirements
- 🐍 Create a Python virtual environment
- 📦 Install all required dependencies
- 📁 Create necessary directories
- ⚙️ Set up configuration files

### Step 3: Manual Installation (if automated installer fails)

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   
   **Windows:**
   ```cmd
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

### Step 1: Database Setup

**Install PostgreSQL:**

**Windows:**
1. Download PostgreSQL from [official website](https://www.postgresql.org/download/windows/)
2. Run the installer with default settings
3. Remember the password you set for the 'postgres' user

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Ubuntu/Linux:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Create Database:**
1. Open PostgreSQL command line:
   ```bash
   sudo -u postgres psql
   ```

2. Create database and user:
   ```sql
   CREATE DATABASE stock_market_db;
   CREATE USER stock_user WITH PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE stock_market_db TO stock_user;
   \q
   ```

### Step 2: Configure Environment Variables

1. **Copy the template:**
   ```bash
   cp .env.template .env
   ```

2. **Edit the .env file** with your favorite text editor:

   **Required Settings:**
   ```env
   # Database Configuration
   POSTGRES_USER=stock_user
   POSTGRES_PASSWORD=your_secure_password
   POSTGRES_DB=stock_market_db
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

   **Optional API Keys:**
   ```env
   # News API (free tier: 1000 requests/month)
   NEWS_API_KEY=your_news_api_key_here
   
   # Alpha Vantage (optional, for additional data)
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
   ```

### Step 3: Get API Keys (Optional but Recommended)

**News API Key (Free):**
1. Go to [NewsAPI.org](https://newsapi.org/)
2. Click "Get API Key"
3. Sign up with your email
4. Copy your API key
5. Add it to your `.env` file

**Alpha Vantage Key (Optional):**
1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Enter your email and click "GET FREE API KEY"
3. Copy your API key
4. Add it to your `.env` file

**Note:** The application will work without these API keys using only Yahoo Finance data, but you'll get more features with them.

---

## Getting Started

### Step 1: Validate Your Setup

Before running the application, validate your configuration:

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Validate configuration
python -m etl.etl_pipeline --validate
```

You should see:
- ✅ Configuration Valid
- ✅ Database Connected
- ⚠️ Warnings for missing API keys (optional)

### Step 2: Load Initial Data

Run the ETL pipeline to download and process stock data:

```bash
# Load initial data (this may take 10-15 minutes)
python -m etl.etl_pipeline --mode full --period 1y
```

This will:
- 📊 Download 1 year of stock data for default stocks (AAPL, GOOGL, MSFT, etc.)
- 📰 Fetch recent news articles (if News API key is configured)
- 🔄 Calculate technical indicators (RSI, MACD, etc.)
- 💾 Store everything in your PostgreSQL database

### Step 3: Start the Web Application

```bash
streamlit run web_app/app.py
```

**Success!** Your browser should open automatically to `http://localhost:8501`

If it doesn't open automatically, manually navigate to: **http://localhost:8501**

---

## Using the Application

### Dashboard Overview

When you first open the application, you'll see the **Dashboard** with:

1. **📊 Key Metrics**
   - Average 30-day returns
   - Volatility measures
   - Trading volume
   - News sentiment

2. **📈 Stock Price Charts**
   - Interactive price trends
   - Multiple stocks comparison
   - Zoom and pan capabilities

3. **😊 Sentiment Analysis**
   - News sentiment distribution
   - Positive vs negative news counts

4. **🔮 Recent Predictions**
   - Latest model predictions
   - Price targets and confidence levels

### Navigation Menu

Use the **sidebar** to navigate between sections:

#### 🏠 Dashboard
- Overview of all stocks
- Key performance metrics
- Recent predictions summary

#### 🔄 Data Pipeline
- **Pipeline Status**: See how much data you have
- **Run ETL Pipeline**: Update data or perform full refresh
- **Data Quality**: Monitor data completeness

**To update your data:**
1. Go to "Data Pipeline" page
2. Select "Incremental Update" for daily updates
3. Select "Full Refresh" to reload everything
4. Check "Include News Data" if you have News API key
5. Click "🚀 Run Pipeline"

#### 🔮 Predictions
This is where the magic happens! 

**Train Models:**
1. Select training period (6 months to 2 years)
2. Choose whether to include sentiment analysis
3. Click "🤖 Train Models"
4. Wait for training to complete (5-10 minutes)

**Generate Predictions:**
1. Select stocks to predict
2. Choose model type:
   - **Random Forest**: Most accurate, slower
   - **Linear**: Fast, simple patterns
   - **Gradient Boosting**: Good balance
3. Click "📈 Generate Predictions"

**Understanding Predictions:**
- **Predicted Price**: Expected price tomorrow
- **Expected Return**: Percentage gain/loss
- **Direction**: UP (bullish), DOWN (bearish), STABLE
- **Confidence**: How sure the model is (0-100%)

#### 📊 Analysis
Advanced analysis tools:

**Technical Analysis:**
- Price charts with moving averages
- RSI (Relative Strength Index)
- MACD indicators
- Support/resistance levels

**Correlation Analysis:**
- See how stocks move together
- Correlation heatmap
- Portfolio diversification insights

**Feature Importance:**
- Which factors matter most for predictions
- Technical vs sentiment factors

#### ⚙️ Settings
- View configuration settings
- Check system status
- Database statistics

### Stock Selection

**In the sidebar**, you can:
1. **Choose stocks** to analyze from the dropdown
2. **Set time range** (1 month to 5 years)
3. **View system status** (database, models, etc.)

**Default stocks included:**
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- TSLA (Tesla)
- AMZN (Amazon)
- META (Facebook)
- NVDA (Nvidia)
- NFLX (Netflix)
- AMD (AMD)
- INTC (Intel)

---

## Troubleshooting

### Common Issues

#### ❌ "Database connection failed"

**Cause**: PostgreSQL is not running or configuration is incorrect

**Solutions:**
1. **Start PostgreSQL:**
   ```bash
   # Windows (in Services or)
   pg_ctl -D "C:\Program Files\PostgreSQL\13\data" start
   
   # macOS
   brew services start postgresql
   
   # Linux
   sudo systemctl start postgresql
   ```

2. **Check your .env file** - ensure database credentials are correct

3. **Test connection manually:**
   ```bash
   psql -h localhost -U stock_user -d stock_market_db
   ```

#### ❌ "No module named 'xxx'"

**Cause**: Dependencies not installed or wrong Python environment

**Solution:**
1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### ❌ "Port 8501 is already in use"

**Cause**: Another Streamlit app is running

**Solutions:**
1. **Stop other Streamlit apps** (Ctrl+C in their terminals)
2. **Use different port:**
   ```bash
   streamlit run web_app/app.py --server.port 8502
   ```

#### ⚠️ "No news data available"

**Cause**: News API key not configured or API limit reached

**Solutions:**
1. **Add News API key** to .env file
2. **Check API usage** at newsapi.org
3. **The app works without news** - just with reduced sentiment features

#### 📊 "No stock data available"

**Cause**: ETL pipeline hasn't run or failed

**Solutions:**
1. **Run ETL pipeline:**
   ```bash
   python -m etl.etl_pipeline --mode full
   ```

2. **Check internet connection**

3. **Verify Yahoo Finance is accessible**

### Performance Issues

#### Slow Loading
1. **Reduce number of selected stocks**
2. **Shorten time range**
3. **Close other applications**
4. **Increase system RAM if possible**

#### Model Training Takes Too Long
1. **Reduce training period**
2. **Select fewer stocks**
3. **Use Linear model instead of Random Forest**

### Getting Help

1. **Check the logs:**
   ```bash
   tail -f etl_pipeline.log
   ```

2. **Validate configuration:**
   ```bash
   python -m etl.etl_pipeline --validate
   ```

3. **Restart everything:**
   ```bash
   # Stop the app (Ctrl+C)
   # Restart PostgreSQL
   # Restart the app
   streamlit run web_app/app.py
   ```

---

## Advanced Features

### Custom Stock Lists

To add your own stocks:

1. **Edit config/config.py**:
   ```python
   DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'YOUR_STOCK_HERE']
   ```

2. **Restart the application**

### Automated Updates

Set up automated daily updates:

**Linux/macOS (using cron):**
1. Edit crontab:
   ```bash
   crontab -e
   ```

2. Add daily update at 6 PM:
   ```bash
   0 18 * * * cd /path/to/app && source venv/bin/activate && python -m etl.etl_pipeline --mode incremental
   ```

**Windows (using Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger to Daily
4. Set action to run: `python -m etl.etl_pipeline --mode incremental`

### Database Management

**View data directly:**
```sql
-- Connect to database
psql -h localhost -U stock_user -d stock_market_db

-- Check data
SELECT symbol, COUNT(*) FROM stock_data GROUP BY symbol;
SELECT symbol, AVG(sentiment_score) FROM news_data GROUP BY symbol;

-- Clean old data
DELETE FROM news_data WHERE published_at < NOW() - INTERVAL '30 days';
```

### Model Customization

**Train models with different parameters:**

1. **Edit models/ml_prediction_model.py**
2. **Modify model configurations:**
   ```python
   'random_forest': {
       'regressor': RandomForestRegressor(n_estimators=200, max_depth=10)
   }
   ```

### API Limits and Costs

**News API (Free Tier):**
- 1,000 requests per month
- Up to 100 articles per request
- Rate limit: 1 request per second

**Alpha Vantage (Free Tier):**
- 5 API requests per minute
- 500 requests per day

**Yahoo Finance:**
- Free, no API key required
- Rate limits apply but are generous

---

## FAQ

### General Questions

**Q: Is this application free?**
A: Yes! The application is completely free. API keys are also free (with usage limits).

**Q: Do I need to pay for stock data?**
A: No, Yahoo Finance provides free stock data, and the application uses this by default.

**Q: How accurate are the predictions?**
A: Accuracy varies by stock and market conditions. Typical accuracy ranges from 55-75%. Always do your own research before making investment decisions.

**Q: Can I use this for real trading?**
A: This is an educational and research tool. While the predictions can be informative, always consult with financial advisors for investment decisions.

### Technical Questions

**Q: Can I run this on a cloud server?**
A: Yes! The application can run on AWS, Google Cloud, or any VPS. You'll need to configure the firewall to allow port 8501.

**Q: How much data does it store?**
A: Typical usage: ~100MB per year of stock data, ~50MB for news data (with API key).

**Q: Can I export the data?**
A: Yes, you can query the PostgreSQL database directly or export via the web interface.

**Q: What if I want to analyze crypto currencies?**
A: The application can be modified to work with crypto data. You'd need to update the data extractors to use crypto APIs.

### Data Questions

**Q: How often should I update the data?**
A: Daily incremental updates are recommended. Full refreshes can be done weekly or monthly.

**Q: What happens if my internet goes down during data loading?**
A: The ETL pipeline is resilient and will resume where it left off on the next run.

**Q: Can I analyze international stocks?**
A: Yes! Yahoo Finance supports international stocks. Use the appropriate ticker symbols (e.g., "TSLA.L" for London Exchange).

---

## Support

### Getting Help

1. **Documentation**: Check this manual first
2. **GitHub Issues**: Report bugs or request features
3. **Community**: Join our Discord/Reddit community
4. **Email**: support@stockpredictionapp.com

### Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: Use GitHub Issues
2. **Suggest Features**: Create feature requests
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Help improve this manual
5. **Testing**: Test on different systems

### Version Updates

**Check for updates:**
```bash
git pull origin main
pip install -r requirements.txt
```

**Update notifications**: Enable GitHub notifications to get alerts about new releases.

### System Information

**For support requests, please include:**
- Operating System and version
- Python version (`python --version`)
- Application version
- Error messages (copy the full error)
- Configuration (without passwords!)

---

## Appendix

### File Structure
```
stock-market-prediction-app/
├── config/             # Configuration files
├── data/              # Data storage
├── docs/              # Documentation
├── etl/               # ETL pipeline
├── models/            # ML models
├── scripts/           # Utility scripts
├── web_app/           # Web application
├── .env              # Environment variables
├── requirements.txt   # Python dependencies
└── README.md         # Project overview
```

### Command Reference

```bash
# Installation
python scripts/install.py

# Configuration validation
python -m etl.etl_pipeline --validate

# Data loading
python -m etl.etl_pipeline --mode full --period 2y
python -m etl.etl_pipeline --mode incremental

# Start application
streamlit run web_app/app.py

# Different port
streamlit run web_app/app.py --server.port 8502

# Development mode
streamlit run web_app/app.py --server.runOnSave true
```

### Default Configuration Values

```env
UPDATE_INTERVAL_HOURS=6
PREDICTION_DAYS=1
DEFAULT_STOCKS=AAPL,GOOGL,MSFT,TSLA,AMZN,META,NVDA,NFLX,AMD,INTC
SPARK_DRIVER_MEMORY=4g
SPARK_EXECUTOR_MEMORY=2g
```

---

**🎉 Congratulations!** You're now ready to start predicting stock prices like a pro!

Remember: This tool is for educational and research purposes. Always do your own research and consult with financial advisors before making investment decisions.

**Happy Trading! 📈✨**
