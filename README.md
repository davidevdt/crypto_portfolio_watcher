# 🚀 Crypto Portfolio Tracker

A comprehensive crypto portfolio monitoring application built with **Streamlit** and **SQLite**. This application features a sophisticated multi-exchange data provider system, real-time portfolio tracking, advanced technical analysis, and intelligent investment management capabilities.

![Portfolio Overview](https://img.shields.io/badge/Portfolio-Overview-blue) ![Technical Analysis](https://img.shields.io/badge/Technical-Analysis-green) ![Real--time-Monitoring](https://img.shields.io/badge/Real--time-Monitoring-orange) ![Multi--Exchange](https://img.shields.io/badge/Multi--Exchange-purple)

## 🌟 Key Features

### 💼 Advanced Portfolio Management
- **Multi-Portfolio Support**: Create and manage multiple portfolios with independent tracking
- **Real-time Price Updates**: Live price feeds from multiple exchange APIs (Binance, Bybit, MEXC, Kraken)
- **Transaction History**: Complete buy/sell tracking with profit/loss calculations
- **Asset Analytics**: Performance metrics, allocation charts, and timeline visualizations
- **USDT ⇄ USDC Exchange**: Built-in stablecoin conversion functionality

### 📊 Professional Charting & Technical Analysis
- **TradingView-Style Interface**: Professional-grade charting with multiple timeframes
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- **Signal Analysis**: Automated trend detection and trading signals
- **Historical Data**: Up to 5 years of price history with intelligent caching

### 🔔 Smart Monitoring & Alerts
- **Multi-Signal Detection**: Oversold/Overbought conditions with strength percentages
- **Trend Analysis**: EMA/SMA/MACD trend indicators with directional signals
- **Volatility Classification**: Low/Moderate/High volatility regime detection
- **Desktop Notifications**: Real-time alerts for price movements and signals
- **Watchlist Integration**: Monitor assets before adding to portfolio

### 💰 Advanced Take Profit Strategies
- **Multi-Round Strategy**: Configure 1-10 profit-taking rounds with custom aggressiveness
- **DCA-Out Strategy**: Dollar-cost averaging for gradual position exits
- **Fibonacci Levels**: Automated Fibonacci-based take profit levels
- **Quick Sell Interface**: Instant position liquidation with real-time calculations
- **Profit Timeline**: Visual tracking of realized profits over time

### ⚙️ Comprehensive Configuration
- **Multi-Channel Notifications**: Desktop, WhatsApp frameworks
- **Display Customization**: Currency formats, decimal precision, themes
- **Technical Settings**: Default periods for all indicators and analysis
- **Data Management**: Import/export settings, database cleanup tools
- **Background Service**: Configurable refresh intervals for automated data collection

## 🏗️ Architecture Overview

### **Modular Page-Based Architecture**
The application uses a clean, modular architecture with dedicated page modules:

```
app.py                     # Main Streamlit router with navigation
run.py                     # Enhanced runner with background services
├── page_modules/          # Individual page implementations
│   ├── portfolio_overview.py    # Portfolio management & analytics
│   ├── asset_charts.py          # TradingView-style charting
│   ├── monitoring.py            # Signal monitoring & alerts
│   ├── take_profit.py           # Profit tracking & quick sell
│   ├── take_profit_levels.py    # Advanced TP strategies
│   ├── watchlist.py             # Asset monitoring & tracking
│   └── settings.py              # Configuration management
├── components/            # Shared UI components
├── data_providers/        # Multi-exchange data integration
├── services/              # Business logic & background services
└── database/              # SQLAlchemy models & utilities
```

### **Enhanced Multi-Exchange Data System**
- **Base Provider Architecture**: Abstract base class with consistent interface
- **Exchange Implementations**: Dedicated providers for Binance, Bybit, MEXC, Kraken
- **Intelligent Fallbacks**: Automatic provider switching on failures
- **Rate Limiting**: Built-in protection against API limits
- **Data Persistence**: Complete offline capability with local caching

### **Robust Database Layer**
- **SQLite Backend**: Lightweight, embedded database
- **Enhanced Models**: 10+ tables including historical prices, portfolio values, alerts
- **Automatic Migration**: Schema creation and updates on startup
- **Performance Optimization**: Intelligent caching and query optimization

## 📱 Application Features

### 📊 Portfolio Overview
Manage multiple portfolios with real-time valuations, progress charts, and allocation analysis. Create portfolios, add assets with buy/add modes, and track performance with interactive visualizations.

### 📈 Asset Charts  
TradingView-style charting with technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands). Multiple timeframes, signal analysis, and professional-grade price analysis.

### 🔔 Monitoring
Automated signal detection for oversold/overbought conditions, trend analysis, and volatility classification. Real-time alerts and comprehensive indicator breakdowns.

### 💰 Take Profit
Profit tracking with timeline visualization and quick sell interface. Monitor success rates, profit metrics, and transaction history.

### 🎯 Take Profit Levels
Advanced strategies including Multi-round, DCA-out, and Fibonacci levels. Real-time profit previews and automated level management.

### 👀 Watchlist
Monitor potential investments with full signal analysis. Integrates with charts and enables direct portfolio addition.

### ⚙️ Settings
Comprehensive configuration for notifications, display preferences, technical indicators, and data management.

### 🔗 Data Providers
Monitor and manage the multi-exchange data provider system. View real-time performance metrics, API call statistics, and provider reliability scores. Track which exchanges work best for each asset and troubleshoot data fetching issues.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/davidevdt/crypto_portfolio_watcher.git
cd crypto_portfolio_watcher
```

2. **Set Up Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize Database** (Optional - happens automatically)
   - Database can be created manually on first run
   - To start completely fresh, you can wipe all data from **Settings → Data Management → Delete All Records**

### Running the Application

#### **Recommended Method (With Background Services)**
```bash
python run.py
```
This starts:
- Streamlit web application at `http://localhost:8501`
- Background data collection service
- Automated price updates every 60 seconds
- Browser opens automatically

#### **Basic Method (Web App Only)**
```bash
streamlit run app.py
```
This runs only the web interface without background data collection.

### First Time Setup

1. Open `http://localhost:8501` in your browser
2. Create your first portfolio in "📊 Portfolio Overview"  
3. Add assets with symbol, quantity, and buy price
4. Explore charts, monitoring, and profit tracking features

## ⚙️ Configuration

### Background Services
- Updates prices every 60 seconds (configurable in Settings)
- Collects historical data and calculates portfolio values
- Maintains technical indicator cache

### Database
- **Location**: `db_data/portfolio.db` (SQLite)
- **Reset**: Run `python reset_database.py` for fresh start
- **Backup**: Copy database file for backup

### API Limits
- **Binance**: 1200 req/min, 20 req/sec
- **Bybit**: ~1200 req/min, 10 req/sec  
- **MEXC/Kraken**: Conservative limits applied

## 📁 Project Structure

```
crypto_portfolio_app/
├── 🏠 Main Application
│   ├── app.py                    # Streamlit app with navigation
│   ├── run.py                    # Enhanced runner with background services
│   └── requirements.txt          # Python dependencies
│
├── 📊 User Interface
│   ├── page_modules/             # Individual page implementations
│   │   ├── portfolio_overview.py    # Portfolio management
│   │   ├── asset_charts.py          # Professional charting
│   │   ├── monitoring.py            # Signal monitoring
│   │   ├── take_profit.py           # Profit tracking
│   │   ├── take_profit_levels.py    # Advanced TP strategies
│   │   ├── watchlist.py             # Asset watchlist
│   │   └── settings.py              # Configuration
│   └── components/               # Shared UI components
│       └── shared.py                # Reusable functions
│
├── 🔄 Data Layer
│   ├── data_providers/           # Exchange integrations
│   │   ├── base_provider.py         # Abstract base class
│   │   ├── binance.py               # Binance API integration
│   │   ├── bybit.py                 # Bybit API integration
│   │   ├── mexc.py                  # MEXC API integration
│   │   ├── kraken.py                # Kraken API integration
│   │   └── data_fetcher.py          # Main data orchestrator
│   │
│   └── database/                 # Database layer
│       ├── models.py                # SQLAlchemy models
│       └── utils.py                 # Database utilities
│
├── 🔧 Business Logic
│   └── services/                 # Core services
│       ├── portfolio_manager.py     # Portfolio CRUD operations
│       ├── technical_indicators.py  # Technical analysis
│       ├── background_data_service.py # Background data collection
│       ├── notification_service.py  # Alert management
│       └── shutdown_handler.py      # Graceful shutdown
│
└── 💾 Data Storage
    └── db_data/                  # Database files
        └── portfolio.db             # SQLite database
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit, Plotly charts
- **Database**: SQLite with SQLAlchemy ORM
- **Data**: Pandas, NumPy, aiohttp for APIs
- **Background**: APScheduler, Plyer notifications
- **Features**: Intelligent caching, rate limiting, offline access

## 🐛 Troubleshooting

**Application won't start**: Check Python 3.8+, reinstall dependencies, reset database if needed

**Prices not updating**: Check internet connection, try manual refresh, verify API access

**Database issues**: Run `python reset_database.py` or delete `db_data/portfolio.db*`

**Performance issues**: Reduce refresh interval, clear cache, limit chart indicators

## ⚠️ Work in Progress

This application is still under development and may not work perfectly in all scenarios. Some features might have bugs or limitations. Please be patient as we continue to improve the application.

For issues or feedback, please refer to the project repository or documentation.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Davide Vidotto**

Please respect exchange API terms of service and usage limits when using this application.

---

**🎉 Happy Investing!** 

*Built using Streamlit, designed for crypto investors who demand alternative portfolio management tools.*