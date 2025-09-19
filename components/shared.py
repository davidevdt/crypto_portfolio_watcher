"""
Shared components and utilities used across all pages
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from contextlib import contextmanager

# Import project modules
from database.models import create_database
from services.portfolio_manager import PortfolioManager
from data_providers.data_fetcher import CryptoPriceFetcher

logger = logging.getLogger(__name__)


# Custom CSS for better styling
def load_custom_css():
    """
    Load custom CSS for the application optimized for Streamlit's dark theme.

    Provides styling for profit/loss indicators, alerts, signals, and data freshness.
    """
    css = """
    <style>
        /* Custom styling that works well with Streamlit's native dark theme */
        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #ff6b6b;
        }
        .profit-positive {
            color: #28a745;
            font-weight: bold;
        }
        .profit-negative {
            color: #dc3545;
            font-weight: bold;
        }
        .alert-high {
            background-color: rgba(220, 53, 69, 0.1);
            border: 1px solid rgba(220, 53, 69, 0.3);
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .alert-medium {
            background-color: rgba(255, 193, 7, 0.1);
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .alert-low {
            background-color: rgba(23, 162, 184, 0.1);
            border: 1px solid rgba(23, 162, 184, 0.3);
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .signal-buy {
            color: #28a745;
            font-weight: bold;
        }
        .signal-sell {
            color: #dc3545;
            font-weight: bold;
        }
        .signal-neutral {
            color: #868e96;
            font-weight: bold;
        }
        .volatility-low {
            color: #28a745;
        }
        .volatility-medium {
            color: #ffc107;
        }
        .volatility-high {
            color: #dc3545;
        }
        
        /* Live data pulse animation */
        .live-pulse {
            animation: pulse 2s ease-in-out infinite alternate;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        /* Data freshness indicators */
        .data-fresh {
            color: #28a745;
            font-weight: bold;
        }
        
        .data-stale {
            color: #ffc107;
            font-weight: bold;
        }
        
        .data-very-stale {
            color: #dc3545;
            font-weight: bold;
        }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


# Initialize database and services
@st.cache_resource
def init_services():
    """
    Initialize database and core services.

    Returns:
        tuple: (portfolio_manager, price_fetcher) instances
    """
    create_database()
    portfolio_manager = PortfolioManager()
    price_fetcher = CryptoPriceFetcher()

    return portfolio_manager, price_fetcher


# Initialize session state
def init_session_state():
    """
    Initialize Streamlit session state variables with database-first approach.

    Sets up portfolio manager, price fetcher, and loads user settings from database.
    """
    if "portfolio_manager" not in st.session_state:
        st.session_state.portfolio_manager, st.session_state.price_fetcher = (
            init_services()
        )

    if "current_prices" not in st.session_state:
        st.session_state.current_prices = {}
        try:
            load_cached_prices_to_session()
            logger.info("Loaded cached prices on startup")
        except Exception as e:
            logger.error(f"Error loading cached prices on startup: {e}")

    if "historical_data" not in st.session_state:
        st.session_state.historical_data = {}

    if "last_price_update" not in st.session_state:
        st.session_state.last_price_update = datetime.now()

    if "selected_portfolio" not in st.session_state:
        st.session_state.selected_portfolio = "all"

    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    if "settings" not in st.session_state:
        default_settings = {
            "auto_refresh": True,
            "refresh_interval": 300,  # 5 minutes default
            "desktop_notifications": False,
            "email_notifications": False,
            "whatsapp_notifications": False,
            "email_address": "",
            "whatsapp_number": "",
            "smtp_server": "",
            "smtp_port": 587,
            "smtp_username": "",
            "smtp_password": "",
            "price_alert_threshold": 5.0,
            "signal_change_alerts": True,
            "decimal_places": 4,
            "chart_theme": "plotly",
            # Monitoring parameters
            "volatility_short_periods": 10,
            "volatility_long_periods": 30,
            "volatility_low_threshold": 0.7,
            "volatility_high_threshold": 1.3,
            "ema_trend_periods": 20,
            "ema_medium_periods": 50,
            "sma_trend_periods": 20,
            "sma_medium_periods": 50,
            "sma_long_periods": 200,
            "macd_trend_periods": 9,
            "rsi_periods": 14,
            "bb_period": 20,
            "bb_std": 2.0,
            "macd_fast_periods": 12,
            "macd_slow_periods": 26,
            # Asset Charts settings
            "num_smas": 2,
            "num_emas": 2,
            "show_debug_info": False,
            # Chart-specific indicator settings (will be saved per session)
            "chart_sma_periods": "20,50",  # Comma-separated SMA periods
            "chart_ema_periods": "12,26",  # Comma-separated EMA periods
            "chart_show_rsi": True,
            "chart_show_macd": True,
            "chart_show_bollinger": True,
            "chart_show_stochastic": False,
            "chart_show_williams": False,
            "chart_default_timeframe": "1 day",
            "chart_default_period": "180",  # Default days to show
            # Individual chart indicator parameters
            "chart_rsi_period": 14,
            "chart_macd_fast": 12,
            "chart_macd_slow": 26,
            "chart_macd_signal": 9,
            "chart_stoch_k": 14,
            "chart_stoch_d": 3,
            "chart_williams_period": 14,
            "chart_bb_period": 20,
            "chart_bb_std": 2.0,
            "chart_volume_ma_period": 20,
            "chart_num_smas": 2,
            "chart_num_emas": 2,
            # Individual SMA and EMA period settings
            "chart_sma_period_0": 20,
            "chart_sma_period_1": 50,
            "chart_sma_period_2": 100,
            "chart_sma_period_3": 150,
            "chart_sma_period_4": 200,
            "chart_ema_period_0": 12,
            "chart_ema_period_1": 26,
            "chart_ema_period_2": 50,
            "chart_ema_period_3": 100,
            "chart_ema_period_4": 200,
            # Selected symbol memory
            "last_selected_symbol": "BTC",
            "last_selected_portfolio": "all",
        }

        loaded_settings = default_settings.copy()
        try:
            portfolio_manager = st.session_state.portfolio_manager
            for key in default_settings.keys():
                saved_value = portfolio_manager.get_setting(key)
                if saved_value is not None:
                    if key in [
                        "auto_refresh",
                        "desktop_notifications",
                        "email_notifications",
                        "whatsapp_notifications",
                        "signal_change_alerts",
                        "show_debug_info",
                        "chart_show_rsi",
                        "chart_show_macd",
                        "chart_show_bollinger",
                        "chart_show_stochastic",
                        "chart_show_williams",
                    ]:
                        loaded_settings[key] = saved_value.lower() == "true"
                    elif key in [
                        "refresh_interval",
                        "smtp_port",
                        "decimal_places",
                        "volatility_short_periods",
                        "volatility_long_periods",
                        "ema_trend_periods",
                        "ema_medium_periods",
                        "sma_trend_periods",
                        "sma_medium_periods",
                        "sma_long_periods",
                        "macd_trend_periods",
                        "rsi_periods",
                        "bb_period",
                        "macd_fast_periods",
                        "macd_slow_periods",
                        "num_smas",
                        "num_emas",
                        "chart_default_period",
                        "chart_rsi_period",
                        "chart_macd_fast",
                        "chart_macd_slow",
                        "chart_macd_signal",
                        "chart_stoch_k",
                        "chart_stoch_d",
                        "chart_williams_period",
                        "chart_bb_period",
                        "chart_volume_ma_period",
                        "chart_num_smas",
                        "chart_num_emas",
                        "chart_sma_period_0",
                        "chart_sma_period_1",
                        "chart_sma_period_2",
                        "chart_sma_period_3",
                        "chart_sma_period_4",
                        "chart_ema_period_0",
                        "chart_ema_period_1",
                        "chart_ema_period_2",
                        "chart_ema_period_3",
                        "chart_ema_period_4",
                    ]:
                        loaded_settings[key] = int(saved_value)
                    elif key in [
                        "price_alert_threshold",
                        "volatility_low_threshold",
                        "volatility_high_threshold",
                        "bb_std",
                        "chart_bb_std",
                    ]:
                        loaded_settings[key] = float(saved_value)
                    else:
                        loaded_settings[key] = saved_value
        except Exception as e:
            logger.warning(f"Could not load settings from database: {e}")

        st.session_state.settings = loaded_settings

        ensure_settings_persistence()


def save_all_settings():
    """
    Save all current settings to the database.

    Iterates through session state settings and persists them to the database.
    """
    try:
        portfolio_manager = st.session_state.portfolio_manager
        for key, value in st.session_state.settings.items():
            portfolio_manager.set_setting(key, str(value))
        logger.info("All settings saved to database")
    except Exception as e:
        logger.error(f"Error saving settings to database: {e}")


def save_chart_preferences(
    symbol: str = None, timeframe: str = None, period: int = None, **chart_settings
):
    """
    Save chart-specific preferences to settings.

    Args:
        symbol: Selected cryptocurrency symbol
        timeframe: Chart timeframe (1h, 4h, 1d, etc.)
        period: Number of periods to display
        **chart_settings: Additional chart configuration options
    """
    try:
        if symbol:
            st.session_state.settings["last_selected_symbol"] = symbol
        if timeframe:
            st.session_state.settings["chart_default_timeframe"] = timeframe
        if period:
            st.session_state.settings["chart_default_period"] = str(period)

        for key, value in chart_settings.items():
            if key.startswith("chart_"):
                st.session_state.settings[key] = value

        save_all_settings()

    except Exception as e:
        logger.error(f"Error saving chart preferences: {e}")


def load_chart_preferences() -> Dict[str, any]:
    """
    Load chart-specific preferences from settings.

    Returns:
        dict: Chart preferences including symbol, timeframe, periods, and indicators
    """
    try:
        return {
            "symbol": st.session_state.settings.get("last_selected_symbol", "BTC"),
            "timeframe": st.session_state.settings.get(
                "chart_default_timeframe", "1 day"
            ),
            "period": int(st.session_state.settings.get("chart_default_period", 180)),
            "sma_periods": st.session_state.settings.get(
                "chart_sma_periods", "20,50"
            ).split(","),
            "ema_periods": st.session_state.settings.get(
                "chart_ema_periods", "12,26"
            ).split(","),
            "show_rsi": st.session_state.settings.get("chart_show_rsi", True),
            "show_macd": st.session_state.settings.get("chart_show_macd", True),
            "show_bollinger": st.session_state.settings.get(
                "chart_show_bollinger", True
            ),
            "show_stochastic": st.session_state.settings.get(
                "chart_show_stochastic", False
            ),
            "show_williams": st.session_state.settings.get(
                "chart_show_williams", False
            ),
        }
    except Exception as e:
        logger.error(f"Error loading chart preferences: {e}")
        return {}


# Database-first price fetching - NO MORE DIRECT API CALLS
def get_prices_from_database(symbols: List[str]) -> Dict[str, float]:
    """
    Get current prices from database cache only (database-first approach).

    Args:
        symbols: List of cryptocurrency symbols to fetch prices for

    Returns:
        dict: Symbol to price mapping from cached database data
    """
    try:
        from database.models import get_session, CachedPrice

        session = get_session()
        prices = {}

        cached_prices = (
            session.query(CachedPrice).filter(CachedPrice.symbol.in_(symbols)).all()
        )

        for cached_price in cached_prices:
            prices[cached_price.symbol] = float(cached_price.price)

        session.close()

        stablecoins = ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]
        for symbol in symbols:
            if symbol in stablecoins and symbol not in prices:
                prices[symbol] = 1.0

        missing_symbols = [s for s in symbols if s not in prices]
        if missing_symbols:
            if st.session_state.settings.get("show_debug_info", False):
                st.info(
                    f"🔧 Debug: Symbols not in cache: {missing_symbols}. Background service will collect these."
                )

        return prices

    except Exception as e:
        st.error(f"Error loading prices from database: {e}")
        return {}


# DEPRECATED: No longer use direct API fetching - use database-first approach
def fetch_prices_sync(symbols: List[str]) -> Dict[str, float]:
    """DEPRECATED: Use get_prices_from_database instead for database-first approach."""
    st.warning("⚠️ Direct API fetching is deprecated. Using database cache instead.")
    return get_prices_from_database(symbols)


def display_last_update_time(
    container=None, help_text: str = "Time when data was last updated"
):
    """Display last update time across all pages with consistent formatting."""
    try:
        if container is None:
            container = st

        # Get the most recent data update time from database
        update_time = None

        try:
            from database.models import get_session, CachedPrice

            session = get_session()

            # Get all symbols we're tracking
            symbols = get_all_tracking_symbols()
            if symbols:
                latest_update = (
                    session.query(CachedPrice)
                    .filter(CachedPrice.symbol.in_(symbols))
                    .order_by(CachedPrice.last_updated.desc())
                    .first()
                )

                if latest_update:
                    update_time = latest_update.last_updated

            session.close()

        except Exception:
            # Fallback to session state
            if (
                hasattr(st.session_state, "last_price_update")
                and st.session_state.last_price_update
            ):
                update_time = st.session_state.last_price_update

        if update_time:
            # Calculate time difference
            time_diff = datetime.now() - update_time
            seconds_ago = int(time_diff.total_seconds())
            minutes_ago = seconds_ago // 60
            hours_ago = minutes_ago // 60

            # Format display based on age
            if seconds_ago < 30:
                container.markdown("🟢 **Last Updated:** Just now", help=help_text)
            elif seconds_ago < 60:
                container.markdown(
                    f"🟢 **Last Updated:** {seconds_ago}s ago", help=help_text
                )
            elif minutes_ago < 5:
                container.markdown(
                    f"🟢 **Last Updated:** {minutes_ago}m ago", help=help_text
                )
            elif minutes_ago < 15:
                container.markdown(
                    f"🟡 **Last Updated:** {minutes_ago}m ago", help=help_text
                )
            elif hours_ago < 1:
                container.markdown(
                    f"🟠 **Last Updated:** {minutes_ago}m ago", help=help_text
                )
            elif hours_ago < 24:
                container.markdown(
                    f"🔴 **Last Updated:** {hours_ago}h ago", help=help_text
                )
            else:
                days_ago = hours_ago // 24
                container.markdown(
                    f"🔴 **Last Updated:** {days_ago}d ago", help=help_text
                )

            # Also show exact timestamp on hover/click
            formatted_time = update_time.strftime("%Y-%m-%d %H:%M:%S")
            container.caption(f"Exact time: {formatted_time}")

        else:
            container.markdown("⚪ **Last Updated:** Never", help="No data available")

    except Exception as e:
        if container:
            container.markdown(
                "❌ **Last Updated:** Error", help=f"Error getting update time: {e}"
            )


def universal_timeframe_selector(
    page_key: str, default_index: int = 0, help_text: str = None
) -> Tuple[str, str, int]:
    """Universal timeframe selector for consistent UX across pages."""
    timeframes = ["1 hour", "4 hours", "1 day", "1 week", "1 month"]

    # Map timeframes to intervals and data periods
    timeframe_config = {
        "1 hour": {
            "interval": "1h",
            "days": 7,
            "periods": 168,
        },  # 7 days of hourly data
        "4 hours": {"interval": "4h", "days": 30, "periods": 180},  # 30 days of 4h data
        "1 day": {
            "interval": "1d",
            "days": 1825,
            "periods": 1825,
        },  # 5 years of daily data
        "1 week": {
            "interval": "1w",
            "days": 1825,
            "periods": 260,
        },  # 5 years of weekly data
        "1 month": {
            "interval": "1M",
            "days": 1825,
            "periods": 60,
        },  # 5 years of monthly data
    }

    # Create unique session key for this page
    session_key = f"{page_key}_selected_timeframe"
    previous_key = f"{page_key}_previous_timeframe"

    selected_timeframe = st.selectbox(
        "⏰ Timeframe",
        timeframes,
        index=default_index,
        key=session_key,
        help=help_text or "Select timeframe for analysis and calculations",
    )

    # Check if timeframe changed to trigger recalculation
    if (
        previous_key in st.session_state
        and st.session_state[previous_key] != selected_timeframe
    ):

        # Clear relevant cache when timeframe changes
        cache_keys_to_clear = []
        for key in st.session_state.keys():
            if f"{page_key}_" in key and (
                "_cache" in key
                or "_data" in key
                or "historical_data" in key
                or "indicators" in key
            ):
                cache_keys_to_clear.append(key)

        for key in cache_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Force immediate update
        if st.session_state.settings.get("show_debug_info", False):
            st.info(
                f"🔧 Debug: Timeframe changed from {st.session_state[previous_key]} to {selected_timeframe} - cleared cache"
            )

    # Store current selection
    st.session_state[previous_key] = selected_timeframe

    # Return timeframe, interval, and days for data fetching
    config = timeframe_config[selected_timeframe]
    return selected_timeframe, config["interval"], config["days"]


def get_historical_data_from_database(
    symbol: str, days: int, interval: str = "1d"
) -> List[Dict]:
    """Get historical data from database cache only (database-first approach)."""
    try:
        from database.models import get_session, HistoricalPrice

        session = get_session()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Query database for historical prices
        historical_prices = (
            session.query(HistoricalPrice)
            .filter(
                HistoricalPrice.symbol == symbol,
                HistoricalPrice.interval == interval,
                HistoricalPrice.date >= start_date,
                HistoricalPrice.date <= end_date,
            )
            .order_by(HistoricalPrice.date.asc())
            .all()
        )

        session.close()

        if historical_prices:
            # Convert to expected format
            data = []
            for price in historical_prices:
                data.append(
                    {
                        "timestamp": int(price.date.timestamp() * 1000),
                        "date": price.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "open": float(price.price),
                        "high": float(price.price),
                        "low": float(price.price),
                        "close": float(price.price),
                        "volume": float(price.volume) if price.volume else 0,
                    }
                )
            return data
        else:
            if st.session_state.settings.get("show_debug_info", False):
                st.info(
                    f"🔧 Debug: No historical data in cache for {symbol}. Background service will collect data automatically."
                )
            return []

    except Exception as e:
        st.error(f"Error loading historical data for {symbol} from database: {e}")
        return []


# DEPRECATED: No longer use direct API fetching - use database-first approach
def fetch_historical_data_sync(symbol: str, days: int, interval: str) -> List[Dict]:
    """DEPRECATED: Use get_historical_data_from_database instead for database-first approach."""
    st.warning("⚠️ Direct API fetching is deprecated. Using database cache instead.")
    return get_historical_data_from_database(symbol, days, interval)


# Notification functions
def send_desktop_notification(title: str, message: str):
    """Send desktop notification."""
    try:
        from plyer import notification

        notification.notify(
            title=title,
            message=message,
            app_name="Crypto Portfolio Tracker",
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Error sending desktop notification: {e}")


def send_whatsapp_notification(message: str, phone_number: str):
    """Send WhatsApp notification using API service."""
    try:
        import streamlit as st
        import requests

        # Handle case when running outside Streamlit context (tests/background threads)
        try:
            settings = st.session_state.settings
            whatsapp_api_url = settings.get("whatsapp_api_url", "")
            whatsapp_api_key = settings.get("whatsapp_api_key", "")
        except AttributeError:
            # Fallback: get settings from database for background threads
            try:
                from database.models import get_session, UserSettings

                session = get_session()

                api_url_setting = (
                    session.query(UserSettings)
                    .filter_by(setting_key="whatsapp_api_url")
                    .first()
                )
                api_key_setting = (
                    session.query(UserSettings)
                    .filter_by(setting_key="whatsapp_api_key")
                    .first()
                )

                whatsapp_api_url = (
                    api_url_setting.setting_value if api_url_setting else ""
                )
                whatsapp_api_key = (
                    api_key_setting.setting_value if api_key_setting else ""
                )

                session.close()
                settings = {}

            except Exception:
                # Final fallback
                whatsapp_api_url = ""
                whatsapp_api_key = ""
                settings = {}

        # Log the message for testing/debugging
        logger.info(f"Sending WhatsApp notification to {phone_number}: {message}")

        # Store in test messages for UI display (if in Streamlit context)
        try:
            if "whatsapp_test_messages" not in st.session_state:
                st.session_state.whatsapp_test_messages = []

            st.session_state.whatsapp_test_messages.append(
                {
                    "timestamp": datetime.now(),
                    "phone": phone_number,
                    "message": message,
                    "status": "pending",
                }
            )
        except AttributeError:
            # Not in Streamlit context, skip UI storage
            pass

        # If API credentials are configured, send actual WhatsApp message
        if whatsapp_api_url and whatsapp_api_key:
            try:
                # Generic API format that works with most WhatsApp services
                payload = {"to": phone_number, "message": message, "type": "text"}

                headers = {
                    "Authorization": f"Bearer {whatsapp_api_key}",
                    "Content-Type": "application/json",
                }

                response = requests.post(
                    whatsapp_api_url, headers=headers, json=payload, timeout=10
                )

                if response.status_code == 200:
                    logger.info(f"WhatsApp message sent successfully to {phone_number}")
                    # Update status in test messages
                    try:
                        if st.session_state.whatsapp_test_messages:
                            st.session_state.whatsapp_test_messages[-1][
                                "status"
                            ] = "sent"
                    except AttributeError:
                        pass
                else:
                    logger.error(
                        f"WhatsApp API error: {response.status_code} - {response.text}"
                    )
                    # Update status in test messages
                    try:
                        if st.session_state.whatsapp_test_messages:
                            st.session_state.whatsapp_test_messages[-1][
                                "status"
                            ] = f"failed ({response.status_code})"
                    except AttributeError:
                        pass

            except requests.exceptions.RequestException as e:
                logger.error(f"WhatsApp API request failed: {e}")
                # Update status in test messages
                try:
                    if st.session_state.whatsapp_test_messages:
                        st.session_state.whatsapp_test_messages[-1][
                            "status"
                        ] = f"error ({str(e)[:50]})"
                except AttributeError:
                    pass
        else:
            # No API configured - just log for testing
            logger.info("WhatsApp API not configured - message logged only")
            # Update status in test messages
            try:
                if st.session_state.whatsapp_test_messages:
                    st.session_state.whatsapp_test_messages[-1][
                        "status"
                    ] = "logged (no API)"
            except AttributeError:
                pass

        # Keep only last 10 test messages
        try:
            if len(st.session_state.whatsapp_test_messages) > 10:
                st.session_state.whatsapp_test_messages = (
                    st.session_state.whatsapp_test_messages[-10:]
                )
        except AttributeError:
            pass

    except Exception as e:
        logger.error(f"Error in WhatsApp notification: {e}")
        # Update status in test messages if available
        try:
            if (
                "whatsapp_test_messages" in st.session_state
                and st.session_state.whatsapp_test_messages
            ):
                st.session_state.whatsapp_test_messages[-1][
                    "status"
                ] = f"error ({str(e)[:50]})"
        except AttributeError:
            pass


# Portfolio selector component
def portfolio_selector(use_sidebar=False):
    """Portfolio selector component to be placed at the top of pages or in sidebar."""
    # Cache portfolios to avoid repeated DB queries
    cache_key = "portfolio_selector_cache"
    if cache_key not in st.session_state or not st.session_state.get(cache_key):
        portfolios = st.session_state.portfolio_manager.get_all_portfolios()
        st.session_state[cache_key] = portfolios
    else:
        portfolios = st.session_state[cache_key]

    if portfolios:
        # Add "All Portfolios" option
        portfolio_options = {"All Portfolios": "all"}
        portfolio_options.update({f"{p.name}": p.id for p in portfolios})

        # Calculate index more efficiently
        current_selected = st.session_state.get("selected_portfolio", "all")
        current_index = 0  # Default to "All Portfolios"

        for i, (_, portfolio_id) in enumerate(portfolio_options.items()):
            if portfolio_id == current_selected:
                current_index = i
                break

        # Use sidebar or main area based on parameter
        container = st.sidebar if use_sidebar else st

        selected = container.selectbox(
            "📁 Select Portfolio",
            options=list(portfolio_options.keys()),
            index=current_index,
            help="Select a specific portfolio or 'All Portfolios' to view aggregated data",
        )

        new_selection = portfolio_options[selected]

        # Clear cache if portfolio changed to refresh data
        if new_selection != st.session_state.get("selected_portfolio"):
            # Clear portfolio-related caches
            keys_to_clear = [
                k
                for k in st.session_state.keys()
                if "portfolio_" in k and "_cache" in k
            ]
            for key in keys_to_clear:
                del st.session_state[key]

        st.session_state.selected_portfolio = new_selection
        return st.session_state.selected_portfolio

    return None


# Price update functionality
def update_prices():
    """Update current prices for all portfolio assets."""
    try:
        # Check if we recently updated prices (within last 30 seconds)
        if (
            "last_price_update" in st.session_state
            and (datetime.now() - st.session_state.last_price_update).seconds < 30
        ):
            st.info("Prices were updated recently. Skipping update.")
            return

        # Cache symbol collection to avoid repeated DB queries
        cache_key = "symbols_cache"
        cache_timestamp_key = "symbols_cache_timestamp"

        # Refresh symbol cache every 5 minutes or if not exists
        need_refresh = (
            cache_key not in st.session_state
            or cache_timestamp_key not in st.session_state
            or (datetime.now() - st.session_state[cache_timestamp_key]).seconds > 300
        )

        if need_refresh:
            symbols = set()
            portfolios = st.session_state.portfolio_manager.get_all_portfolios()

            for portfolio in portfolios:
                assets = st.session_state.portfolio_manager.get_portfolio_assets(
                    portfolio.id
                )
                for asset in assets:
                    symbols.add(asset.symbol)

            # Add watchlist symbols
            try:
                watchlist = st.session_state.portfolio_manager.get_watchlist()
                for item in watchlist:
                    symbols.add(item.symbol)
            except Exception as e:
                logger.error(f"Error getting watchlist: {e}")

            st.session_state[cache_key] = list(symbols)
            st.session_state[cache_timestamp_key] = datetime.now()
        else:
            symbols = st.session_state.get(cache_key, [])

        if symbols:
            with st.spinner("Updating prices..."):
                prices = fetch_prices_sync(symbols)

                # Check for price alerts only if we got new prices
                if prices:
                    check_price_alerts(prices)

                    st.session_state.current_prices.update(prices)
                    st.session_state.last_price_update = datetime.now()

                    # Cache prices in database
                    st.session_state.portfolio_manager.update_cached_prices(prices)

                    st.success(f"✅ Updated prices for {len(prices)} assets")
                else:
                    st.warning("⚠️ No price data received")
        else:
            st.info("No assets found to update prices for")

    except Exception as e:
        st.error(f"❌ Error updating prices: {e}")
        logger.error(f"Error updating prices: {e}")


def check_price_alerts(new_prices: Dict[str, float]):
    """Check for price alerts and trigger notifications."""
    try:
        cached_prices = st.session_state.portfolio_manager.get_cached_prices()
        threshold = st.session_state.settings.get("price_alert_threshold", 5.0)

        for symbol, new_price in new_prices.items():
            if symbol in cached_prices:
                old_price = cached_prices[symbol]
                change_pct = ((new_price - old_price) / old_price) * 100

                if abs(change_pct) >= threshold:
                    alert_type = "📈 PUMP" if change_pct > 0 else "📉 DUMP"
                    message = f"{alert_type}: {symbol} {change_pct:+.2f}% (${old_price:.4f} → ${new_price:.4f})"

                    # Add to alerts list
                    alert = {
                        "timestamp": datetime.now(),
                        "symbol": symbol,
                        "type": "PRICE_CHANGE",
                        "message": message,
                        "change_pct": change_pct,
                        "old_price": old_price,
                        "new_price": new_price,
                        "active": True,
                        "severity": (
                            "HIGH" if abs(change_pct) >= threshold * 2 else "MEDIUM"
                        ),
                    }

                    st.session_state.alerts.insert(0, alert)

                    # Send notifications if enabled
                    if st.session_state.settings.get("desktop_notifications"):
                        send_desktop_notification("Price Alert", message)

                    if st.session_state.settings.get("whatsapp_notifications"):
                        phone = st.session_state.settings.get("whatsapp_number")
                        if phone:
                            send_whatsapp_notification(message, phone)

        # Keep only last 50 alerts
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts = st.session_state.alerts[:50]

    except Exception as e:
        logger.error(f"Error checking price alerts: {e}")


# Utility functions for data formatting (with comma thousand separators)
def format_currency(value: float, decimals: int = 2) -> str:
    """Format currency value with comma thousand separators and dot decimal separator."""
    if pd.isna(value) or value is None:
        return "$0.00"
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value with proper sign and decimals."""
    if pd.isna(value) or value is None:
        return "0.00%"
    return f"{value:+.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format any number with comma thousand separators and dot decimal separator."""
    if pd.isna(value) or value is None:
        return "0.00"
    return f"{value:,.{decimals}f}"


def format_crypto_quantity(value: float, decimals: int = 6) -> str:
    """Format crypto quantities with higher precision and comma separators."""
    if pd.isna(value) or value is None:
        return "0.000000"
    return f"{value:,.{decimals}f}"


def format_large_number(value: float) -> str:
    """Format large numbers with K, M, B suffixes and comma separators."""
    if pd.isna(value) or value is None:
        return "$0.00"
    if abs(value) >= 1e9:
        return f"${value/1e9:,.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:,.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:,.1f}K"
    else:
        return f"${value:,.2f}"


# Signal analysis functions
def get_signal_color_and_emoji(signal: str) -> Tuple[str, str]:
    """Get color and emoji for signal display."""
    signal_map = {
        "BUY": ("green", "🟢"),
        "SELL": ("red", "🔴"),
        "NEUTRAL": ("gray", "⚪"),
        "OVERSOLD": ("green", "🟢"),
        "OVERBOUGHT": ("red", "🔴"),
    }
    return signal_map.get(signal.upper(), ("gray", "⚪"))


def get_volatility_indicator(volatility_ratio: float) -> Tuple[str, str, str]:
    """Get volatility status, color, and emoji."""
    settings = st.session_state.settings
    low_threshold = settings.get("volatility_low_threshold", 0.7)
    high_threshold = settings.get("volatility_high_threshold", 1.3)

    if np.isnan(volatility_ratio):
        return "N/A", "gray", "⚫"
    elif volatility_ratio < low_threshold:
        return "Low", "green", "🟢"
    elif volatility_ratio < high_threshold:
        return "Moderate", "yellow", "🟡"
    else:
        return "High", "red", "🔴"


def get_trend_indicator(current: float, previous: float) -> Tuple[str, str]:
    """Get trend direction and arrow."""
    if np.isnan(current) or np.isnan(previous):
        return "neutral", "➡️"

    diff = current - previous
    if abs(diff) < 0.001:  # Small threshold for stability
        return "neutral", "➡️"
    elif diff > 0:
        return "up", "↗️"
    else:
        return "down", "↘️"


# Chart creation utilities
def create_portfolio_allocation_chart(
    assets_data: List[Dict], title: str = "Portfolio Allocation"
) -> go.Figure:
    """Create a pie chart for portfolio allocation."""
    if not assets_data:
        # Create an empty chart with informative message
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color="#888")),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text="No assets to display",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="#999"),
                    align="center",
                )
            ],
        )
        return fig

    values = [
        float(row["Current Value"].replace("$", "").replace(",", ""))
        for row in assets_data
    ]
    labels = [row["Symbol"] for row in assets_data]

    # Group small allocations into "Other"
    total_value = sum(values)
    threshold = 0.05 * total_value  # 5% threshold

    filtered_values = []
    filtered_labels = []
    other_value = 0

    for value, label in zip(values, labels):
        if value >= threshold:
            filtered_values.append(value)
            filtered_labels.append(label)
        else:
            other_value += value

    if other_value > 0:
        filtered_values.append(other_value)
        filtered_labels.append("Other")

    # Use nice colors
    colors = px.colors.qualitative.Set3

    fig = go.Figure(
        data=[
            go.Pie(
                labels=filtered_labels,
                values=filtered_values,
                hole=0.3,
                marker_colors=colors[: len(filtered_labels)],
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>"
                + "Value: $%{value:,.2f}<br>"
                + "Percentage: %{percent}<br>"
                + "<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
    )

    return fig


def create_performance_bar_chart(
    assets_data: List[Dict],
    exclude_stablecoins: bool = True,
    title: str = "Asset Performance",
) -> go.Figure:
    """Create horizontal bar chart for asset performance."""
    if not assets_data:
        return go.Figure()

    # Filter out stablecoins if requested
    filtered_data = assets_data
    if exclude_stablecoins:
        stablecoins = ["USDT", "USDC", "BUSD", "DAI"]
        filtered_data = [row for row in assets_data if row["Symbol"] not in stablecoins]

    if not filtered_data:
        return go.Figure()

    symbols = [row["Symbol"] for row in filtered_data]
    # Handle both percentage and dollar formats
    pnl_values = []
    is_dollar_format = False

    for row in filtered_data:
        value_str = row["P&L %"]
        if "$" in value_str:
            is_dollar_format = True
        # Remove common formatting characters
        cleaned_value = (
            value_str.replace("%", "")
            .replace("$", "")
            .replace("+", "")
            .replace(",", "")
        )
        pnl_values.append(float(cleaned_value))

    colors = ["#28a745" if v >= 0 else "#dc3545" for v in pnl_values]

    # Format text and hover based on whether it's dollar or percentage
    if is_dollar_format:
        text_labels = [f"${v:+.2f}" for v in pnl_values]
        hover_template = "<b>%{y}</b><br>" + "P&L: $%{x:+.2f}<br>" + "<extra></extra>"
        x_axis_title = "P&L ($)"
    else:
        text_labels = [f"{v:+.1f}%" for v in pnl_values]
        hover_template = "<b>%{y}</b><br>" + "P&L: %{x:+.2f}%<br>" + "<extra></extra>"
        x_axis_title = "P&L %"

    fig = go.Figure(
        data=[
            go.Bar(
                y=symbols,
                x=pnl_values,
                orientation="h",
                marker_color=colors,
                text=text_labels,
                textposition="auto",
                hovertemplate=hover_template,
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title="Assets",
        showlegend=False,
        height=max(300, len(symbols) * 30),
        bargap=0.2,
    )

    # Add vertical line at 0
    fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.5)

    return fig


# Enhanced Database Caching System
def load_cached_prices_to_session():
    """Load cached prices from database to session state with freshness check."""
    try:
        cached_prices = st.session_state.portfolio_manager.get_cached_prices()
        cached_ages = st.session_state.portfolio_manager.get_cached_prices_age()

        fresh_prices = {}
        stale_symbols = []

        for symbol, price in cached_prices.items():
            if symbol in cached_ages:
                age = datetime.now() - cached_ages[symbol].replace(tzinfo=None)
                if age.total_seconds() < 600:  # Less than 10 minutes old
                    fresh_prices[symbol] = price
                else:
                    stale_symbols.append(symbol)
            else:
                stale_symbols.append(symbol)

        # Update session state with fresh cached data
        if fresh_prices:
            st.session_state.current_prices.update(fresh_prices)
            logger.info(f"Loaded {len(fresh_prices)} fresh prices from cache")

        # Return stale symbols that need refresh
        return stale_symbols

    except Exception as e:
        logger.error(f"Error loading cached prices: {e}")
        return []


def poll_database_for_updates() -> Dict[str, any]:
    """Poll database for fresh data and return update status (database-first approach)."""
    try:
        update_status = {
            "prices_updated": 0,
            "new_data_available": False,
            "last_update": None,
            "data_freshness": "stale",
        }

        # Get all symbols we need to track
        symbols = get_all_tracking_symbols()
        if not symbols:
            return update_status

        # Always load from database cache
        cached_prices = get_prices_from_database(symbols)

        # Count how many prices we have
        update_status["prices_updated"] = len(cached_prices)
        update_status["new_data_available"] = len(cached_prices) > 0

        # Update session state with fresh database data
        if cached_prices:
            if not hasattr(st.session_state, "current_prices"):
                st.session_state.current_prices = {}
            st.session_state.current_prices.update(cached_prices)

        # Check data freshness from database timestamps
        try:
            from database.models import get_session, CachedPrice

            session = get_session()
            latest_update = (
                session.query(CachedPrice)
                .filter(CachedPrice.symbol.in_(symbols))
                .order_by(CachedPrice.last_updated.desc())
                .first()
            )

            if latest_update:
                update_status["last_update"] = latest_update.last_updated
                st.session_state.last_price_update = latest_update.last_updated

                time_diff = datetime.now() - latest_update.last_updated
                minutes_ago = int(time_diff.total_seconds() / 60)

                if minutes_ago < 2:
                    update_status["data_freshness"] = "fresh"
                elif minutes_ago < 10:
                    update_status["data_freshness"] = "good"
                else:
                    update_status["data_freshness"] = "stale"
            else:
                # No cached data available
                update_status["data_freshness"] = "no_data"

            session.close()

        except Exception as db_error:
            logger.error(f"Error checking database timestamps: {db_error}")
            update_status["data_freshness"] = "error"

        return update_status

    except Exception as e:
        logger.error(f"Error polling database: {e}")
        return {
            "prices_updated": 0,
            "new_data_available": False,
            "last_update": None,
            "data_freshness": "error",
            "error": str(e),
        }


def smart_price_refresh(force_refresh: bool = False):
    """Database-first price refresh - loads from database cache (no API calls)."""
    try:
        # Ensure session state is initialized before doing anything
        if not hasattr(st.session_state, "current_prices"):
            st.session_state.current_prices = {}

        # Get all symbols we need to track
        symbols = get_all_tracking_symbols()

        if not symbols:
            # Still return True - empty state is valid
            return True

        # ALWAYS load from database cache - database-first approach
        cached_prices = get_prices_from_database(symbols)

        if cached_prices:
            # Update session state with cached prices - be more aggressive about syncing
            if not hasattr(st.session_state, "current_prices"):
                st.session_state.current_prices = {}

            # Always fully replace with fresh database data to ensure sync
            st.session_state.current_prices = cached_prices.copy()

            # Update the last portfolio price update timestamp
            st.session_state.last_portfolio_price_update = datetime.now()

            # IMPORTANT: Clear portfolio history cache when prices are updated
            # This ensures period calculations use fresh price data
            if hasattr(st.session_state, "historical_data") and force_refresh:
                # Clear all portfolio history cache to force recalculation with new prices
                keys_to_remove = []
                for key in list(st.session_state.historical_data.keys()):
                    if key.endswith("_history") or "_portfolio" in key:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    if key in st.session_state.historical_data:
                        del st.session_state.historical_data[key]

                if keys_to_remove and st.session_state.settings.get(
                    "show_debug_info", False
                ):
                    st.info(
                        f"🔧 Debug: Cleared {len(keys_to_remove)} portfolio history cache entries for fresh calculations"
                    )

            # Update last refresh time based on database timestamp if available
            try:
                from database.models import get_session, CachedPrice

                session = get_session()
                latest_update = (
                    session.query(CachedPrice)
                    .filter(CachedPrice.symbol.in_(symbols))
                    .order_by(CachedPrice.last_updated.desc())
                    .first()
                )

                if latest_update:
                    st.session_state.last_price_update = latest_update.last_updated
                else:
                    st.session_state.last_price_update = datetime.now()

                session.close()

            except Exception as e:
                # Fallback to current time
                st.session_state.last_price_update = datetime.now()

            if st.session_state.settings.get("show_debug_info", False):
                st.success(
                    f"🔧 Debug: Loaded {len(cached_prices)} prices from database cache"
                )

            logger.info(
                f"Successfully loaded {len(cached_prices)} prices from database cache"
            )
            return True
        else:
            # No cached data available
            if st.session_state.settings.get("show_debug_info", False):
                st.warning(
                    "🔧 Debug: No cached price data available. Background service needs to collect data."
                )
            logger.warning("No cached price data available")
            return False

    except Exception as e:
        logger.error(f"Error in smart price refresh: {e}")
        return False


def get_all_tracking_symbols() -> List[str]:
    """Get all symbols that should be tracked from TrackedAsset table (database-first approach)."""
    try:
        # Sync tracked assets less frequently to avoid database contention
        # Only sync if we haven't done it recently or if we have no tracked assets
        should_sync = False

        if "last_tracked_assets_sync" not in st.session_state:
            should_sync = True
        else:
            time_since_sync = (
                datetime.now() - st.session_state.last_tracked_assets_sync
            ).total_seconds()
            should_sync = time_since_sync > 300  # Only sync every 5 minutes

        # Get symbols from TrackedAsset table first
        tracked_assets = st.session_state.portfolio_manager.get_tracked_assets(
            active_only=True
        )

        # If we have no tracked assets or it's time to sync, then sync
        if not tracked_assets or should_sync:
            try:
                st.session_state.portfolio_manager.sync_tracked_assets()
                st.session_state.last_tracked_assets_sync = datetime.now()
                # Get updated tracked assets
                tracked_assets = st.session_state.portfolio_manager.get_tracked_assets(
                    active_only=True
                )
            except Exception as e:
                logger.warning(f"Could not sync tracked assets (using existing): {e}")

        symbols = [asset.symbol for asset in tracked_assets]
        logger.info(f"Using TrackedAsset table: {len(symbols)} active symbols")
        return symbols

    except Exception as e:
        logger.error(f"Error getting tracked symbols from database: {e}")
        # Fallback to old method if TrackedAsset table has issues
        try:
            symbols = set()

            # Get portfolio assets as fallback
            all_assets = st.session_state.portfolio_manager.get_all_assets()
            for asset in all_assets:
                symbols.add(asset.symbol)

            # Get watchlist items as fallback
            try:
                watchlist = st.session_state.portfolio_manager.get_watchlist()
                for item in watchlist:
                    symbols.add(item.symbol)
            except Exception:
                pass

            logger.warning(f"Using fallback method: {len(symbols)} symbols")
            return list(symbols)

        except Exception as fallback_error:
            logger.error(f"Fallback method also failed: {fallback_error}")
            return []


def get_cache_status_info() -> Dict[str, Any]:
    """Get detailed information about cache status for debugging/monitoring."""
    try:
        cached_prices = st.session_state.portfolio_manager.get_cached_prices()
        cached_ages = st.session_state.portfolio_manager.get_cached_prices_age()
        tracking_symbols = get_all_tracking_symbols()

        fresh_count = 0
        stale_count = 0
        missing_count = 0

        for symbol in tracking_symbols:
            if symbol in cached_prices and symbol in cached_ages:
                age = datetime.now() - cached_ages[symbol].replace(tzinfo=None)
                if age.total_seconds() < 600:  # 10 minutes
                    fresh_count += 1
                else:
                    stale_count += 1
            else:
                missing_count += 1

        return {
            "total_symbols": len(tracking_symbols),
            "fresh_cached": fresh_count,
            "stale_cached": stale_count,
            "missing": missing_count,
            "cache_efficiency": (
                (fresh_count / len(tracking_symbols)) * 100 if tracking_symbols else 0
            ),
            "last_session_update": st.session_state.get("last_price_update"),
            "cached_symbols": list(cached_prices.keys()),
        }

    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return {"error": str(e)}


def display_cache_status():
    """Display cache status information in the UI for debugging."""
    if st.session_state.settings.get("show_debug_info", False):
        with st.expander("🔧 Cache Status (Debug)", expanded=False):
            cache_info = get_cache_status_info()

            if "error" in cache_info:
                st.error(f"Cache error: {cache_info['error']}")
                return

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Symbols", cache_info["total_symbols"])

            with col2:
                st.metric(
                    "Fresh Cache",
                    cache_info["fresh_cached"],
                    delta=f"{cache_info['cache_efficiency']:.1f}%",
                )

            with col3:
                st.metric("Stale Cache", cache_info["stale_cached"])

            with col4:
                st.metric("Missing", cache_info["missing"])

            if cache_info["last_session_update"]:
                age = datetime.now() - cache_info["last_session_update"]
                st.info(f"Last session update: {age.total_seconds():.0f}s ago")

            if cache_info["cached_symbols"]:
                st.text(f"Cached symbols: {', '.join(cache_info['cached_symbols'])}")


# =============================================================================
# DATABASE-FIRST AUTO-REFRESH SYSTEM
# =============================================================================


def init_auto_refresh_system():
    """Initialize the auto-refresh system for database-first updates."""
    if "auto_refresh_initialized" not in st.session_state:
        st.session_state.auto_refresh_initialized = True
        st.session_state.last_auto_poll_time = datetime.now()
        st.session_state.auto_refresh_counter = 0


def should_auto_refresh() -> bool:
    """Check if it's time to auto-refresh based on user settings."""
    if not st.session_state.settings.get("auto_refresh", True):
        return False

    refresh_interval = st.session_state.settings.get("refresh_interval", 60)

    # Initialize timer system
    if "last_successful_refresh" not in st.session_state:
        st.session_state.last_successful_refresh = datetime.now() - timedelta(
            seconds=refresh_interval + 1
        )
        logger.info(f"Initialized auto-refresh timer - will refresh immediately")

    time_since_last = (
        datetime.now() - st.session_state.last_successful_refresh
    ).total_seconds()

    # Use user's refresh interval setting
    should_refresh = time_since_last >= refresh_interval

    # Console logging to track timer resets
    if time_since_last < 5:
        logger.warning(
            f"TIMER RESET DETECTED: Only {time_since_last:.1f}s since last refresh - something is resetting the timer!"
        )

    # Prevent rapid refresh cycles by adding cooldown
    if time_since_last < 5:
        logger.debug(
            f"Auto-refresh cooldown: {time_since_last:.1f}s elapsed, waiting for 5s minimum"
        )
        return False

    logger.debug(
        f"Auto-refresh check: {time_since_last:.1f}s elapsed, need {refresh_interval}s, should_refresh={should_refresh}"
    )

    return should_refresh


def auto_refresh_database_data():
    """Auto-refresh data from database and trigger UI update if needed."""
    try:
        init_auto_refresh_system()

        if not should_auto_refresh():
            return False

        # Store current data hash to detect changes
        current_prices_hash = (
            hash(str(sorted(st.session_state.current_prices.items())))
            if st.session_state.current_prices
            else 0
        )

        st.session_state.auto_refresh_counter = (
            getattr(st.session_state, "auto_refresh_counter", 0) + 1
        )

        # Load fresh data from database cache (no API calls)
        success = smart_price_refresh(force_refresh=True)

        if success and st.session_state.current_prices:
            price_count = len(st.session_state.current_prices)
            logger.info(f"Loaded {price_count} prices from database")
        else:
            logger.error("Failed to load data from database")

        if success:
            # Check if data actually changed
            new_prices_hash = (
                hash(str(sorted(st.session_state.current_prices.items())))
                if st.session_state.current_prices
                else 0
            )

            if new_prices_hash != current_prices_hash:
                logger.info(
                    f"Auto-refresh cycle #{st.session_state.auto_refresh_counter}: Data changed - triggering UI refresh"
                )

                # Update timestamp ONLY after successful data change
                st.session_state.last_successful_refresh = datetime.now()
                logger.info(
                    "Timer reset: Data changed, updated last_successful_refresh"
                )

                # Only rerun if data actually changed
                st.rerun()
                return True
            else:
                logger.debug(
                    f"Auto-refresh cycle #{st.session_state.auto_refresh_counter}: No data changes detected, but forcing UI refresh"
                )

                # Update timestamp and force refresh anyway to update timestamps in UI
                st.session_state.last_successful_refresh = datetime.now()
                logger.info(
                    "Timer reset: No data changes, but refreshing UI and resetting timer"
                )
                st.rerun()
                return True
        else:
            logger.warning(
                f"Auto-refresh cycle #{st.session_state.auto_refresh_counter}: No new data available"
            )
            return False

    except Exception as e:
        logger.error(f"Error in auto-refresh: {e}")
        return False


def start_auto_refresh_timer():
    """Start the auto-refresh timer that triggers database polling."""
    auto_refresh_enabled = st.session_state.settings.get("auto_refresh", True)
    refresh_interval = st.session_state.settings.get("refresh_interval", 60)

    # Clean UI - show essential status and countdown
    if "last_successful_refresh" in st.session_state:
        time_since = (
            datetime.now() - st.session_state.last_successful_refresh
        ).total_seconds()
        next_refresh = max(0, refresh_interval - time_since)
        st.sidebar.markdown(
            f"**🔄 Auto-refresh**: {'ON' if auto_refresh_enabled else 'OFF'} ({next_refresh:.0f}s)"
        )
    else:
        st.sidebar.markdown(
            f"**🔄 Auto-refresh**: {'ON' if auto_refresh_enabled else 'OFF'} ({refresh_interval}s)"
        )

    if not auto_refresh_enabled:
        return

    # Always ensure we have fresh data loaded from database cache
    # This ensures session state is synchronized with database
    if (
        not hasattr(st.session_state, "current_prices")
        or not st.session_state.current_prices
    ):
        smart_price_refresh(force_refresh=True)

    # Check if we need to refresh (with console logging only)
    if should_auto_refresh():
        logger.info("🔄 Auto-refresh timer triggered - attempting database refresh")
        # Load fresh data from database (auto_refresh_database_data handles st.rerun())
        auto_refresh_database_data()


def setup_smooth_auto_refresh():
    """Setup smooth auto-refresh that only updates data without page reload."""
    if not st.session_state.settings.get("auto_refresh", True):
        return

    # Alternative approach: Use a refresh button that auto-clicks
    refresh_interval = st.session_state.settings.get("refresh_interval", 60)

    # Initialize auto-refresh timer
    if "auto_refresh_start_time" not in st.session_state:
        st.session_state.auto_refresh_start_time = datetime.now()

    # Calculate elapsed time
    elapsed = (
        datetime.now() - st.session_state.auto_refresh_start_time
    ).total_seconds()

    # If enough time has passed, trigger refresh
    if elapsed >= refresh_interval:
        st.session_state.auto_refresh_start_time = datetime.now()

        # Load fresh data
        success = smart_price_refresh(force_refresh=True)
        if success:
            st.sidebar.success(
                f"🔄 Auto-refreshed at {datetime.now().strftime('%H:%M:%S')}"
            )
            # Force UI refresh using experimental_rerun
            st.rerun()

    # Show countdown
    remaining = max(0, refresh_interval - elapsed)
    if remaining <= 10:
        st.sidebar.markdown(f"🔄 Auto-refresh in {remaining:.0f}s")
    else:
        st.sidebar.markdown(f"⏳ Next auto-refresh: {remaining:.0f}s")


@st.fragment(run_every=None)
def create_auto_refreshing_fragment(refresh_interval_seconds: int = 60):
    """Create a fragment that auto-refreshes at specified intervals."""

    # This fragment will be used for components that need frequent updates
    # without affecting the entire page
    if not st.session_state.settings.get("auto_refresh", True):
        return False

    # Check if enough time has passed for a refresh
    if "fragment_last_refresh" not in st.session_state:
        st.session_state.fragment_last_refresh = datetime.now() - timedelta(
            seconds=refresh_interval_seconds + 1
        )

    time_since_refresh = (
        datetime.now() - st.session_state.fragment_last_refresh
    ).total_seconds()

    if time_since_refresh >= refresh_interval_seconds:
        # Update timestamp
        st.session_state.fragment_last_refresh = datetime.now()

        # Load fresh data from database
        success = smart_price_refresh(force_refresh=True)

        if success:
            logger.info("Fragment-based refresh: Successfully loaded fresh data")
            return True

    return False


def setup_clean_auto_refresh():
    """Setup clean auto-refresh with HTML meta refresh (no UI clutter)."""
    if not st.session_state.settings.get("auto_refresh", True):
        return

    # Get refresh interval (default 5 minutes = 300 seconds)
    refresh_interval = st.session_state.settings.get("refresh_interval", 300)

    # Add HTML meta refresh - reliable and works in background
    # meta_refresh = f'<meta http-equiv="refresh" content="{refresh_interval}">'
    # st.markdown(meta_refresh, unsafe_allow_html=True)  # DISABLED - may cause spacing issues

    # Load fresh data on each page load
    if not hasattr(st.session_state, "data_loaded_this_session"):
        smart_price_refresh(force_refresh=True)
        st.session_state.data_loaded_this_session = True
        logger.info(
            f"Auto-refresh: Fresh data loaded, page will refresh in {refresh_interval}s ({refresh_interval/60:.1f} minutes)"
        )


def create_auto_refresh_mechanism():
    """Create a more reliable auto-refresh mechanism using time-based approach."""
    if not st.session_state.settings.get("auto_refresh", True):
        return

    refresh_interval = st.session_state.settings.get("refresh_interval", 60)

    # Use a simple time-based approach that works every time the page is accessed
    current_time = datetime.now()

    # Initialize or get last refresh time
    if "last_successful_refresh" not in st.session_state:
        st.session_state.last_successful_refresh = current_time - timedelta(
            seconds=refresh_interval + 1
        )

    time_since_refresh = (
        current_time - st.session_state.last_successful_refresh
    ).total_seconds()

    # Auto-refresh condition: enough time has passed
    if time_since_refresh >= refresh_interval:
        st.sidebar.markdown("🔄 **AUTO-REFRESHING NOW...**")

        # Refresh data
        success = smart_price_refresh(force_refresh=True)

        if success:
            new_price_count = (
                len(st.session_state.current_prices)
                if st.session_state.current_prices
                else 0
            )
            st.session_state.last_successful_refresh = current_time

            st.sidebar.success(
                f"✅ Refreshed {new_price_count} prices at {current_time.strftime('%H:%M:%S')}"
            )

            # Force a rerun to show updated data
            time.sleep(0.1)  # Small delay to ensure data is processed
            st.rerun()
        else:
            st.sidebar.warning("⚠️ Auto-refresh failed")

    else:
        # Show countdown
        remaining = refresh_interval - time_since_refresh
        if remaining <= 10:
            st.sidebar.info(f"🔄 Auto-refresh in {remaining:.0f}s")


@contextmanager
def database_polling_context():
    """Context manager for database polling that ensures proper cleanup."""
    try:
        # Initialize auto-refresh system
        init_auto_refresh_system()
        yield
    finally:
        # Could add cleanup logic here if needed
        pass


def poll_database_continuously():
    """Continuously poll database for updates (called by each page)."""
    try:
        with database_polling_context():
            # Ensure settings persistence is active
            ensure_settings_persistence()

            # Always ensure we have the latest database data loaded first
            if (
                not hasattr(st.session_state, "current_prices")
                or not st.session_state.current_prices
            ):
                smart_price_refresh(force_refresh=True)
                logger.info("Initial database data loaded to session state")

            # Use HTML meta refresh with 5-minute default interval
            setup_clean_auto_refresh()

    except Exception as e:
        logger.error(f"Error in continuous database polling: {e}")


def get_database_freshness_info() -> Dict[str, Any]:
    """Get information about database data freshness for UI display."""
    try:
        from database.models import get_session, CachedPrice

        session = get_session()

        # Get latest database update time
        latest_update = (
            session.query(CachedPrice).order_by(CachedPrice.last_updated.desc()).first()
        )

        if latest_update:
            age_seconds = (datetime.now() - latest_update.last_updated).total_seconds()
            age_minutes = age_seconds / 60

            # Determine freshness level
            if age_seconds < 30:
                freshness_level = "live"
                freshness_color = "🟢"
                freshness_text = f"Live ({age_seconds:.0f}s)"
            elif age_minutes < 2:
                freshness_level = "fresh"
                freshness_color = "🟢"
                freshness_text = f"Fresh ({age_seconds:.0f}s)"
            elif age_minutes < 5:
                freshness_level = "recent"
                freshness_color = "🟡"
                freshness_text = f"Recent ({age_minutes:.0f}m)"
            elif age_minutes < 15:
                freshness_level = "stale"
                freshness_color = "🟠"
                freshness_text = f"Stale ({age_minutes:.0f}m)"
            else:
                freshness_level = "very_stale"
                freshness_color = "🔴"
                freshness_text = f"Very Stale ({age_minutes:.0f}m)"

            session.close()

            return {
                "last_update": latest_update.last_updated,
                "age_seconds": age_seconds,
                "age_minutes": age_minutes,
                "freshness_level": freshness_level,
                "freshness_color": freshness_color,
                "freshness_text": freshness_text,
            }
        else:
            session.close()
            return {
                "last_update": None,
                "age_seconds": None,
                "age_minutes": None,
                "freshness_level": "no_data",
                "freshness_color": "⚪",
                "freshness_text": "No Data",
            }

    except Exception as e:
        logger.error(f"Error getting database freshness info: {e}")
        return {
            "last_update": None,
            "age_seconds": None,
            "age_minutes": None,
            "freshness_level": "error",
            "freshness_color": "❌",
            "freshness_text": "Error",
        }


def show_auto_refresh_status():
    """Show auto-refresh status in the sidebar."""
    try:
        freshness_info = get_database_freshness_info()

        # Show current data freshness
        st.sidebar.markdown(
            f"**Data Status**: {freshness_info['freshness_color']} {freshness_info['freshness_text']}"
        )

        # Show auto-refresh status
        if st.session_state.settings.get("auto_refresh", True):
            refresh_interval = st.session_state.settings.get("refresh_interval", 60)

            if "last_auto_poll_time" in st.session_state:
                time_since = (
                    datetime.now() - st.session_state.last_auto_poll_time
                ).total_seconds()
                next_refresh = max(0, refresh_interval - time_since)

                if next_refresh <= 2:
                    st.sidebar.markdown("🔄 **Refreshing Soon...**")
                else:
                    mins = int(next_refresh // 60)
                    secs = int(next_refresh % 60)
                    if mins > 0:
                        st.sidebar.markdown(f"⏱️ **Next Refresh**: {mins}m {secs}s")
                    else:
                        st.sidebar.markdown(f"⏱️ **Next Refresh**: {secs}s")
        else:
            st.sidebar.markdown("⏸️ **Auto-refresh Disabled**")

    except Exception as e:
        st.sidebar.error(f"Auto-refresh error: {e}")


def save_all_settings_to_database():
    """Save all current settings to database for persistence across sessions."""
    try:
        if "settings" in st.session_state and st.session_state.settings:
            # Save all settings to database
            for key, value in st.session_state.settings.items():
                st.session_state.portfolio_manager.set_setting(key, str(value))

            logger.info(f"Saved {len(st.session_state.settings)} settings to database")

    except Exception as e:
        logger.error(f"Error saving settings to database: {e}")


def load_all_settings_from_database():
    """Load all settings from database and update session state."""
    try:
        saved_settings = st.session_state.portfolio_manager.get_all_settings()

        if saved_settings and st.session_state.settings:
            settings_loaded = 0
            for key, value in saved_settings.items():
                if key in st.session_state.settings:
                    # Convert string back to appropriate type based on default value
                    default_value = st.session_state.settings[key]

                    if isinstance(default_value, bool):
                        st.session_state.settings[key] = value.lower() == "true"
                    elif isinstance(default_value, int):
                        st.session_state.settings[key] = int(value)
                    elif isinstance(default_value, float):
                        st.session_state.settings[key] = float(value)
                    else:
                        st.session_state.settings[key] = value

                    settings_loaded += 1

            logger.info(f"Loaded {settings_loaded} settings from database")

    except Exception as e:
        logger.error(f"Error loading settings from database: {e}")


def auto_save_settings():
    """Automatically save settings when they change (call this after settings updates)."""
    try:
        # Check if settings have changed by comparing with a stored hash
        import hashlib

        current_settings_str = json.dumps(st.session_state.settings, sort_keys=True)
        current_hash = hashlib.md5(current_settings_str.encode()).hexdigest()

        # Only save if settings changed and we haven't saved recently (reduce database contention)
        should_save = False
        if (
            "settings_hash" not in st.session_state
            or st.session_state.settings_hash != current_hash
        ):
            should_save = True

        if should_save:
            # Rate limit auto-saves to avoid database locks
            if "last_settings_save" not in st.session_state:
                st.session_state.last_settings_save = datetime.now() - timedelta(
                    seconds=60
                )

            time_since_save = (
                datetime.now() - st.session_state.last_settings_save
            ).total_seconds()
            if time_since_save > 30:  # Only auto-save every 30 seconds
                save_all_settings_to_database()
                st.session_state.settings_hash = current_hash
                st.session_state.last_settings_save = datetime.now()

    except Exception as e:
        logger.error(f"Error in auto-save settings: {e}")


def ensure_settings_persistence():
    """Ensure settings are loaded and will be auto-saved."""
    try:
        # Load settings from database if not already loaded
        if "settings_loaded_from_db" not in st.session_state:
            load_all_settings_from_database()
            st.session_state.settings_loaded_from_db = True

        # Auto-save if settings have changed
        auto_save_settings()

    except Exception as e:
        logger.error(f"Error ensuring settings persistence: {e}")


def is_operation_in_progress(operation_type: str = "general") -> bool:
    """
    Check if an operation is currently in progress to prevent duplicate operations.
    """
    operation_key = f"operation_in_progress_{operation_type}"
    return st.session_state.get(operation_key, False)


def set_operation_in_progress(
    operation_type: str = "general", in_progress: bool = True
):
    """
    Set operation status to prevent duplicate operations.
    """
    operation_key = f"operation_in_progress_{operation_type}"
    st.session_state[operation_key] = in_progress


def refresh_portfolio_data_after_operation():
    """
    Enhanced refresh function to address Streamlit state sync issues.
    This ensures the UI reflects the latest database state and prevents repeated operations.

    This function should be called after any database modification operation
    (add_asset, update_asset, delete_asset, etc.) to ensure the UI stays in sync.
    """
    try:
        # STEP 1: Force complete session state cache clear
        cache_patterns_to_clear = [
            "cached",
            "cache",
            "_cache",
            "portfolio_",
            "_history",
            "historical_data",
            "last_portfolio_price_update",
            "last_table_price_check",
            "last_auto_rerun",
            "period_tracker_",
            "_assets_",
            "_portfolios_",
        ]

        keys_to_clear = []
        for key in list(st.session_state.keys()):
            if any(pattern in key.lower() for pattern in cache_patterns_to_clear):
                keys_to_clear.append(key)

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        if st.session_state.settings.get("show_debug_info", False):
            st.info(f"🔧 Debug: Cleared {len(keys_to_clear)} cache keys for UI sync")

        # STEP 2: Force complete portfolio manager refresh
        if hasattr(st.session_state, "portfolio_manager"):
            # Multiple attempts to ensure database consistency
            for attempt in range(3):
                try:
                    fresh_prices = (
                        st.session_state.portfolio_manager.get_cached_prices()
                    )
                    if fresh_prices:
                        # Replace, don't update, to avoid conflicts
                        st.session_state.current_prices = fresh_prices.copy()
                        if st.session_state.settings.get("show_debug_info", False):
                            st.info(
                                f"🔧 Debug: Replaced session prices with {len(fresh_prices)} fresh DB prices (attempt {attempt + 1})"
                            )
                        break
                    else:
                        # Reset to empty if no prices available
                        st.session_state.current_prices = {}
                except Exception as price_error:
                    if attempt == 2:  # Last attempt
                        st.session_state.current_prices = {}
                        logger.warning(
                            f"Could not refresh prices from database after 3 attempts: {price_error}"
                        )

            # Force tracked assets sync to prevent repeated operations
            try:
                st.session_state.portfolio_manager.sync_tracked_assets()
                if st.session_state.settings.get("show_debug_info", False):
                    st.info(f"🔧 Debug: Synced tracked assets with background service")
            except Exception as sync_error:
                logger.warning(f"Could not sync tracked assets: {sync_error}")

        # STEP 3: Add operation completion marker to prevent repeated operations
        operation_marker_key = f"operation_completed_{datetime.now().timestamp()}"
        st.session_state[operation_marker_key] = True

        # Clean old operation markers (keep only last 5)
        operation_markers = [
            k for k in st.session_state.keys() if k.startswith("operation_completed_")
        ]
        if len(operation_markers) > 5:
            old_markers = sorted(operation_markers)[:-5]
            for marker in old_markers:
                del st.session_state[marker]

        if st.session_state.settings.get("show_debug_info", False):
            st.success("🔧 Debug: Portfolio data refresh completed successfully")

        # STEP 4: Set rerun request flag (but don't immediately rerun to prevent conflicts)
        st.session_state._rerun_requested = True

    except Exception as e:
        st.error(f"Error refreshing portfolio data: {e}")
        logger.error(f"Error in refresh_portfolio_data_after_operation: {e}")


# Empty State Components
def show_empty_state(
    title: str,
    message: str,
    icon: str = "📊",
    action_text: Optional[str] = None,
    action_key: Optional[str] = None,
):
    """
    Display a user-friendly empty state with optional call-to-action.

    Args:
        title: Main title for the empty state
        message: Descriptive message explaining the empty state
        icon: Emoji icon to display
        action_text: Optional text for call-to-action button
        action_key: Optional key for the action button

    Returns:
        bool: True if action button was clicked (if provided)
    """
    with st.container():
        # Center align content
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
            <div style="text-align: center; padding: 2rem 0; color: #666;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                <h3 style="color: #888; margin-bottom: 0.5rem;">{title}</h3>
                <p style="color: #999; font-size: 0.9rem; line-height: 1.4;">{message}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if action_text and action_key:
                return st.button(action_text, key=action_key, type="primary")

    return False


def show_empty_chart(
    title: str = "No Data Available",
    message: str = "No data to display for this chart.",
    height: int = 400,
):
    """
    Display an empty chart placeholder with informative message.

    Args:
        title: Title for the empty chart
        message: Description of why the chart is empty
        height: Height of the placeholder in pixels
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add invisible trace to maintain chart structure
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=0, opacity=0),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Style the empty chart
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color="#888")),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="#999"),
                align="center",
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)


def show_empty_portfolio_table(portfolio_name: str = "portfolio"):
    """
    Display an empty portfolio table with helpful guidance.

    Args:
        portfolio_name: Name of the portfolio that's empty
    """
    show_empty_state(
        title="Empty Portfolio",
        message=f"Your {portfolio_name} doesn't have any assets yet. Add some cryptocurrencies to start tracking your investments!",
        icon="💼",
        action_text="➕ Add Your First Asset",
        action_key="empty_portfolio_add_asset",
    )


def show_empty_watchlist():
    """Display empty watchlist state with call-to-action."""
    return show_empty_state(
        title="Empty Watchlist",
        message="Your watchlist is empty. Add some cryptocurrencies to monitor their performance and receive alerts!",
        icon="👀",
        action_text="➕ Add to Watchlist",
        action_key="empty_watchlist_add",
    )


def show_empty_profit_history():
    """Display empty profit history state."""
    show_empty_state(
        title="No Profit History",
        message="You haven't made any trades yet. Your realized profits and losses will appear here after you sell assets.",
        icon="💰",
    )


def show_insufficient_data_for_analysis(symbol: str, required_days: int = 14):
    """
    Display message for insufficient historical data for technical analysis.

    Args:
        symbol: The cryptocurrency symbol
        required_days: Minimum days of data required
    """
    show_empty_state(
        title="Insufficient Data",
        message=f"Not enough historical data for {symbol}. Technical analysis requires at least {required_days} days of price data.",
        icon="📈",
    )


def check_and_handle_empty_assets(assets: List, page_name: str = "page") -> bool:
    """
    Check if assets list is empty and show appropriate empty state.

    Args:
        assets: List of assets to check
        page_name: Name of the current page for context

    Returns:
        bool: True if assets are empty (caller should return early)
    """
    if not assets:
        if page_name.lower() == "monitoring":
            show_empty_state(
                title="No Assets to Monitor",
                message="Add some assets to your portfolio to start monitoring price movements and technical signals.",
                icon="🔔",
            )
        elif page_name.lower() == "charts":
            show_empty_state(
                title="No Assets for Charting",
                message="Add assets to your portfolio or watchlist to view detailed price charts and technical analysis.",
                icon="📈",
            )
        else:
            show_empty_portfolio_table()
        return True
    return False


def safe_create_portfolio_chart(
    data: List[Dict],
    title: str = "Portfolio Chart",
    height: int = 400,
):
    """
    Safely create portfolio charts with empty state handling.

    Args:
        data: List of data points for the chart
        title: Chart title
        height: Chart height in pixels

    Returns:
        bool: True if chart was created, False if empty state was shown
    """
    if not data or len(data) == 0:
        show_empty_chart(
            title=title,
            message="No historical data available for this time period.",
            height=height,
        )
        return False

    return True
