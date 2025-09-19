"""
Monitoring Page - Enhanced monitoring and alerts with advanced signal detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from components.shared import (
    send_desktop_notification,
    send_whatsapp_notification,
    get_signal_color_and_emoji,
    get_volatility_indicator,
    get_trend_indicator,
    universal_timeframe_selector,
    display_last_update_time,
    get_historical_data_from_database,
    format_currency,
    format_number,
    show_empty_state,
    check_and_handle_empty_assets,
)
from services.technical_indicators import TechnicalIndicators


def get_extended_historical_data(symbol: str, days: int, interval: str) -> List[Dict]:
    """Get extended historical data using Asset Charts approach with proper data limits."""
    try:
        from database.models import get_session, HistoricalPrice
        from datetime import timedelta

        # Use same maximum data periods as Asset Charts
        max_data_periods = {
            "1h": 30,  # 30 days max for 1h (720 hourly points)
            "4h": 90,  # 90 days max for 4h (540 4-hour points)
            "1d": 1825,  # 5 years max for 1d (1825 daily points)
            "1w": 1825,  # 5 years max for 1w (need daily data to aggregate)
            "1M": 1825,  # 5 years max for 1M (need daily data to aggregate)
        }

        # Get maximum data for this timeframe
        extended_days = max_data_periods.get(interval, days)

        session = get_session()
        end_date = datetime.now()

        # Set date limits based on interval
        if interval == "4h":
            # For 4h, get 30 days of hourly data (720 hourly points -> 180 4h periods)
            start_date = end_date - timedelta(days=30)
        elif interval in ["1M"]:
            start_date = None  # Get all available daily data
        else:
            start_date = end_date - timedelta(days=extended_days)

        if interval == "4h":
            # Fetch hourly data for 4h aggregation
            historical_prices = (
                session.query(HistoricalPrice)
                .filter(
                    HistoricalPrice.symbol == symbol,
                    HistoricalPrice.interval == "1h",
                    HistoricalPrice.date >= start_date,
                    HistoricalPrice.date <= end_date,
                )
                .order_by(HistoricalPrice.date.asc())
                .all()
            )
        elif interval in ["1w", "1M"]:
            # Fetch daily data for weekly/monthly aggregation
            if start_date is not None:
                # Use date filtering for 1w
                historical_prices = (
                    session.query(HistoricalPrice)
                    .filter(
                        HistoricalPrice.symbol == symbol,
                        HistoricalPrice.interval == "1d",
                        HistoricalPrice.date >= start_date,
                        HistoricalPrice.date <= end_date,
                    )
                    .order_by(HistoricalPrice.date.asc())
                    .all()
                )
            else:
                # Get ALL daily data for better SMA coverage
                historical_prices = (
                    session.query(HistoricalPrice)
                    .filter(
                        HistoricalPrice.symbol == symbol,
                        HistoricalPrice.interval == "1d",
                    )
                    .order_by(HistoricalPrice.date.asc())
                    .all()
                )
        else:
            # For other intervals (1h, 1d), fetch data directly
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

        if historical_prices and len(historical_prices) >= 10:
            # Convert to DataFrame and aggregate
            import pandas as pd

            data = []
            for price in historical_prices:
                # Skip records with missing price data
                if price.price is None:
                    continue

                # Use proper close_price field with fallback to legacy price field
                if hasattr(price, "close_price") and price.close_price is not None:
                    close_price = float(price.close_price)
                else:
                    close_price = float(price.price)  # Fallback to legacy field

                # Handle potentially missing OHLC data with null checks
                open_price = close_price  # Default fallback
                if hasattr(price, "open_price") and price.open_price is not None:
                    open_price = float(price.open_price)

                high_price = close_price  # Default fallback
                if hasattr(price, "high_price") and price.high_price is not None:
                    high_price = float(price.high_price)

                low_price = close_price  # Default fallback
                if hasattr(price, "low_price") and price.low_price is not None:
                    low_price = float(price.low_price)

                volume = 0  # Default fallback
                if price.volume is not None:
                    volume = float(price.volume)

                data.append(
                    {
                        "date": price.date,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                    }
                )

            df = pd.DataFrame(data)
            df.set_index("date", inplace=True)

            # Aggregate based on interval
            if interval == "4h":
                # For 4h: Aggregate hourly data into proper 4-hour candles starting from midnight
                # This creates genuine 4-hour periods: 00-04, 04-08, 08-12, 12-16, 16-20, 20-24
                agg_df = (
                    df.resample("4h", origin="start")
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
            elif interval == "1w":
                # Weekly: aggregate from daily data
                agg_df = (
                    df.resample("W")
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
            elif interval == "1M":
                # Monthly: aggregate from daily data to proper monthly candles
                agg_df = (
                    df.resample("ME")
                    .agg(
                        {  # Use 'ME' instead of deprecated 'M'
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
            else:
                # Use data as-is for other intervals (1h, 1d)
                agg_df = df

            # Convert back to expected format
            cached_data = []
            for date, row in agg_df.iterrows():
                cached_data.append(
                    {
                        "timestamp": int(date.timestamp() * 1000),
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                    }
                )

            session.close()
            return cached_data

        session.close()
        return []

    except Exception as e:
        import streamlit as st

        st.error(f"Error fetching extended data for {symbol}: {e}")
        return []


def show():
    """Main monitoring and alerts page."""
    st.title("🔔 Monitoring & Alerts")

    # Portfolio monitoring (get timeframe parameters)
    monitoring_interval, monitoring_days = show_portfolio_monitoring()

    # Monitoring parameter configuration (moved after timeframe)
    show_monitoring_parameters()

    show_legend()

    # Detailed asset information (using same timeframe as monitoring)
    show_detailed_asset_info(monitoring_interval, monitoring_days)

    # Alert settings (moved to bottom)
    st.markdown("---")
    show_alert_settings()

    # Active alerts management (moved to bottom)
    show_active_alerts()


def show_monitoring_parameters():
    """Show configurable monitoring parameters."""
    with st.expander("⚙️ Monitoring Parameters", expanded=False):
        st.subheader("📊 Indicator Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Volatility Settings**")
            st.session_state.settings["volatility_short_periods"] = st.number_input(
                "Short-term periods",
                min_value=5,
                max_value=50,
                value=st.session_state.settings.get("volatility_short_periods", 10),
                help="Number of periods for short-term volatility calculation",
            )
            st.session_state.settings["volatility_long_periods"] = st.number_input(
                "Long-term periods",
                min_value=10,
                max_value=100,
                value=st.session_state.settings.get("volatility_long_periods", 30),
                help="Number of periods for long-term volatility calculation",
            )
            st.session_state.settings["volatility_low_threshold"] = st.slider(
                "Low volatility threshold",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.settings.get("volatility_low_threshold", 0.7),
                step=0.05,
                help="Below this ratio = Low volatility",
            )
            st.session_state.settings["volatility_high_threshold"] = st.slider(
                "High volatility threshold",
                min_value=1.0,
                max_value=3.0,
                value=st.session_state.settings.get("volatility_high_threshold", 1.3),
                step=0.05,
                help="Above this ratio = High volatility",
            )

        with col2:
            st.write("**SMA Settings**")
            st.session_state.settings["sma_trend_periods"] = st.number_input(
                "SMA short periods",
                min_value=5,
                max_value=50,
                value=st.session_state.settings.get("sma_trend_periods", 20),
                help="Number of periods for short-term SMA calculation",
            )
            st.session_state.settings["sma_medium_periods"] = st.number_input(
                "SMA medium periods",
                min_value=30,
                max_value=100,
                value=st.session_state.settings.get("sma_medium_periods", 50),
                help="Number of periods for medium-term SMA calculation",
            )
            st.session_state.settings["sma_long_periods"] = st.number_input(
                "SMA long periods",
                min_value=100,
                max_value=300,
                value=st.session_state.settings.get("sma_long_periods", 200),
                help="Number of periods for long-term SMA calculation",
            )

            st.write("**EMA Settings**")
            st.session_state.settings["ema_trend_periods"] = st.number_input(
                "EMA short periods",
                min_value=5,
                max_value=50,
                value=st.session_state.settings.get("ema_trend_periods", 20),
                help="Number of periods for short-term EMA calculation",
            )
            st.session_state.settings["ema_medium_periods"] = st.number_input(
                "EMA medium periods",
                min_value=30,
                max_value=100,
                value=st.session_state.settings.get("ema_medium_periods", 50),
                help="Number of periods for medium-term EMA calculation",
            )

        with col3:
            st.write("**MACD Settings**")
            st.session_state.settings["macd_fast_periods"] = st.number_input(
                "MACD fast periods",
                min_value=5,
                max_value=20,
                value=st.session_state.settings.get("macd_fast_periods", 12),
                help="Number of periods for MACD fast moving average",
            )
            st.session_state.settings["macd_slow_periods"] = st.number_input(
                "MACD slow periods",
                min_value=15,
                max_value=35,
                value=st.session_state.settings.get("macd_slow_periods", 26),
                help="Number of periods for MACD slow moving average",
            )
            st.session_state.settings["macd_trend_periods"] = st.number_input(
                "MACD signal periods",
                min_value=5,
                max_value=20,
                value=st.session_state.settings.get("macd_trend_periods", 9),
                help="Number of periods for MACD signal line",
            )

            st.write("**RSI Settings**")
            st.session_state.settings["rsi_periods"] = st.number_input(
                "RSI periods",
                min_value=5,
                max_value=30,
                value=st.session_state.settings.get("rsi_periods", 14),
                help="Number of periods for RSI calculation",
            )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Monitoring Parameters"):
                # Save all settings to database
                from components.shared import save_all_settings

                save_all_settings()

                # Clear monitoring cache to force recalculation with new parameters
                if hasattr(st.session_state, "historical_data"):
                    # Clear all monitoring-related cache entries
                    keys_to_remove = [
                        key
                        for key in st.session_state.historical_data.keys()
                        if "_monitoring" in key
                    ]
                    for key in keys_to_remove:
                        del st.session_state.historical_data[key]

                    # Also clear timestamp cache entries
                    timestamp_keys_to_remove = [
                        key
                        for key in st.session_state.__dict__.keys()
                        if "_monitoring_timestamp" in key
                    ]
                    for key in timestamp_keys_to_remove:
                        delattr(st.session_state, key)

                st.success(
                    "Monitoring parameters saved! Refreshing with new parameters..."
                )
                st.rerun()

        with col2:
            if st.button("🔄 Reset to Defaults"):
                # Reset all monitoring parameters to default values
                default_params = {
                    "volatility_short_periods": 10,
                    "volatility_long_periods": 30,
                    "volatility_low_threshold": 0.7,
                    "volatility_high_threshold": 1.3,
                    "sma_trend_periods": 20,
                    "sma_medium_periods": 50,
                    "sma_long_periods": 200,
                    "ema_trend_periods": 20,
                    "ema_medium_periods": 50,
                    "macd_fast_periods": 12,
                    "macd_slow_periods": 26,
                    "macd_trend_periods": 9,
                    "rsi_periods": 14,
                }

                # Update session state with defaults
                for key, default_value in default_params.items():
                    st.session_state.settings[key] = default_value

                # Save defaults to database
                from components.shared import save_all_settings

                save_all_settings()

                # Clear monitoring cache to force recalculation with default parameters
                if hasattr(st.session_state, "historical_data"):
                    # Clear all monitoring-related cache entries
                    keys_to_remove = [
                        key
                        for key in st.session_state.historical_data.keys()
                        if "_monitoring" in key
                    ]
                    for key in keys_to_remove:
                        del st.session_state.historical_data[key]

                    # Also clear timestamp cache entries
                    timestamp_keys_to_remove = [
                        key
                        for key in st.session_state.__dict__.keys()
                        if "_monitoring_timestamp" in key
                    ]
                    for key in timestamp_keys_to_remove:
                        delattr(st.session_state, key)

                st.success(
                    "Parameters reset to defaults! Refreshing with default values..."
                )
                st.rerun()


def show_alert_settings():
    """Show alert configuration settings."""
    st.subheader("🚨 Alert Settings")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.session_state.settings["signal_change_alerts"] = st.checkbox(
            "Signal Change Alerts",
            value=st.session_state.settings.get("signal_change_alerts", True),
            help="Get notified when signals change (e.g., Neutral to Oversold)",
        )

    with col2:
        st.session_state.settings["desktop_notifications"] = st.checkbox(
            "Desktop Notifications",
            value=st.session_state.settings.get("desktop_notifications", False),
            help="Show system notifications for alerts",
        )

        if st.session_state.settings["desktop_notifications"]:
            if st.button("🧪 Test Desktop Notification"):
                send_desktop_notification(
                    "Test Alert", "Desktop notifications are working!"
                )

    with col3:
        st.session_state.settings["whatsapp_notifications"] = st.checkbox(
            "WhatsApp Notifications",
            value=st.session_state.settings.get("whatsapp_notifications", False),
            help="Send WhatsApp alerts (requires setup)",
        )


def show_active_alerts():
    """Show and manage active alerts."""
    st.subheader("🚨 Recent Alerts")

    # Alert controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Check All Signals"):
            check_all_signals()

    with col2:
        if st.button("✅ Mark All as Read"):
            for alert in st.session_state.alerts:
                alert["active"] = False
            st.success("All alerts marked as read")
            st.rerun()

    with col3:
        if st.button("🗑️ Clear All Alerts"):
            st.session_state.alerts = []
            st.success("All alerts cleared")
            st.rerun()

    # Display alerts
    active_alerts = [a for a in st.session_state.alerts if a.get("active", True)]

    if active_alerts:
        for i, alert in enumerate(active_alerts[:20]):  # Show last 20 alerts
            timestamp = alert.get("timestamp", datetime.now())
            symbol = alert.get("symbol", "")
            message = alert.get("message", "")
            severity = alert.get("severity", "LOW")
            alert_type = alert.get("type", "UNKNOWN")

            # Determine alert class
            if severity == "HIGH":
                alert_class = "alert-high"
            elif severity == "MEDIUM":
                alert_class = "alert-medium"
            else:
                alert_class = "alert-low"

            col1, col2 = st.columns([4, 1])

            with col1:
                # Add type emoji
                type_emoji = {
                    "PRICE_CHANGE": "💰",
                    "SIGNAL_CHANGE": "📊",
                    "TECHNICAL_SIGNAL": "📈",
                    "VOLATILITY_ALERT": "⚠️",
                }.get(alert_type, "🔔")

                st.markdown(
                    f"""
                <div class="{alert_class}">
                    <strong>{type_emoji} {symbol}</strong> - {message}<br>
                    <small>{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                if st.button("✅", key=f"dismiss_{i}"):
                    st.session_state.alerts[i]["active"] = False
                    st.rerun()
    else:
        st.info("No active alerts. Price changes and signals will appear here.")


def show_portfolio_monitoring():
    """Show enhanced portfolio monitoring with new signal types. Returns timeframe parameters."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Portfolio Monitoring")
    with col2:
        # Database-first monitoring - no API calls
        from components.shared import (
            poll_database_continuously,
            get_database_freshness_info,
        )

        poll_database_continuously()

        freshness_info = get_database_freshness_info()
        st.markdown(
            f"**Database**: {freshness_info['freshness_color']} {freshness_info['freshness_text']}"
        )

    # Universal timeframe selection with automatic recalculation
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_timeframe, interval, days = universal_timeframe_selector(
            "monitoring",
            default_index=2,  # Default to "1 day"
            help_text="Timeframe for technical indicator calculations and monitoring analysis",
        )

    with col2:
        # Display last update time
        display_last_update_time(help_text="Time when monitoring data was last updated")

    # Use the universal timeframe parameters from selector
    monitoring_interval = interval
    monitoring_days = days

    # Clear cache when timeframe changes to ensure fresh data
    current_timeframe_key = f"{monitoring_interval}_{monitoring_days}"
    if (
        not hasattr(st.session_state, "last_monitoring_timeframe")
        or st.session_state.last_monitoring_timeframe != current_timeframe_key
    ):
        # Timeframe changed - clear all monitoring cache
        if hasattr(st.session_state, "historical_data"):
            keys_to_remove = [
                key
                for key in st.session_state.historical_data.keys()
                if "_monitoring" in key
            ]
            for key in keys_to_remove:
                del st.session_state.historical_data[key]

        # Clear timestamp cache entries too
        timestamp_keys_to_remove = [
            key
            for key in st.session_state.__dict__.keys()
            if "_monitoring_timestamp" in key
        ]
        for key in timestamp_keys_to_remove:
            delattr(st.session_state, key)

        st.session_state.last_monitoring_timeframe = current_timeframe_key

    # Get portfolio assets for monitoring (exclude duplicates across portfolios)
    all_assets = st.session_state.portfolio_manager.get_all_assets()
    unique_symbols = {}

    # Create a dictionary with unique symbols (avoid duplicates across portfolios)
    for asset in all_assets:
        if asset.symbol not in ["USDT", "USDC", "BUSD"]:  # Exclude stablecoins
            if asset.symbol not in unique_symbols:
                unique_symbols[asset.symbol] = asset

    if not unique_symbols:
        show_empty_state(
            title="No Assets to Monitor",
            message="Add some cryptocurrencies (non-stablecoins) to your portfolio to start monitoring price movements and technical signals.",
            icon="🔔",
        )
        return monitoring_interval, monitoring_days

    monitoring_data = []

    # Pre-fetch all historical data with better caching
    symbols_list = list(unique_symbols.keys())
    historical_cache = {}

    with st.spinner("Fetching data for analysis..."):
        for symbol in symbols_list:
            # Use more aggressive caching for monitoring data with dynamic timeframes
            cache_key = f"{symbol}_{monitoring_interval}_{monitoring_days}_monitoring"
            cache_timestamp_key = (
                f"{symbol}_{monitoring_interval}_{monitoring_days}_monitoring_timestamp"
            )

            # Check if cache is fresh (within 30 seconds for responsive timeframe switching)
            current_time = datetime.now()
            if (
                cache_key in st.session_state.historical_data
                and cache_timestamp_key in st.session_state
                and (current_time - st.session_state[cache_timestamp_key]).seconds < 30
            ):  # 30 seconds cache
                historical_cache[symbol] = st.session_state.historical_data[cache_key]
            else:
                # Fetch data using EXACT same approach as Asset Charts with extended data
                recent_data = get_extended_historical_data(
                    symbol, monitoring_days, monitoring_interval
                )
                if recent_data:
                    historical_cache[symbol] = recent_data
                    st.session_state.historical_data[cache_key] = recent_data
                    st.session_state[cache_timestamp_key] = current_time

    # Debug info removed for production

    with st.spinner("Analyzing signals..."):
        for symbol, asset in unique_symbols.items():
            try:
                # Get current price
                current_price = st.session_state.current_prices.get(
                    symbol, asset.average_buy_price
                )

                # Get cached historical data
                recent_data = historical_cache.get(symbol, [])

                if (
                    recent_data and len(recent_data) >= 10
                ):  # Exact same requirement as Asset Charts
                    df = pd.DataFrame(recent_data)

                    # Calculate comprehensive analysis
                    analysis = analyze_asset_signals(df, symbol)

                    monitoring_data.append(
                        {
                            "Symbol": symbol,
                            "Current Price": current_price,
                            "RSI": analysis["rsi"],
                            "RSI_Periods": analysis["rsi_periods"],
                            "SMA_Short": analysis["sma_short"],
                            "SMA_Short_Periods": analysis["sma_short_periods"],
                            "SMA_Medium": analysis["sma_medium"],
                            "SMA_Medium_Periods": analysis["sma_medium_periods"],
                            "SMA_Long": analysis["sma_long"],
                            "SMA_Long_Periods": analysis["sma_long_periods"],
                            "EMA_Short": analysis["ema_short"],
                            "EMA_Short_Periods": analysis["ema_short_periods"],
                            "EMA_Medium": analysis["ema_medium"],
                            "EMA_Medium_Periods": analysis["ema_medium_periods"],
                            "Volatility_Ratio": analysis["volatility_ratio"],
                            "Volatility_Regime": analysis["volatility_regime"],
                            "Vol_Emoji": analysis["vol_emoji"],
                            "Vol_Short_Periods": analysis["vol_short_periods"],
                            "Vol_Long_Periods": analysis["vol_long_periods"],
                            "MACD": analysis["macd"],
                        }
                    )
                else:
                    # Fallback for insufficient data - use default settings
                    settings = st.session_state.settings
                    monitoring_data.append(
                        {
                            "Symbol": symbol,
                            "Current Price": current_price,
                            "RSI": np.nan,
                            "RSI_Periods": settings.get("rsi_periods", 14),
                            "SMA_Short": np.nan,
                            "SMA_Short_Periods": settings.get("sma_trend_periods", 20),
                            "SMA_Medium": np.nan,
                            "SMA_Medium_Periods": settings.get(
                                "sma_medium_periods", 50
                            ),
                            "SMA_Long": np.nan,
                            "SMA_Long_Periods": settings.get("sma_long_periods", 200),
                            "EMA_Short": np.nan,
                            "EMA_Short_Periods": settings.get("ema_trend_periods", 20),
                            "EMA_Medium": np.nan,
                            "EMA_Medium_Periods": settings.get(
                                "ema_medium_periods", 50
                            ),
                            "Volatility_Ratio": np.nan,
                            "Volatility_Regime": "N/A",
                            "Vol_Emoji": "⚫",
                            "Vol_Short_Periods": settings.get(
                                "volatility_short_periods", 10
                            ),
                            "Vol_Long_Periods": settings.get(
                                "volatility_long_periods", 30
                            ),
                            "MACD": np.nan,
                        }
                    )

            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")

    if monitoring_data:
        # Create enhanced monitoring table
        df_monitoring = pd.DataFrame(monitoring_data)

        # Display with proper color-coded table
        st.markdown("### 📈 Asset Monitoring Dashboard")

        # Create simple styled dataframe like Portfolio Assets table
        create_monitoring_dataframe(df_monitoring, monitoring_interval)

    else:
        show_empty_state(
            title="Analysis Failed",
            message="Unable to analyze any assets. This may be due to insufficient historical data or connectivity issues. Data is collected in the background.",
            icon="❌",
        )

    # Return timeframe parameters for use by detailed asset info
    return monitoring_interval, monitoring_days


def calculate_signal_summary(
    current_price: float,
    rsi: float,
    sma_short: float,
    sma_medium: float,
    sma_long: float,
) -> str:
    """Calculate signal summary based on price vs SMAs and RSI conditions."""
    try:
        # Check if we have valid data
        if np.isnan(current_price) or np.isnan(rsi):
            return "Neutral"

        # Check if price is below all SMAs and RSI < 30 (Buy signal)
        sma_values = [sma_short, sma_medium, sma_long]
        valid_smas = [sma for sma in sma_values if not np.isnan(sma)]

        if valid_smas and len(valid_smas) >= 2:  # At least 2 valid SMAs
            price_below_all_smas = all(current_price < sma for sma in valid_smas)
            if price_below_all_smas and rsi < 30:
                return "Buy"

            # Check if price is above all SMAs and RSI > 70 (Sell signal)
            price_above_all_smas = all(current_price > sma for sma in valid_smas)
            if price_above_all_smas and rsi > 70:
                return "Sell"

        return "Neutral"

    except Exception:
        return "Neutral"


def is_price_below_all_smas(row) -> bool:
    """Check if price is below all SMA columns in the row."""
    try:
        current_price = float(row["Price"].replace("$", "").replace(",", ""))
        sma_columns = [
            col for col in row.index if col.startswith("SMA(") and row[col] != "N/A"
        ]

        if len(sma_columns) < 2:  # Need at least 2 SMAs
            return False

        for col in sma_columns:
            sma_value = float(row[col].replace("$", "").replace(",", ""))
            if current_price >= sma_value:
                return False
        return True
    except:
        return False


def is_price_above_all_smas(row) -> bool:
    """Check if price is above all SMA columns in the row."""
    try:
        current_price = float(row["Price"].replace("$", "").replace(",", ""))
        sma_columns = [
            col for col in row.index if col.startswith("SMA(") and row[col] != "N/A"
        ]

        if len(sma_columns) < 2:  # Need at least 2 SMAs
            return False

        for col in sma_columns:
            sma_value = float(row[col].replace("$", "").replace(",", ""))
            if current_price <= sma_value:
                return False
        return True
    except:
        return False


def is_price_below_all_emas(row) -> bool:
    """Check if price is below all EMA columns in the row."""
    try:
        current_price = float(row["Price"].replace("$", "").replace(",", ""))
        ema_columns = [
            col for col in row.index if col.startswith("EMA(") and row[col] != "N/A"
        ]

        if len(ema_columns) < 2:  # Need at least 2 EMAs
            return False

        for col in ema_columns:
            ema_value = float(row[col].replace("$", "").replace(",", ""))
            if current_price >= ema_value:
                return False
        return True
    except:
        return False


def is_price_above_all_emas(row) -> bool:
    """Check if price is above all EMA columns in the row."""
    try:
        current_price = float(row["Price"].replace("$", "").replace(",", ""))
        ema_columns = [
            col for col in row.index if col.startswith("EMA(") and row[col] != "N/A"
        ]

        if len(ema_columns) < 2:  # Need at least 2 EMAs
            return False

        for col in ema_columns:
            ema_value = float(row[col].replace("$", "").replace(",", ""))
            if current_price <= ema_value:
                return False
        return True
    except:
        return False


def analyze_asset_signals(df: pd.DataFrame, symbol: str) -> Dict:
    """Comprehensive asset signal analysis with enhanced metrics."""
    try:
        # Convert timestamp if needed
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values if "volume" in df.columns else None

        # Get settings
        settings = st.session_state.settings

        # Calculate indicators with user-configurable parameters
        rsi_periods = settings.get("rsi_periods", 14)
        sma_trend_periods = settings.get("sma_trend_periods", 20)
        sma_medium_periods = settings.get("sma_medium_periods", 50)
        sma_long_periods = settings.get("sma_long_periods", 200)
        ema_trend_periods = settings.get("ema_trend_periods", 20)
        ema_medium_periods = settings.get("ema_medium_periods", 50)

        rsi = TechnicalIndicators.calculate_rsi(closes, rsi_periods)
        current_rsi = rsi[-1] if len(rsi) > 0 else np.nan

        # Calculate indicators directly on all available data using configurable periods (like Asset Charts)
        sma_20 = TechnicalIndicators.calculate_sma(closes, sma_trend_periods)
        sma_50 = TechnicalIndicators.calculate_sma(closes, sma_medium_periods)
        sma_200 = TechnicalIndicators.calculate_sma(closes, sma_long_periods)
        ema_21 = TechnicalIndicators.calculate_ema(closes, ema_trend_periods)
        ema_50 = TechnicalIndicators.calculate_ema(closes, ema_medium_periods)

        # Get current values
        current_price = closes[-1] if len(closes) > 0 else np.nan
        current_sma_20 = sma_20[-1] if len(sma_20) > 0 else np.nan
        current_sma_50 = sma_50[-1] if len(sma_50) > 0 else np.nan
        current_sma_200 = sma_200[-1] if len(sma_200) > 0 else np.nan
        current_ema_21 = ema_21[-1] if len(ema_21) > 0 else np.nan
        current_ema_50 = ema_50[-1] if len(ema_50) > 0 else np.nan

        # MACD analysis
        macd_fast_periods = settings.get("macd_fast_periods", 12)
        macd_slow_periods = settings.get("macd_slow_periods", 26)
        macd_signal_periods = settings.get("macd_trend_periods", 9)
        macd_data = TechnicalIndicators.calculate_macd(
            closes, macd_fast_periods, macd_slow_periods, macd_signal_periods
        )
        current_macd = macd_data["macd"][-1] if len(macd_data["macd"]) > 0 else np.nan

        # Volatility regime analysis
        vol_short = settings.get("volatility_short_periods", 10)
        vol_long = settings.get("volatility_long_periods", 30)
        vol_details = TechnicalIndicators.calculate_volatility_details(
            closes, vol_short, vol_long
        )
        vol_ratio = vol_details.get("volatility_ratio", np.nan)
        vol_regime, vol_color, vol_emoji = get_volatility_indicator(vol_ratio)

        # Determine signal and strength based on RSI and price vs SMAs
        signal = "NEUTRAL"
        strength = "N/A"

        # Check for oversold condition
        if current_rsi < 30 or (
            current_price < current_sma_20
            and current_price < current_sma_50
            and current_price < current_sma_200
        ):
            signal = "OVERSOLD"
            # Calculate strength based on how extreme the conditions are
            rsi_strength = (
                max(0, (30 - current_rsi) / 30 * 100) if current_rsi < 30 else 0
            )
            sma_strength = 0
            if not np.isnan(current_sma_20) and not np.isnan(current_price):
                if (
                    current_price < current_sma_20
                    and current_price < current_sma_50
                    and current_price < current_sma_200
                ):
                    sma_strength = 75  # Strong signal when below all SMAs
            strength = f"{max(rsi_strength, sma_strength):.0f}%"

        # Check for overbought condition
        elif current_rsi > 70 or (
            current_price > current_sma_20
            and current_price > current_sma_50
            and current_price > current_sma_200
        ):
            signal = "OVERBOUGHT"
            # Calculate strength based on how extreme the conditions are
            rsi_strength = (
                max(0, (current_rsi - 70) / 30 * 100) if current_rsi > 70 else 0
            )
            sma_strength = 0
            if not np.isnan(current_sma_20) and not np.isnan(current_price):
                if (
                    current_price > current_sma_20
                    and current_price > current_sma_50
                    and current_price > current_sma_200
                ):
                    sma_strength = 75  # Strong signal when above all SMAs
            strength = f"{max(rsi_strength, sma_strength):.0f}%"

        # Return comprehensive data for table display
        return {
            "current_price": current_price,
            "rsi": current_rsi,
            "rsi_periods": rsi_periods,
            "sma_short": current_sma_20,
            "sma_short_periods": sma_trend_periods,
            "sma_medium": current_sma_50,
            "sma_medium_periods": sma_medium_periods,
            "sma_long": current_sma_200,
            "sma_long_periods": sma_long_periods,
            "ema_short": current_ema_21,
            "ema_short_periods": ema_trend_periods,
            "ema_medium": current_ema_50,
            "ema_medium_periods": ema_medium_periods,
            "macd": current_macd,
            "volatility_ratio": vol_ratio,
            "volatility_regime": vol_regime,
            "vol_emoji": vol_emoji,
            "vol_short_periods": vol_short,
            "vol_long_periods": vol_long,
            "signal": signal,
            "strength": strength,
        }

    except Exception as e:
        st.error(f"Error in signal analysis for {symbol}: {e}")
        return {
            "current_price": np.nan,
            "rsi": np.nan,
            "rsi_periods": 14,
            "sma_short": np.nan,
            "sma_short_periods": 20,
            "sma_medium": np.nan,
            "sma_medium_periods": 50,
            "sma_long": np.nan,
            "sma_long_periods": 200,
            "ema_short": np.nan,
            "ema_short_periods": 20,
            "ema_medium": np.nan,
            "ema_medium_periods": 50,
            "macd": np.nan,
            "volatility_ratio": np.nan,
            "volatility_regime": "Error",
            "vol_emoji": "❌",
            "vol_short_periods": 10,
            "vol_long_periods": 30,
            "signal": "ERROR",
            "strength": "N/A",
        }


def create_monitoring_dataframe(df: pd.DataFrame, monitoring_interval: str = "1d"):
    """Create a simple styled dataframe like Portfolio Assets table."""

    # Get configurable parameters from settings
    vol_low_threshold = st.session_state.settings.get("volatility_low_threshold", 0.7)
    vol_high_threshold = st.session_state.settings.get("volatility_high_threshold", 1.3)
    rsi_period = st.session_state.settings.get("rsi_periods", 14)
    sma_trend_periods = st.session_state.settings.get("sma_trend_periods", 20)
    sma_medium_periods = st.session_state.settings.get("sma_medium_periods", 50)
    sma_long_periods = st.session_state.settings.get("sma_long_periods", 200)
    ema_trend_periods = st.session_state.settings.get("ema_trend_periods", 20)
    ema_medium_periods = st.session_state.settings.get("ema_medium_periods", 50)
    macd_fast_periods = st.session_state.settings.get("macd_fast_periods", 12)
    macd_slow_periods = st.session_state.settings.get("macd_slow_periods", 26)
    macd_signal_periods = st.session_state.settings.get("macd_trend_periods", 9)

    # Prepare display dataframe
    display_data = []

    for _, row in df.iterrows():
        symbol = row["Symbol"]
        current_price = row["Current Price"]
        rsi = row["RSI"]
        rsi_periods = row["RSI_Periods"]

        sma_short = row["SMA_Short"]
        sma_short_periods = row["SMA_Short_Periods"]
        sma_medium = row["SMA_Medium"]
        sma_medium_periods = row["SMA_Medium_Periods"]
        sma_long = row["SMA_Long"]
        sma_long_periods = row["SMA_Long_Periods"]

        ema_short = row["EMA_Short"]
        ema_short_periods = row["EMA_Short_Periods"]
        ema_medium = row["EMA_Medium"]
        ema_medium_periods = row["EMA_Medium_Periods"]

        vol_ratio = row["Volatility_Ratio"]
        vol_emoji = row["Vol_Emoji"]
        vol_regime = row["Volatility_Regime"]
        vol_short_periods = row["Vol_Short_Periods"]
        vol_long_periods = row["Vol_Long_Periods"]
        macd = row["MACD"]

        # Calculate signal for this row
        signal = calculate_signal_summary(
            current_price, rsi, sma_short, sma_medium, sma_long
        )

        display_data.append(
            {
                "Asset": symbol,
                "Price": (
                    format_currency(current_price, 4)
                    if not np.isnan(current_price)
                    else "N/A"
                ),
                f"RSI({rsi_periods})": f"{rsi:.1f}" if not np.isnan(rsi) else "N/A",
                f"SMA({sma_short_periods})": (
                    format_currency(sma_short, 4) if not np.isnan(sma_short) else "N/A"
                ),
                f"SMA({sma_medium_periods})": (
                    format_currency(sma_medium, 4)
                    if not np.isnan(sma_medium)
                    else "N/A"
                ),
                f"SMA({sma_long_periods})": (
                    format_currency(sma_long, 4) if not np.isnan(sma_long) else "N/A"
                ),
                f"EMA({ema_short_periods})": (
                    format_currency(ema_short, 4) if not np.isnan(ema_short) else "N/A"
                ),
                f"EMA({ema_medium_periods})": (
                    format_currency(ema_medium, 4)
                    if not np.isnan(ema_medium)
                    else "N/A"
                ),
                f"Vol({vol_short_periods},{vol_long_periods})": f"{vol_emoji} {vol_regime}",
                f"MACD({macd_fast_periods},{macd_slow_periods},{macd_signal_periods})": (
                    f"{macd:.6f}" if not np.isnan(macd) else "N/A"
                ),
                "Signal": signal,
            }
        )

    display_df = pd.DataFrame(display_data)

    # Create styling approach similar to Portfolio Assets table
    def highlight_cells(row):
        styles = pd.Series([""] * len(row), index=row.index)

        if row["Price"] != "N/A":
            try:
                current_price = float(row["Price"].replace("$", "").replace(",", ""))

                # Style RSI column (find RSI column dynamically) with bubble indicators
                for col_name in row.index:
                    if col_name.startswith("RSI(") and row[col_name] != "N/A":
                        try:
                            rsi_value = float(row[col_name])
                            if rsi_value >= 70:
                                styles[col_name] = (
                                    "color: #dc3545; background-color: rgba(220, 53, 69, 0.2); border-radius: 8px; padding: 2px 4px"  # Red bubble - Overbought
                                )
                            elif rsi_value <= 30:
                                styles[col_name] = (
                                    "color: #28a745; background-color: rgba(40, 167, 69, 0.2); border-radius: 8px; padding: 2px 4px"  # Green bubble - Oversold
                                )
                            else:
                                styles[col_name] = "color: #ffc107"  # Yellow - Neutral
                        except:
                            pass

                # Style SMA/EMA columns dynamically with bubble notifications
                for col_name in row.index:
                    if (
                        col_name.startswith("SMA(") or col_name.startswith("EMA(")
                    ) and row[col_name] != "N/A":
                        try:
                            indicator_value = float(
                                row[col_name].replace("$", "").replace(",", "")
                            )
                            if current_price < indicator_value:
                                # Check if this indicates oversold condition (price below all SMAs/EMAs)
                                if col_name.startswith(
                                    "SMA("
                                ) and is_price_below_all_smas(row):
                                    styles[col_name] = (
                                        "color: #28a745; background-color: rgba(40, 167, 69, 0.2); border-radius: 8px; padding: 2px 4px"  # Green bubble
                                    )
                                elif col_name.startswith(
                                    "EMA("
                                ) and is_price_below_all_emas(row):
                                    styles[col_name] = (
                                        "color: #28a745; background-color: rgba(40, 167, 69, 0.2); border-radius: 8px; padding: 2px 4px"  # Green bubble
                                    )
                                else:
                                    styles[col_name] = "color: #28a745"  # Green
                            else:
                                # Check if this indicates overbought condition (price above all SMAs/EMAs)
                                if col_name.startswith(
                                    "SMA("
                                ) and is_price_above_all_smas(row):
                                    styles[col_name] = (
                                        "color: #dc3545; background-color: rgba(220, 53, 69, 0.2); border-radius: 8px; padding: 2px 4px"  # Red bubble
                                    )
                                elif col_name.startswith(
                                    "EMA("
                                ) and is_price_above_all_emas(row):
                                    styles[col_name] = (
                                        "color: #dc3545; background-color: rgba(220, 53, 69, 0.2); border-radius: 8px; padding: 2px 4px"  # Red bubble
                                    )
                                else:
                                    styles[col_name] = "color: #dc3545"  # Red
                        except:
                            pass
            except:
                pass

        # Style volatility columns dynamically
        for col_name in row.index:
            if col_name.startswith("Vol("):
                if "Low" in row[col_name]:
                    styles[col_name] = "color: #28a745"  # Green
                elif "High" in row[col_name]:
                    styles[col_name] = "color: #dc3545"  # Red
                elif "Medium" in row[col_name]:
                    styles[col_name] = "color: #ffc107"  # Yellow

        # Style MACD columns dynamically
        for col_name in row.index:
            if col_name.startswith("MACD(") and row[col_name] != "N/A":
                try:
                    macd_value = float(row[col_name])
                    if macd_value > 0:
                        styles[col_name] = "color: #28a745"  # Green
                    else:
                        styles[col_name] = "color: #dc3545"  # Red
                except:
                    pass

        # Style Signal column
        if "Signal" in row.index:
            if row["Signal"] == "Buy":
                styles["Signal"] = "color: #28a745; font-weight: bold"  # Green
            elif row["Signal"] == "Sell":
                styles["Signal"] = "color: #dc3545; font-weight: bold"  # Red
            elif row["Signal"] == "Neutral":
                styles["Signal"] = "color: #ffc107; font-weight: bold"  # Yellow

        return styles

    # Apply styling
    styled_df = display_df.style.apply(highlight_cells, axis=1)

    # Display the dataframe
    st.dataframe(styled_df, width="stretch")


def show_legend():
    # Add simple explanation
    with st.expander("Legend", expanded=False):
        st.markdown(
            """
        **Color Legend:**
        - 🟢 **Green**: Price < Moving Averages, Low Volatility, MACD > 0, RSI ≤ 30 (Oversold)
        - 🔴 **Red**: Price > Moving Averages, High Volatility, MACD < 0, RSI ≥ 70 (Overbought)  
        - 🟡 **Yellow**: Medium Volatility, RSI 30-70 (Neutral)
        - ⚪ **White**: No data available
        
        **Bubble Indicators**: Highlight significant oversold (green bubble) / overbought (red bubble) conditions
        
        **Signal Summary**:
        - 🟢 **Buy**: Price below all SMAs AND RSI < 30 (Strong oversold)
        - 🔴 **Sell**: Price above all SMAs AND RSI > 70 (Strong overbought)
        - 🟡 **Neutral**: Mixed conditions or insufficient data
        
        **Note**: Column headers dynamically update based on your configured parameters in Monitoring Parameters.
        """
        )


def show_detailed_asset_info(monitoring_interval: str, monitoring_days: int):
    """Show detailed asset information using the same timeframe as main monitoring."""
    st.subheader("📋 Detailed Asset Information")

    with st.expander("🔍 Detailed Info", expanded=False):
        st.info(
            f"📊 Using same timeframe as monitoring dashboard: {monitoring_interval} interval, {monitoring_days} days"
        )
        st.write("Select assets to view detailed indicator values:")

        # Get unique symbols
        all_assets = st.session_state.portfolio_manager.get_all_assets()
        unique_symbols = list(
            set(
                asset.symbol
                for asset in all_assets
                if asset.symbol not in ["USDT", "USDC", "BUSD"]
            )
        )

        if not unique_symbols:
            st.info("No assets available for detailed analysis.")
            return

        selected_assets = st.multiselect(
            "Select assets for detailed analysis",
            unique_symbols,
            default=unique_symbols[:3] if len(unique_symbols) >= 3 else unique_symbols,
        )

        if selected_assets:
            detailed_data = []

            with st.spinner("Calculating detailed indicators..."):
                for symbol in selected_assets:
                    try:
                        # Get cached historical data using same timeframe as monitoring
                        cache_key = f"{symbol}_{monitoring_interval}_{monitoring_days}_monitoring"
                        cache_timestamp_key = f"{symbol}_{monitoring_interval}_{monitoring_days}_monitoring_timestamp"

                        current_time = datetime.now()
                        if (
                            cache_key in st.session_state.historical_data
                            and cache_timestamp_key in st.session_state
                            and (
                                current_time - st.session_state[cache_timestamp_key]
                            ).seconds
                            < 300
                        ):  # 5 minutes cache
                            recent_data = st.session_state.historical_data[cache_key]
                        else:
                            # Fetch fresh data using database-first approach (same as monitoring)
                            recent_data = get_historical_data_from_database(
                                symbol, monitoring_days, monitoring_interval
                            )
                            if recent_data:
                                st.session_state.historical_data[cache_key] = (
                                    recent_data
                                )
                                st.session_state[cache_timestamp_key] = current_time

                        if recent_data and len(recent_data) > 50:
                            df = pd.DataFrame(recent_data)
                            closes = df["close"].values

                            # Calculate all indicators using monitoring parameters
                            current_price = closes[-1]

                            # Use parameters from monitoring settings (configurable)
                            rsi_period = st.session_state.settings.get(
                                "rsi_periods", 14
                            )
                            sma_short_periods = st.session_state.settings.get(
                                "sma_trend_periods", 20
                            )
                            sma_medium_periods = st.session_state.settings.get(
                                "sma_medium_periods", 50
                            )
                            sma_long_periods = st.session_state.settings.get(
                                "sma_long_periods", 200
                            )
                            ema_short_periods = st.session_state.settings.get(
                                "ema_trend_periods", 20
                            )
                            ema_medium_periods = st.session_state.settings.get(
                                "ema_medium_periods", 50
                            )
                            macd_fast = st.session_state.settings.get(
                                "macd_fast_periods", 12
                            )
                            macd_slow = st.session_state.settings.get(
                                "macd_slow_periods", 26
                            )
                            macd_signal = st.session_state.settings.get(
                                "macd_trend_periods", 9
                            )
                            vol_short = st.session_state.settings.get(
                                "volatility_short_periods", 10
                            )
                            vol_long = st.session_state.settings.get(
                                "volatility_long_periods", 30
                            )

                            # Keep Bollinger Bands with fixed standard parameters
                            bb_period = 20  # Fixed
                            bb_std = 2.0  # Fixed

                            rsi = TechnicalIndicators.calculate_rsi(closes, rsi_period)

                            # Calculate SMAs using configurable parameters + SMA(100)
                            sma_short = TechnicalIndicators.calculate_sma(
                                closes, sma_short_periods
                            )
                            sma_medium = TechnicalIndicators.calculate_sma(
                                closes, sma_medium_periods
                            )
                            sma_100 = TechnicalIndicators.calculate_sma(
                                closes, 100
                            )  # Fixed SMA(100)
                            sma_long = TechnicalIndicators.calculate_sma(
                                closes, sma_long_periods
                            )

                            # Calculate EMAs using configurable parameters
                            ema_short = TechnicalIndicators.calculate_ema(
                                closes, ema_short_periods
                            )
                            ema_medium = TechnicalIndicators.calculate_ema(
                                closes, ema_medium_periods
                            )

                            macd_data = TechnicalIndicators.calculate_macd(
                                closes, macd_fast, macd_slow, macd_signal
                            )

                            bb_data = TechnicalIndicators.calculate_bollinger_bands(
                                closes, bb_period, bb_std
                            )

                            # Calculate volatility with user parameters
                            vol_details = {
                                "short_term_vol": (
                                    np.std(closes[-vol_short:])
                                    / np.mean(closes[-vol_short:])
                                    if len(closes) >= vol_short
                                    else np.nan
                                ),
                                "long_term_vol": (
                                    np.std(closes[-vol_long:])
                                    / np.mean(closes[-vol_long:])
                                    if len(closes) >= vol_long
                                    else np.nan
                                ),
                            }
                            vol_details["volatility_ratio"] = (
                                vol_details["short_term_vol"]
                                / vol_details["long_term_vol"]
                                if not np.isnan(vol_details["short_term_vol"])
                                and not np.isnan(vol_details["long_term_vol"])
                                and vol_details["long_term_vol"] != 0
                                else np.nan
                            )

                            detailed_data.append(
                                {
                                    "Symbol": symbol,
                                    "Current Price": format_currency(current_price, 4),
                                    f"RSI ({rsi_period})": (
                                        f"{rsi[-1]:.2f}" if len(rsi) > 0 else "N/A"
                                    ),
                                    f"SMA ({sma_short_periods})": (
                                        format_currency(sma_short[-1], 4)
                                        if len(sma_short) > 0
                                        else "N/A"
                                    ),
                                    f"SMA ({sma_medium_periods})": (
                                        format_currency(sma_medium[-1], 4)
                                        if len(sma_medium) > 0
                                        else "N/A"
                                    ),
                                    "SMA (100)": (
                                        format_currency(sma_100[-1], 4)
                                        if len(sma_100) > 0
                                        else "N/A"
                                    ),
                                    f"SMA ({sma_long_periods})": (
                                        format_currency(sma_long[-1], 4)
                                        if len(sma_long) > 0
                                        else "N/A"
                                    ),
                                    f"EMA ({ema_short_periods})": (
                                        format_currency(ema_short[-1], 4)
                                        if len(ema_short) > 0
                                        else "N/A"
                                    ),
                                    f"EMA ({ema_medium_periods})": (
                                        format_currency(ema_medium[-1], 4)
                                        if len(ema_medium) > 0
                                        else "N/A"
                                    ),
                                    f"MACD ({macd_fast},{macd_slow},{macd_signal})": (
                                        f"{macd_data['macd'][-1]:.6f}"
                                        if len(macd_data["macd"]) > 0
                                        else "N/A"
                                    ),
                                    "MACD Signal": (
                                        f"{macd_data['signal'][-1]:.6f}"
                                        if len(macd_data["signal"]) > 0
                                        else "N/A"
                                    ),
                                    f"BB Upper ({bb_period},{bb_std})": (
                                        format_currency(bb_data["upper"][-1], 4)
                                        if len(bb_data["upper"]) > 0
                                        else "N/A"
                                    ),
                                    f"BB Lower ({bb_period},{bb_std})": (
                                        format_currency(bb_data["lower"][-1], 4)
                                        if len(bb_data["lower"]) > 0
                                        else "N/A"
                                    ),
                                    "Volatility Ratio": (
                                        f"{vol_details['volatility_ratio']:.2f}"
                                        if not np.isnan(
                                            vol_details.get("volatility_ratio", np.nan)
                                        )
                                        else "N/A"
                                    ),
                                    f"Short Vol ({vol_short})": (
                                        f"{vol_details['short_term_vol']:.2%}"
                                        if not np.isnan(
                                            vol_details.get("short_term_vol", np.nan)
                                        )
                                        else "N/A"
                                    ),
                                    f"Long Vol ({vol_long})": (
                                        f"{vol_details['long_term_vol']:.2%}"
                                        if not np.isnan(
                                            vol_details.get("long_term_vol", np.nan)
                                        )
                                        else "N/A"
                                    ),
                                }
                            )

                    except Exception as e:
                        st.warning(f"Could not analyze {symbol}: {e}")

            if detailed_data:
                df_detailed = pd.DataFrame(detailed_data)
                st.dataframe(df_detailed, width="stretch")
            else:
                st.warning("No detailed data available for selected assets.")


def check_all_signals():
    """Check signals for all portfolio assets and generate alerts."""
    try:
        # Get all unique assets
        all_assets = st.session_state.portfolio_manager.get_all_assets()
        watchlist = st.session_state.portfolio_manager.get_watchlist()

        unique_symbols = set()
        for asset in all_assets:
            if asset.symbol not in ["USDT", "USDC", "BUSD"]:
                unique_symbols.add(asset.symbol)

        for item in watchlist:
            if item.symbol not in ["USDT", "USDC", "BUSD"]:
                unique_symbols.add(item.symbol)

        signal_count = 0

        with st.spinner("Checking signals for all assets..."):
            for symbol in unique_symbols:
                try:
                    # Get historical data using database cache first
                    historical_data = get_extended_historical_data(symbol, 7, "1h")

                    if historical_data and len(historical_data) > 50:
                        df = pd.DataFrame(historical_data)
                        analysis = analyze_asset_signals(df, symbol)

                        # Generate alerts for significant signals
                        signal = analysis["signal"]
                        strength = analysis["strength"]

                        if signal in ["OVERBOUGHT", "OVERSOLD"] and strength != "N/A":
                            strength_value = float(strength.replace("%", ""))
                            if strength_value >= 60:  # Only alert on strong signals
                                alert = {
                                    "timestamp": datetime.now(),
                                    "symbol": symbol,
                                    "type": "TECHNICAL_SIGNAL",
                                    "message": f"{signal} signal detected (Strength: {strength})",
                                    "active": True,
                                    "severity": (
                                        "HIGH" if strength_value >= 80 else "MEDIUM"
                                    ),
                                    "signal": signal,
                                    "strength": strength,
                                }

                                st.session_state.alerts.insert(0, alert)
                                signal_count += 1

                                # Send notifications
                                if st.session_state.settings.get(
                                    "desktop_notifications"
                                ):
                                    send_desktop_notification(
                                        f"Trading Signal: {symbol}",
                                        f"{signal} - {strength} strength",
                                    )

                                if st.session_state.settings.get(
                                    "whatsapp_notifications"
                                ):
                                    phone = st.session_state.settings.get(
                                        "whatsapp_number"
                                    )
                                    if phone:
                                        send_whatsapp_notification(
                                            f"🚨 {symbol} {signal} signal ({strength} strength)",
                                            phone,
                                        )

                        # Check for volatility alerts
                        vol_ratio = analysis.get("volatility_ratio", np.nan)
                        if (
                            not np.isnan(vol_ratio) and vol_ratio > 2.0
                        ):  # Very high volatility
                            alert = {
                                "timestamp": datetime.now(),
                                "symbol": symbol,
                                "type": "VOLATILITY_ALERT",
                                "message": f"High volatility detected (Ratio: {vol_ratio:.2f})",
                                "active": True,
                                "severity": "MEDIUM",
                                "volatility_ratio": vol_ratio,
                            }
                            st.session_state.alerts.insert(0, alert)
                            signal_count += 1

                except Exception as e:
                    st.warning(f"Error checking signals for {symbol}: {e}")

        # Keep only last 50 alerts
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts = st.session_state.alerts[:50]

        if signal_count > 0:
            st.success(f"Signal check complete. Found {signal_count} new alerts.")
        else:
            st.info("Signal check complete. No new significant signals found.")

    except Exception as e:
        st.error(f"Error during signal check: {e}")
