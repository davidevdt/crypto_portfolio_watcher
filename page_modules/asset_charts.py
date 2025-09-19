"""
Asset Charts Page - Advanced TradingView-style charting with technical indicators
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from components.shared import (
    get_signal_color_and_emoji,
    show_empty_state,
    show_empty_chart,
    show_insufficient_data_for_analysis,
)
from services.technical_indicators import TechnicalIndicators


def get_historical_data_from_db(
    symbol: str, days: int, interval: str = "1d"
) -> List[Dict]:
    """Get historical data from database cache for chart display (database-first approach)."""
    try:
        from database.models import get_session, HistoricalPrice
        from datetime import timedelta
        import pandas as pd

        session = get_session()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Handle 4h aggregation from hourly data
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

            session.close()

            if historical_prices:
                # Convert hourly data to DataFrame and aggregate to 4h periods
                hourly_data = []
                for price in historical_prices:
                    hourly_data.append(
                        {
                            "date": price.date,
                            "open": (
                                float(price.open_price)
                                if (
                                    hasattr(price, "open_price")
                                    and price.open_price is not None
                                )
                                else float(price.price)
                            ),
                            "high": (
                                float(price.high_price)
                                if (
                                    hasattr(price, "high_price")
                                    and price.high_price is not None
                                )
                                else float(price.price)
                            ),
                            "low": (
                                float(price.low_price)
                                if (
                                    hasattr(price, "low_price")
                                    and price.low_price is not None
                                )
                                else float(price.price)
                            ),
                            "close": (
                                float(price.close_price)
                                if (
                                    hasattr(price, "close_price")
                                    and price.close_price is not None
                                )
                                else float(price.price)
                            ),
                            "volume": float(price.volume) if price.volume else 0,
                        }
                    )

                df_hourly = pd.DataFrame(hourly_data)
                df_hourly.set_index("date", inplace=True)

                # Aggregate to 4h periods using proper OHLC logic:
                # Open = first hour's open, Close = last hour's close,
                # High = max high across 4 hours, Low = min low across 4 hours
                # Use 'epoch' origin to align with standard 4-hour boundaries (00:00, 04:00, 08:00, etc.)
                df_4h = (
                    df_hourly.resample("4h", origin="epoch")
                    .agg(
                        {
                            "open": "first",  # First hour's open price
                            "high": "max",  # Maximum high price across 4 hours
                            "low": "min",  # Minimum low price across 4 hours
                            "close": "last",  # Last hour's close price
                            "volume": "sum",  # Sum volume across 4 hours
                        }
                    )
                    .dropna()
                )

                # Sort by index to ensure chronological order (most recent data last)
                df_4h = df_4h.sort_index()

                # Convert back to expected format
                data = []
                for date, row in df_4h.iterrows():
                    data.append(
                        {
                            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"],
                            "timestamp": int(date.timestamp() * 1000),
                        }
                    )

                return data
            else:
                # Try to fetch some immediate data if none exists
                st.info(
                    f"No hourly data available for {symbol} 4-hour aggregation. Attempting immediate fetch..."
                )
                try:
                    from data_providers.data_fetcher import CryptoPriceFetcher
                    import asyncio

                    fetcher = CryptoPriceFetcher()
                    # Try to get some recent hourly data immediately
                    end_date = datetime.now()
                    start_date = end_date - timedelta(hours=24)
                    immediate_data = asyncio.run(
                        fetcher.get_historical_data(symbol, start_date, end_date, "1h")
                    )  # Last 24 hours
                    if immediate_data:
                        st.success(
                            f"Fetched {len(immediate_data)} hours of immediate data for {symbol}"
                        )
                        # Convert and return the immediate data
                        data = []
                        for item in immediate_data:
                            # Convert timestamp to date string if needed
                            if "timestamp" in item:
                                timestamp_ms = item["timestamp"]
                                date_obj = datetime.fromtimestamp(timestamp_ms / 1000)
                                date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                date_str = item.get("date", "")

                            data.append(
                                {
                                    "date": date_str,
                                    "open": item.get("open", item.get("close", 0)),
                                    "high": item.get("high", item.get("close", 0)),
                                    "low": item.get("low", item.get("close", 0)),
                                    "close": item.get("close", 0),
                                    "volume": item.get("volume", 0),
                                    "timestamp": item.get(
                                        "timestamp",
                                        int(timestamp_ms if "timestamp" in item else 0),
                                    ),
                                }
                            )
                        return data
                    else:
                        st.info(
                            "No immediate data available - background service will collect data automatically."
                        )
                except Exception as e:
                    st.info(
                        f"Could not fetch immediate data: {e}. Background service will collect data automatically."
                    )
                return []
        else:
            # For other intervals, query database directly
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
                # Convert to expected format using real OHLC data from database
                data = []

                for price in historical_prices:
                    # Use real OHLC data if available, with smart fallback handling
                    if (
                        hasattr(price, "open_price")
                        and price.open_price is not None
                        and hasattr(price, "high_price")
                        and price.high_price is not None
                        and hasattr(price, "low_price")
                        and price.low_price is not None
                    ):
                        # We have OHL data, now get close price
                        open_price = float(price.open_price)
                        high_price = float(price.high_price)
                        low_price = float(price.low_price)

                        # For close, prefer close_price but fallback to legacy price field
                        if (
                            hasattr(price, "close_price")
                            and price.close_price is not None
                        ):
                            close_price = float(price.close_price)
                        else:
                            close_price = float(
                                price.price
                            )  # Use legacy price field as close
                    else:
                        # Fallback for legacy data that only has single price point
                        close_price = float(price.price)
                        open_price = close_price
                        high_price = close_price
                        low_price = close_price

                    data.append(
                        {
                            "date": price.date.strftime("%Y-%m-%d %H:%M:%S"),
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": float(price.volume) if price.volume else 0.0,
                            "timestamp": int(price.date.timestamp() * 1000),
                        }
                    )

                # Check if volume data is missing and trigger re-fetch if needed
                missing_volume_count = sum(1 for item in data if item["volume"] == 0.0)
                if (
                    missing_volume_count > len(data) * 0.5
                ):  # If more than 50% of volume data is missing
                    st.warning(
                        f"⚠️ Volume data missing for {symbol}. Triggering background re-fetch..."
                    )
                    if (
                        hasattr(st.session_state, "background_service")
                        and st.session_state.background_service
                    ):
                        try:
                            # Trigger immediate data update for this symbol
                            asyncio.create_task(
                                st.session_state.background_service.update_historical_data(
                                    [symbol], []
                                )
                            )
                            st.info(f"🔄 Initiated volume data re-fetch for {symbol}")
                        except Exception as e:
                            st.warning(f"Could not trigger re-fetch: {e}")

                return data
            else:
                # Try to fetch some immediate data if none exists
                st.info(
                    f"No historical data available for {symbol}. Attempting immediate fetch..."
                )
                try:
                    from data_providers.data_fetcher import CryptoPriceFetcher
                    import asyncio

                    fetcher = CryptoPriceFetcher()
                    # Determine appropriate period based on interval
                    from datetime import timedelta

                    end_date = datetime.now()

                    if interval == "1h":
                        period_hours = min(days * 24, 168)  # Max 1 week of hourly data
                        start_date = end_date - timedelta(hours=period_hours)
                        immediate_data = asyncio.run(
                            fetcher.get_historical_data(
                                symbol, start_date, end_date, "1h"
                            )
                        )
                    elif interval == "1d":
                        period_days = min(days, 365)
                        start_date = end_date - timedelta(days=period_days)
                        immediate_data = asyncio.run(
                            fetcher.get_historical_data(
                                symbol, start_date, end_date, "1d"
                            )
                        )
                    else:
                        start_date = end_date - timedelta(days=days)
                        immediate_data = asyncio.run(
                            fetcher.get_historical_data(
                                symbol, start_date, end_date, interval
                            )
                        )

                    if immediate_data:
                        st.success(
                            f"Fetched {len(immediate_data)} data points for {symbol} ({interval})"
                        )
                        # Convert and return the immediate data
                        data = []
                        for item in immediate_data:
                            # Convert timestamp to date string if needed
                            if "timestamp" in item:
                                timestamp_ms = item["timestamp"]
                                date_obj = datetime.fromtimestamp(timestamp_ms / 1000)
                                date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                date_str = item.get("date", "")

                            data.append(
                                {
                                    "date": date_str,
                                    "open": item.get("open", item.get("close", 0)),
                                    "high": item.get("high", item.get("close", 0)),
                                    "low": item.get("low", item.get("close", 0)),
                                    "close": item.get("close", 0),
                                    "volume": item.get("volume", 0),
                                    "timestamp": item.get(
                                        "timestamp",
                                        int(timestamp_ms if "timestamp" in item else 0),
                                    ),
                                }
                            )
                        return data
                    else:
                        st.info(
                            "No immediate data available - background service will collect data automatically."
                        )
                except Exception as e:
                    st.info(
                        f"Could not fetch immediate data: {e}. Background service will collect data automatically."
                    )
                return []

    except Exception as e:
        st.error(f"Error loading historical data for {symbol} from database: {e}")
        return []


def fill_missing_dates(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Fill missing dates in OHLCV data using forward fill to prevent gaps in charts."""
    try:
        if len(df) == 0:
            return df

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            else:
                return df

        # Sort by date to ensure proper order
        df = df.sort_index()

        # Create complete date range from start to end with detected frequency
        start_date = df.index.min()
        end_date = df.index.max()

        # Auto-detect the data frequency from the existing data
        if len(df) >= 2:
            time_diff = df.index[1] - df.index[0]
            if time_diff <= pd.Timedelta(hours=1):
                freq = "h"  # Hourly data
            elif time_diff <= pd.Timedelta(hours=4):
                freq = "4h"  # 4-hourly data
            else:
                freq = "D"  # Daily data (default)
        else:
            freq = "D"  # Default to daily if we can't detect

        complete_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Reindex with complete date range and forward fill missing values
        df_filled = df.reindex(complete_dates)

        # Forward fill OHLCV data for missing dates
        df_filled["close"] = df_filled["close"].ffill()
        df_filled["open"] = df_filled["open"].fillna(df_filled["close"])
        df_filled["high"] = df_filled["high"].fillna(df_filled["close"])
        df_filled["low"] = df_filled["low"].fillna(df_filled["close"])
        df_filled["volume"] = df_filled["volume"].fillna(
            0
        )  # Zero volume for missing days

        # Fill any remaining NaN values at the beginning with backward fill
        df_filled = df_filled.bfill()

        filled_count = len(df_filled) - len(df)

        return df_filled

    except Exception as e:
        st.warning(f"Could not fill missing dates for {symbol}: {e}")
        return df


def aggregate_to_weekly_with_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Aggregate daily OHLCV data to weekly OHLCV data and recalculate indicators on weekly data."""
    try:
        if len(df) == 0:
            return df

        # Ensure we have the required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(
                f"Cannot aggregate {symbol} to weekly: missing columns {missing_columns}"
            )
            return df

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            else:
                st.error(f"Cannot aggregate {symbol}: no datetime index or date column")
                return df

        # Fill missing dates before aggregation to prevent gaps
        df = fill_missing_dates(df, symbol)

        # Aggregate OHLCV data to weekly (starting Monday)
        weekly_ohlcv = {}
        weekly_ohlcv["open"] = df["open"].resample("W-MON").first()
        weekly_ohlcv["high"] = df["high"].resample("W-MON").max()
        weekly_ohlcv["low"] = df["low"].resample("W-MON").min()
        weekly_ohlcv["close"] = df["close"].resample("W-MON").last()
        weekly_ohlcv["volume"] = df["volume"].resample("W-MON").sum()

        # Create DataFrame from aggregated OHLCV data
        weekly_df = pd.DataFrame(weekly_ohlcv)

        # Drop any rows with NaN values in OHLCV
        weekly_df = weekly_df.dropna(subset=["open", "high", "low", "close"])

        if len(weekly_df) == 0:
            st.warning(f"No valid weekly data after aggregation for {symbol}")
            return df

        # Recalculate ALL indicators on the weekly OHLCV data (this is key!)
        # Get user settings for indicator parameters or use weekly-appropriate defaults
        chart_settings = (
            getattr(st.session_state, "chart_settings", {})
            if hasattr(st, "session_state")
            else {}
        )

        if len(weekly_df) >= 3:  # Need minimum data for any indicators
            # Moving averages - use user periods directly on weekly data for full chart coverage
            user_sma_periods = chart_settings.get("sma_periods", [])
            user_ema_periods = chart_settings.get("ema_periods", [])

            # Use user periods directly (don't convert) to maintain same period references across timeframes
            # This ensures SMA 20 means 20 weekly candles, providing full chart coverage
            weekly_sma_periods = [
                p for p in user_sma_periods if p > 0 and p <= len(weekly_df)
            ]
            weekly_ema_periods = [
                p for p in user_ema_periods if p > 0 and p <= len(weekly_df)
            ]

            # Add default periods if no user periods specified
            if not weekly_sma_periods:
                weekly_sma_periods = [p for p in [20, 50] if p <= len(weekly_df)]
            if not weekly_ema_periods:
                weekly_ema_periods = [p for p in [12, 26] if p <= len(weekly_df)]

            # Calculate SMAs and EMAs using full periods for maximum chart coverage
            for period in weekly_sma_periods:
                weekly_df[f"SMA_{period}"] = TechnicalIndicators.calculate_sma(
                    weekly_df["close"].values, period
                )
            for period in weekly_ema_periods:
                weekly_df[f"EMA_{period}"] = TechnicalIndicators.calculate_ema(
                    weekly_df["close"].values, period
                )

            # RSI - Use user setting directly for consistency across timeframes
            rsi_period = chart_settings.get("rsi_period", 14)
            if len(weekly_df) >= rsi_period:
                weekly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    weekly_df["close"].values, rsi_period
                )
            elif len(weekly_df) >= 14:  # Standard fallback
                weekly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    weekly_df["close"].values, 14
                )
            elif len(weekly_df) >= 7:
                weekly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    weekly_df["close"].values, 7
                )

            # MACD - Use user settings directly for consistency across timeframes
            macd_fast = chart_settings.get("macd_fast", 12)
            macd_slow = chart_settings.get("macd_slow", 26)
            macd_signal = chart_settings.get("macd_signal", 9)

            if len(weekly_df) >= max(macd_fast, macd_slow, macd_signal):
                macd_data = TechnicalIndicators.calculate_macd(
                    weekly_df["close"].values, macd_fast, macd_slow, macd_signal
                )
                weekly_df["MACD"] = macd_data["macd"]
                weekly_df["MACD_Signal"] = macd_data["signal"]
                weekly_df["MACD_Histogram"] = macd_data["histogram"]
            elif len(weekly_df) >= 26:  # Standard fallback
                macd_data = TechnicalIndicators.calculate_macd(
                    weekly_df["close"].values, 12, 26, 9
                )
                weekly_df["MACD"] = macd_data["macd"]
                weekly_df["MACD_Signal"] = macd_data["signal"]
                weekly_df["MACD_Histogram"] = macd_data["histogram"]

            # Bollinger Bands - Use user settings directly for consistency
            bb_period = chart_settings.get("bb_period", 20)
            bb_std = chart_settings.get("bb_std", 2.0)
            if len(weekly_df) >= bb_period:
                bb_data = TechnicalIndicators.calculate_bollinger_bands(
                    weekly_df["close"].values, bb_period, bb_std
                )
                weekly_df["BB_Upper"] = bb_data["upper"]
                weekly_df["BB_Middle"] = bb_data["middle"]
                weekly_df["BB_Lower"] = bb_data["lower"]
            elif len(weekly_df) >= 20:  # Standard fallback
                bb_data = TechnicalIndicators.calculate_bollinger_bands(
                    weekly_df["close"].values, 20, bb_std
                )
                weekly_df["BB_Upper"] = bb_data["upper"]
                weekly_df["BB_Middle"] = bb_data["middle"]
                weekly_df["BB_Lower"] = bb_data["lower"]

            # Stochastic - Use user settings directly for consistency
            stoch_k = chart_settings.get("stoch_k", 14)
            stoch_d = chart_settings.get("stoch_d", 3)
            if len(weekly_df) >= max(stoch_k, stoch_d):
                stoch_k_values, stoch_d_values = (
                    TechnicalIndicators.calculate_stochastic(
                        weekly_df["high"].values,
                        weekly_df["low"].values,
                        weekly_df["close"].values,
                        stoch_k,
                        stoch_d,
                    )
                )
                weekly_df["Stoch_K"] = stoch_k_values
                weekly_df["Stoch_D"] = stoch_d_values
            elif len(weekly_df) >= 14:  # Standard fallback
                stoch_k_values, stoch_d_values = (
                    TechnicalIndicators.calculate_stochastic(
                        weekly_df["high"].values,
                        weekly_df["low"].values,
                        weekly_df["close"].values,
                        14,
                        3,
                    )
                )
                weekly_df["Stoch_K"] = stoch_k_values
                weekly_df["Stoch_D"] = stoch_d_values

            # Williams %R - Use user settings directly for consistency
            williams_period = chart_settings.get("williams_period", 14)
            if len(weekly_df) >= williams_period:
                weekly_df["Williams_R"] = TechnicalIndicators.calculate_williams_r(
                    weekly_df["high"].values,
                    weekly_df["low"].values,
                    weekly_df["close"].values,
                    williams_period,
                )
            elif len(weekly_df) >= 14:  # Standard fallback
                weekly_df["Williams_R"] = TechnicalIndicators.calculate_williams_r(
                    weekly_df["high"].values,
                    weekly_df["low"].values,
                    weekly_df["close"].values,
                    14,
                )

        # Successfully aggregated to weekly candles with recalculated indicators
        return weekly_df

    except Exception as e:
        st.error(f"Error aggregating weekly data with indicators for {symbol}: {e}")
        return df


def aggregate_to_monthly_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Aggregate daily OHLCV data to monthly OHLCV data for proper monthly visualization."""
    try:
        if len(df) == 0:
            return df

        # Ensure we have the required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(
                f"Cannot aggregate {symbol} to monthly: missing columns {missing_columns}"
            )
            return df

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                st.error(
                    f"Cannot convert {symbol} index to datetime for monthly aggregation"
                )
                return df

        # Create monthly aggregation - use proper resampling
        monthly_agg = {}
        monthly_agg["open"] = df["open"].resample("MS").first()  # First open of month
        monthly_agg["high"] = df["high"].resample("MS").max()  # Highest high of month
        monthly_agg["low"] = df["low"].resample("MS").min()  # Lowest low of month
        monthly_agg["close"] = df["close"].resample("MS").last()  # Last close of month
        monthly_agg["volume"] = (
            df["volume"].resample("MS").sum()
        )  # Total volume for month

        # Create DataFrame from aggregated data
        monthly_df = pd.DataFrame(monthly_agg)

        # Drop any rows with NaN values
        monthly_df = monthly_df.dropna()

        # Recalculate technical indicators on monthly data
        # Only calculate basic indicators that make sense on monthly timeframes
        if len(monthly_df) >= 20:  # Need sufficient data for indicators
            # Moving averages with adjusted periods for monthly data
            for period in [3, 6, 12]:  # 3, 6, 12 months instead of days
                if len(monthly_df) >= period:
                    monthly_df[f"SMA_{period}"] = TechnicalIndicators.calculate_sma(
                        monthly_df["close"].values, period
                    )
                    monthly_df[f"EMA_{period}"] = TechnicalIndicators.calculate_ema(
                        monthly_df["close"].values, period
                    )

            # RSI - adjust period for monthly data since we have fewer data points
            if len(monthly_df) >= 14:
                monthly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    monthly_df["close"].values, 14
                )
            elif len(monthly_df) >= 7:  # Use shorter period for limited monthly data
                monthly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    monthly_df["close"].values, min(7, len(monthly_df) - 1)
                )
            elif len(monthly_df) >= 3:  # Minimum viable RSI period
                monthly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    monthly_df["close"].values, min(3, len(monthly_df) - 1)
                )

            # Bollinger Bands
            if len(monthly_df) >= 20:
                bb_data = TechnicalIndicators.calculate_bollinger_bands(
                    monthly_df["close"].values, 20, 2.0
                )
                monthly_df["BB_Upper"] = bb_data["upper"]
                monthly_df["BB_Middle"] = bb_data["middle"]
                monthly_df["BB_Lower"] = bb_data["lower"]

            # MACD
            if len(monthly_df) >= 26:
                macd_data = TechnicalIndicators.calculate_macd(
                    monthly_df["close"].values, 12, 26, 9
                )
                monthly_df["MACD"] = macd_data["macd"]
                monthly_df["MACD_Signal"] = macd_data["signal"]
                monthly_df["MACD_Histogram"] = macd_data["histogram"]

        # Remove rows with all NaN values
        monthly_df = monthly_df.dropna(how="all")

        if len(monthly_df) == 0:
            st.warning(f"No valid monthly data after aggregation for {symbol}")
            return df  # Return original daily data as fallback

        # Successfully aggregated to monthly candles
        return monthly_df

    except Exception as e:
        st.error(f"Error aggregating monthly data for {symbol}: {e}")
        return df  # Return original data as fallback


def aggregate_to_monthly_with_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Aggregate daily data to monthly OHLCV and recalculate indicators on monthly data."""
    try:
        if len(df) == 0:
            return df

        # Ensure we have the required OHLCV columns
        required_ohlcv = ["open", "high", "low", "close", "volume"]
        missing_ohlcv = [col for col in required_ohlcv if col not in df.columns]
        if missing_ohlcv:
            st.warning(
                f"Cannot aggregate {symbol} to monthly: missing OHLCV columns {missing_ohlcv}"
            )
            return df

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                st.error(
                    f"Cannot convert {symbol} index to datetime for monthly aggregation"
                )
                return df

        # Fill missing dates before aggregation to prevent gaps
        df = fill_missing_dates(df, symbol)

        # Aggregate OHLCV data to monthly
        monthly_ohlcv = {}
        monthly_ohlcv["open"] = df["open"].resample("MS").first()
        monthly_ohlcv["high"] = df["high"].resample("MS").max()
        monthly_ohlcv["low"] = df["low"].resample("MS").min()
        monthly_ohlcv["close"] = df["close"].resample("MS").last()
        monthly_ohlcv["volume"] = df["volume"].resample("MS").sum()

        # Create DataFrame from aggregated OHLCV data
        monthly_df = pd.DataFrame(monthly_ohlcv)

        # Drop any rows with NaN values in OHLCV
        monthly_df = monthly_df.dropna(subset=["open", "high", "low", "close"])

        if len(monthly_df) == 0:
            st.warning(f"No valid monthly data after aggregation for {symbol}")
            return df

        # Recalculate ALL indicators on the monthly OHLCV data (this is key!)
        # Get user settings for indicator parameters or use monthly-appropriate defaults
        chart_settings = (
            getattr(st.session_state, "chart_settings", {})
            if hasattr(st, "session_state")
            else {}
        )

        if len(monthly_df) >= 3:  # Need minimum data for any indicators
            # Moving averages - use user periods directly on monthly data for full chart coverage
            user_sma_periods = chart_settings.get("sma_periods", [])
            user_ema_periods = chart_settings.get("ema_periods", [])

            # Use user periods directly (don't convert) to maintain same period references across timeframes
            # This ensures SMA 20 means 20 monthly candles, providing full chart coverage when data allows
            monthly_sma_periods = [
                p for p in user_sma_periods if p > 0 and p <= len(monthly_df)
            ]
            monthly_ema_periods = [
                p for p in user_ema_periods if p > 0 and p <= len(monthly_df)
            ]

            # Add default periods if no user periods specified or fit within available data
            if not monthly_sma_periods:
                monthly_sma_periods = [p for p in [12, 24] if p <= len(monthly_df)]
                # If still no periods fit, use smaller defaults
                if not monthly_sma_periods:
                    monthly_sma_periods = [p for p in [3, 6] if p <= len(monthly_df)]
            if not monthly_ema_periods:
                monthly_ema_periods = [p for p in [12, 26] if p <= len(monthly_df)]
                # If still no periods fit, use smaller defaults
                if not monthly_ema_periods:
                    monthly_ema_periods = [p for p in [6, 12] if p <= len(monthly_df)]

            # Calculate SMAs and EMAs using full periods for maximum chart coverage
            for period in monthly_sma_periods:
                monthly_df[f"SMA_{period}"] = TechnicalIndicators.calculate_sma(
                    monthly_df["close"].values, period
                )
            for period in monthly_ema_periods:
                monthly_df[f"EMA_{period}"] = TechnicalIndicators.calculate_ema(
                    monthly_df["close"].values, period
                )

            # RSI - Use user setting directly for consistency across timeframes
            rsi_period = chart_settings.get("rsi_period", 14)
            if len(monthly_df) >= rsi_period:
                monthly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    monthly_df["close"].values, rsi_period
                )
            elif len(monthly_df) >= 14:  # Standard fallback
                monthly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    monthly_df["close"].values, 14
                )
            elif len(monthly_df) >= 6:
                monthly_df["RSI"] = TechnicalIndicators.calculate_rsi(
                    monthly_df["close"].values, 6
                )

            # MACD - Use user settings directly for consistency across timeframes
            macd_fast = chart_settings.get("macd_fast", 12)
            macd_slow = chart_settings.get("macd_slow", 26)
            macd_signal = chart_settings.get("macd_signal", 9)

            if len(monthly_df) >= max(macd_fast, macd_slow, macd_signal):
                macd_data = TechnicalIndicators.calculate_macd(
                    monthly_df["close"].values, macd_fast, macd_slow, macd_signal
                )
                monthly_df["MACD"] = macd_data["macd"]
                monthly_df["MACD_Signal"] = macd_data["signal"]
                monthly_df["MACD_Histogram"] = macd_data["histogram"]
            elif len(monthly_df) >= 26:  # Standard fallback
                macd_data = TechnicalIndicators.calculate_macd(
                    monthly_df["close"].values, 12, 26, 9
                )
                monthly_df["MACD"] = macd_data["macd"]
                monthly_df["MACD_Signal"] = macd_data["signal"]
                monthly_df["MACD_Histogram"] = macd_data["histogram"]

            # Bollinger Bands - Use user settings directly for consistency
            bb_period = chart_settings.get("bb_period", 20)
            bb_std = chart_settings.get("bb_std", 2.0)
            if len(monthly_df) >= bb_period:
                bb_data = TechnicalIndicators.calculate_bollinger_bands(
                    monthly_df["close"].values, bb_period, bb_std
                )
                monthly_df["BB_Upper"] = bb_data["upper"]
                monthly_df["BB_Middle"] = bb_data["middle"]
                monthly_df["BB_Lower"] = bb_data["lower"]
            elif len(monthly_df) >= 20:  # Standard fallback
                bb_data = TechnicalIndicators.calculate_bollinger_bands(
                    monthly_df["close"].values, 20, bb_std
                )
                monthly_df["BB_Upper"] = bb_data["upper"]
                monthly_df["BB_Middle"] = bb_data["middle"]
                monthly_df["BB_Lower"] = bb_data["lower"]

            # Stochastic - Use user settings directly for consistency
            stoch_k = chart_settings.get("stoch_k", 14)
            stoch_d = chart_settings.get("stoch_d", 3)
            if len(monthly_df) >= max(stoch_k, stoch_d):
                stoch_k_values, stoch_d_values = (
                    TechnicalIndicators.calculate_stochastic(
                        monthly_df["high"].values,
                        monthly_df["low"].values,
                        monthly_df["close"].values,
                        stoch_k,
                        stoch_d,
                    )
                )
                monthly_df["Stoch_K"] = stoch_k_values
                monthly_df["Stoch_D"] = stoch_d_values
            elif len(monthly_df) >= 14:  # Standard fallback
                stoch_k_values, stoch_d_values = (
                    TechnicalIndicators.calculate_stochastic(
                        monthly_df["high"].values,
                        monthly_df["low"].values,
                        monthly_df["close"].values,
                        14,
                        3,
                    )
                )
                monthly_df["Stoch_K"] = stoch_k_values
                monthly_df["Stoch_D"] = stoch_d_values

            # Williams %R - Use user settings directly for consistency
            williams_period = chart_settings.get("williams_period", 14)
            if len(monthly_df) >= williams_period:
                monthly_df["Williams_R"] = TechnicalIndicators.calculate_williams_r(
                    monthly_df["high"].values,
                    monthly_df["low"].values,
                    monthly_df["close"].values,
                    williams_period,
                )
            elif len(monthly_df) >= 14:  # Standard fallback
                monthly_df["Williams_R"] = TechnicalIndicators.calculate_williams_r(
                    monthly_df["high"].values,
                    monthly_df["low"].values,
                    monthly_df["close"].values,
                    14,
                )

        # Successfully aggregated to monthly candles with recalculated indicators
        return monthly_df

    except Exception as e:
        st.error(f"Error aggregating monthly data with indicators for {symbol}: {e}")
        return df


def apply_indicators_to_subset(
    subset_df: pd.DataFrame, full_df_with_indicators: pd.DataFrame
) -> pd.DataFrame:
    """Apply indicators calculated on full dataset to subset data by matching dates."""
    try:
        # Start with the subset OHLCV data
        result_df = subset_df.copy()

        # Get all indicator columns (non-OHLCV columns)
        base_columns = {"open", "high", "low", "close", "volume", "date"}
        indicator_columns = [
            col for col in full_df_with_indicators.columns if col not in base_columns
        ]

        # For each indicator column, copy values from full dataset where dates match
        for col in indicator_columns:
            if col in full_df_with_indicators.columns:
                # Use reindex to align indicator values with subset dates
                result_df[col] = full_df_with_indicators[col].reindex(subset_df.index)

        return result_df

    except Exception as e:
        st.error(f"Error applying indicators to subset: {e}")
        return subset_df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators on the full dataset for maximum accuracy using user settings."""
    df_copy = df.copy()

    try:
        # Get user settings for indicator parameters or use defaults
        chart_settings = getattr(st.session_state, "chart_settings", {})

        # Moving averages - use user-configured periods or defaults
        default_sma_periods = [5, 10, 20, 50, 100, 150, 200]
        default_ema_periods = [9, 12, 21, 26, 50, 100, 200]

        user_sma_periods = chart_settings.get("sma_periods", [])
        user_ema_periods = chart_settings.get("ema_periods", [])

        # Combine user periods with defaults, removing duplicates
        sma_periods = list(set(default_sma_periods + user_sma_periods))
        ema_periods = list(set(default_ema_periods + user_ema_periods))

        # Calculate SMAs
        for period in sma_periods:
            if period > 0 and period <= len(df_copy):  # Ensure valid period
                df_copy[f"SMA_{period}"] = TechnicalIndicators.calculate_sma(
                    df_copy["close"].values, period
                )

        # Calculate EMAs
        for period in ema_periods:
            if period > 0 and period <= len(df_copy):  # Ensure valid period
                df_copy[f"EMA_{period}"] = TechnicalIndicators.calculate_ema(
                    df_copy["close"].values, period
                )

        # Bollinger Bands - use user settings or defaults
        bb_period = chart_settings.get("bb_period", 20)
        bb_std = chart_settings.get("bb_std", 2.0)
        if bb_period > 0 and bb_period <= len(df_copy):
            bb_data = TechnicalIndicators.calculate_bollinger_bands(
                df_copy["close"].values, bb_period, bb_std
            )
            df_copy["BB_Upper"] = bb_data["upper"]
            df_copy["BB_Middle"] = bb_data["middle"]
            df_copy["BB_Lower"] = bb_data["lower"]

        # RSI - use user settings or default
        rsi_period = chart_settings.get("rsi_period", 14)
        if rsi_period > 0 and rsi_period <= len(df_copy):
            df_copy["RSI"] = TechnicalIndicators.calculate_rsi(
                df_copy["close"].values, rsi_period
            )

        # MACD - use user settings or defaults
        macd_fast = chart_settings.get("macd_fast", 12)
        macd_slow = chart_settings.get("macd_slow", 26)
        macd_signal = chart_settings.get("macd_signal", 9)
        if (
            macd_fast > 0
            and macd_slow > 0
            and macd_signal > 0
            and max(macd_fast, macd_slow, macd_signal) <= len(df_copy)
        ):
            macd_data = TechnicalIndicators.calculate_macd(
                df_copy["close"].values, macd_fast, macd_slow, macd_signal
            )
            df_copy["MACD"] = macd_data["macd"]
            df_copy["MACD_Signal"] = macd_data["signal"]
            df_copy["MACD_Histogram"] = macd_data["histogram"]

        # Stochastic - use user settings or defaults
        stoch_k = chart_settings.get("stoch_k", 14)
        stoch_d = chart_settings.get("stoch_d", 3)
        if stoch_k > 0 and stoch_d > 0 and max(stoch_k, stoch_d) <= len(df_copy):
            stoch_k_values, stoch_d_values = TechnicalIndicators.calculate_stochastic(
                df_copy["high"].values,
                df_copy["low"].values,
                df_copy["close"].values,
                stoch_k,
                stoch_d,
            )
            df_copy["Stoch_K"] = stoch_k_values
            df_copy["Stoch_D"] = stoch_d_values

        # Williams %R - use user settings or default
        williams_period = chart_settings.get("williams_period", 14)
        if williams_period > 0 and williams_period <= len(df_copy):
            df_copy["Williams_R"] = TechnicalIndicators.calculate_williams_r(
                df_copy["high"].values,
                df_copy["low"].values,
                df_copy["close"].values,
                williams_period,
            )

        # Volume MA for volume trend if enabled
        if chart_settings.get("show_volume_trend", False):
            volume_ma_period = chart_settings.get("volume_ma_period", 20)
            if (
                volume_ma_period > 0
                and volume_ma_period <= len(df_copy)
                and "volume" in df_copy.columns
            ):
                df_copy[f"Volume_MA_{volume_ma_period}"] = (
                    TechnicalIndicators.calculate_sma(
                        df_copy["volume"].values, volume_ma_period
                    )
                )

    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

    return df_copy


def show():
    """Main asset charts page."""
    st.title("📈 Asset Charts")

    # Get all available assets
    assets = st.session_state.portfolio_manager.get_all_assets()
    watchlist = st.session_state.portfolio_manager.get_watchlist()

    # Combine portfolio assets and watchlist, exclude stablecoins
    stablecoins = ["USDT", "USDC", "BUSD", "DAI"]
    all_symbols = []

    for asset in assets:
        if asset.symbol not in stablecoins:
            all_symbols.append(asset.symbol)

    for item in watchlist:
        if item.symbol not in stablecoins and item.symbol not in all_symbols:
            all_symbols.append(item.symbol)

    if not all_symbols:
        show_empty_state(
            title="No Assets for Charting",
            message="Add some cryptocurrencies to your portfolio or watchlist to view detailed price charts and technical analysis. Stablecoins are excluded from charting.",
            icon="📈",
        )
        return

    # Asset and timeframe selection
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_symbol = st.selectbox("📊 Select Asset", all_symbols)

    with col2:
        # Separate timeframe and period selection
        timeframes = ["1 hour", "4 hours", "1 day", "1 week", "1 month"]
        selected_timeframe = st.selectbox(
            "⏰ Timeframe", timeframes, index=2
        )  # Default to 1 day

    with col3:
        # Period options based on selected timeframe - Following instructions for max data per timeframe:
        # 1h: 30 days max, 4h: 90 days max, 1d/1w/1M: 5 years max
        period_options = {
            "1 hour": {
                "1 day": 1,
                "3 days": 3,
                "7 days": 7,
                "14 days": 14,
                "30 days": 30,
            },
            "4 hours": {
                "3 days": 3,
                "7 days": 7,
                "14 days": 14,
                "30 days": 30,
                "60 days": 60,
                "90 days": 90,
            },
            "1 day": {
                "7 days": 7,
                "30 days": 30,
                "90 days": 90,
                "6 months": 180,
                "1 year": 365,
                "3 years": 1095,
                "5 years": 1825,
            },
            "1 week": {
                "3 months": 90,
                "6 months": 180,
                "1 year": 365,
                "2 years": 730,
                "3 years": 1095,
                "5 years": 1825,
            },
            "1 month": {
                "6 months": 180,
                "1 year": 365,
                "2 years": 730,
                "3 years": 1095,
                "5 years": 1825,
            },
        }

        available_periods = list(period_options[selected_timeframe].keys())
        default_periods = {
            "1 hour": "7 days",
            "4 hours": "30 days",
            "1 day": "90 days",
            "1 week": "1 year",
            "1 month": "2 years",
        }

        default_index = (
            available_periods.index(default_periods[selected_timeframe])
            if default_periods[selected_timeframe] in available_periods
            else 0
        )
        selected_period = st.selectbox(
            "📊 Period", available_periods, index=default_index
        )

        # Show explanation of the combination
        explanation = {
            "1 hour": f"Shows {selected_period} of hourly candles",
            "4 hours": f"Shows {selected_period} of 4-hour candles",
            "1 day": f"Shows {selected_period} of daily candles",
            "1 week": f"Shows {selected_period} of weekly candles",
            "1 month": f"Shows {selected_period} of monthly candles",
        }

        st.caption(f"💡 {explanation[selected_timeframe]}")

    # Convert to interval format with proper mapping for each timeframe
    # For weekly and monthly, we use daily data and aggregate it
    interval_map = {
        "1 hour": "1h",
        "4 hours": "4h",
        "1 day": "1d",
        "1 week": "1d",  # Use daily data for weekly aggregation
        "1 month": "1d",  # Use daily data for monthly aggregation
    }

    # Each timeframe now uses its native interval from the exchange API
    # This provides the most accurate candle data for each timeframe

    # Calculate the settings for data fetching
    # Always use the mapped interval for the selected timeframe
    tf_settings = {
        "days": period_options[selected_timeframe][selected_period],
        "interval": interval_map[selected_timeframe],
    }

    # Configuration info removed as requested

    # Create cache key from timeframe and period
    cache_key = f"{selected_symbol}_{selected_timeframe}_{selected_period}"

    # Refresh button
    if st.button("🔄 Refresh Chart Data"):
        # Clear all cached data for this symbol to force refresh across all timeframes
        keys_to_remove = [
            k
            for k in st.session_state.historical_data.keys()
            if k.startswith(selected_symbol)
        ]
        for key in keys_to_remove:
            del st.session_state.historical_data[key]
        st.success(f"Cleared cache for {selected_symbol} - all timeframes will reload")

    # Implement the strategy from instructions:
    # Get historical data for the maximum possible period per timeframe,
    # then calculate indicators on that maximum period,
    # then display subset based on selected visualization period

    # Define maximum data periods per timeframe (from instructions)
    # These limits ensure we don't fetch excessive data while having enough for indicators
    max_data_periods = {
        "1 hour": 30,  # 30 days max for 1h (720 hourly points)
        "4 hours": 90,  # 90 days max for 4h (540 4-hour points)
        "1 day": 1825,  # 5 years max for 1d (1825 daily points)
        "1 week": 1825,  # 5 years max for 1w (need daily data to aggregate)
        "1 month": 1825,  # 5 years max for 1M (need daily data to aggregate)
    }

    # Validate the combination
    max_allowed = max_data_periods[selected_timeframe]
    if tf_settings["days"] > max_allowed:
        st.warning(
            f"⚠️ Period {selected_period} exceeds maximum for {selected_timeframe} ({max_allowed} days). This combination may not work properly."
        )
        st.info(
            f"💡 Try selecting a shorter period for {selected_timeframe} timeframe."
        )

    # Get maximum data for indicator calculation
    max_days_for_timeframe = max_data_periods[selected_timeframe]
    base_days = tf_settings["days"]  # Days for visualization

    # Use maximum available data for this timeframe for indicator calculation
    extended_days = max_days_for_timeframe

    # Include indicator settings in cache key to force recalculation when parameters change
    chart_settings = getattr(st.session_state, "chart_settings", {})
    settings_hash = (
        str(hash(str(sorted(chart_settings.items())))) if chart_settings else "default"
    )
    extended_cache_key = f"{selected_symbol}_{selected_timeframe}_{tf_settings['interval']}_{extended_days}_max_data_{settings_hash}"

    # Fetch extended historical data from database for proper indicator calculation
    if extended_cache_key not in st.session_state.historical_data:
        with st.spinner(
            f"Loading {selected_symbol} data: {extended_days} days of {tf_settings['interval']} data for {selected_timeframe} analysis..."
        ):
            historical_data = get_historical_data_from_db(
                selected_symbol, extended_days, tf_settings["interval"]
            )
            st.session_state.historical_data[extended_cache_key] = historical_data

    historical_data = st.session_state.historical_data.get(extended_cache_key, [])

    if not historical_data:
        st.info(
            f"No data available for {selected_symbol}. Attempting immediate data fetch..."
        )
        # Try to fetch immediate data from exchange
        try:
            from data_providers.data_fetcher import CryptoPriceFetcher
            import asyncio

            fetcher = CryptoPriceFetcher()

            # Determine appropriate period based on timeframe
            from datetime import timedelta

            end_date = datetime.now()

            if tf_settings["interval"] == "1h":
                fetch_period = min(extended_days * 24, 168)  # Max 1 week of hourly data
                start_date = end_date - timedelta(hours=fetch_period)
                immediate_data = asyncio.run(
                    fetcher.get_historical_data(
                        selected_symbol, start_date, end_date, "1h"
                    )
                )
            elif tf_settings["interval"] == "1d":
                period_days = min(extended_days, 365)
                start_date = end_date - timedelta(days=period_days)
                immediate_data = asyncio.run(
                    fetcher.get_historical_data(
                        selected_symbol, start_date, end_date, "1d"
                    )
                )
            else:
                start_date = end_date - timedelta(days=extended_days)
                immediate_data = asyncio.run(
                    fetcher.get_historical_data(
                        selected_symbol, start_date, end_date, tf_settings["interval"]
                    )
                )

            if immediate_data:
                st.success(
                    f"✅ Fetched {len(immediate_data)} data points for {selected_symbol}"
                )
                # Convert to expected format
                converted_data = []
                for item in immediate_data:
                    # Convert timestamp to date string if needed
                    if "timestamp" in item:
                        timestamp_ms = item["timestamp"]
                        date_obj = datetime.fromtimestamp(timestamp_ms / 1000)
                        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        date_str = item.get("date", "")

                    converted_data.append(
                        {
                            "date": date_str,
                            "open": item.get("open", item.get("close", 0)),
                            "high": item.get("high", item.get("close", 0)),
                            "low": item.get("low", item.get("close", 0)),
                            "close": item.get("close", 0),
                            "volume": item.get("volume", 0),
                            "timestamp": item.get(
                                "timestamp",
                                int(timestamp_ms if "timestamp" in item else 0),
                            ),
                        }
                    )
                historical_data = converted_data
                # Cache the fetched data
                st.session_state.historical_data[extended_cache_key] = historical_data
            else:
                st.error(
                    f"No data available for {selected_symbol} from any exchange. Please check symbol or try again later."
                )
                return
        except Exception as e:
            st.error(f"Failed to fetch data for {selected_symbol}: {e}")
            st.info("💡 Background service will collect data automatically over time.")
            return

    # Convert to DataFrame with full dataset
    df_full = pd.DataFrame(historical_data)

    # Handle different timestamp formats and ensure proper datetime index
    if "timestamp" in df_full.columns:
        df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], unit="ms")
        df_full.set_index("timestamp", inplace=True)
    elif "date" in df_full.columns:
        df_full["date"] = pd.to_datetime(df_full["date"])
        df_full.set_index("date", inplace=True)

    # Ensure the index is timezone-naive for consistent comparison
    if df_full.index.tz is not None:
        df_full.index = df_full.index.tz_localize(None)

    # Sort by index to ensure proper chronological order
    df_full = df_full.sort_index()

    # Fill missing dates before any processing to prevent gaps in charts
    df_full = fill_missing_dates(df_full, selected_symbol)

    # Calculate all indicators on the full dataset first (ALWAYS use longest period for accuracy)
    df_with_indicators = calculate_all_indicators(df_full)

    # Strategy: Use full dataset with indicators for chart display
    # Let the chart/visualization handle the selected period range, not the data processing
    # This ensures indicators show full coverage from the beginning of available data

    # Use the full dataset - do NOT subset here!
    # The selected period will be handled by chart display logic
    df_for_display = df_with_indicators.copy()

    # Validate we have data
    if len(df_for_display) == 0:
        show_empty_chart(
            title=f"No Data for {selected_symbol}",
            message=f"No historical data available for {selected_symbol}. Data is being collected in the background.",
            height=500,
        )
        return

    # Handle aggregation for weekly and monthly timeframes
    if selected_timeframe == "1 week":
        # Check if we have enough data for weekly aggregation (need at least 7 days)
        if len(df_for_display) < 7:
            st.warning(
                f"⚠️ Limited data: Only {len(df_for_display)} daily candles available. Need at least 7 days for weekly aggregation."
            )
            st.info(
                "Displaying available daily data instead. Try increasing the period or wait for more data collection."
            )
            df = df_for_display
        else:
            with st.spinner(
                f"Aggregating {len(df_for_display)} daily candles to weekly data..."
            ):
                original_df = df_for_display.copy()
                df = aggregate_to_weekly_with_indicators(
                    df_for_display, selected_symbol
                )
                if len(df) > 0 and len(df) != len(original_df):
                    # Successfully converted to weekly candles
                    df = df
                else:
                    st.warning(
                        f"⚠️ Weekly aggregation returned same data. Using daily candles instead."
                    )
                    df = original_df
    elif selected_timeframe == "1 month":
        # Check if we have enough data for monthly aggregation (need at least 30 days for 1 month)
        if len(df_for_display) < 30:
            st.warning(
                f"⚠️ Limited data: Only {len(df_for_display)} daily candles available. Need at least 30 days for proper monthly aggregation."
            )
            st.info(
                "Displaying available daily data instead. Try increasing the period or wait for more data collection."
            )
            df = df_for_display
        else:
            with st.spinner(
                f"Aggregating {len(df_for_display)} daily candles to monthly data..."
            ):
                original_df = (
                    df_for_display.copy()
                )  # Keep copy in case aggregation fails
                df = aggregate_to_monthly_with_indicators(
                    df_for_display, selected_symbol
                )
                if len(df) > 0 and len(df) != len(
                    original_df
                ):  # Check if aggregation actually happened
                    # Successfully converted to monthly candles
                    df = df
                else:
                    st.warning(
                        f"⚠️ Monthly aggregation returned same data. Using daily candles instead."
                    )
                    df = original_df
    else:
        # Use daily/hourly data directly
        df = df_for_display

    # Final validation
    if len(df) == 0:
        show_empty_chart(
            title=f"No Data for {selected_timeframe} - {selected_period}",
            message=f"No data points available after processing. Try a different timeframe or period combination, or wait for more data collection.",
            height=500,
        )
        return

    # Data processing summary removed as requested

    # Only proceed with charting if we have sufficient data
    if len(df) >= 10:  # Need at least 10 points for meaningful charts
        # Chart controls
        show_chart_controls(df, selected_symbol)

        # Handle indicator calculation differently based on timeframe
        if selected_timeframe in ["1 week", "1 month"]:
            # For weekly/monthly timeframes, indicators are already calculated on aggregated data
            # DO NOT overwrite with daily indicators as they would be meaningless
            # df already has the correct indicators calculated in the aggregation functions
            pass
        else:
            # For daily/hourly timeframes: recalculate indicators on the FULL dataset (not subset)
            # This ensures accurate indicator values using maximum historical data (5 years for daily, etc.)
            # Indicators calculated on subsets (e.g., 30 days) would be inaccurate
            df_with_indicators_full = calculate_all_indicators(df_with_indicators)

            # Apply the recalculated indicators to our subset data by matching dates
            # This gives us accurate indicator values for the visualization period
            df = apply_indicators_to_subset(df, df_with_indicators_full)

        # Validation: Check if indicators are present and valid before charting
        if "RSI" in df.columns:
            rsi_values = df["RSI"].dropna()
            if len(rsi_values) > 0:
                pass  # RSI values are valid
            else:
                pass  # No valid RSI values
        else:
            pass  # RSI column not present

        # Create the main chart with period info for visualization filtering
        # Calculate expected periods based on selected timeframe and period
        if selected_timeframe == "1 hour":
            # For hourly: base_days × 24 hours per day
            expected_periods = base_days * 24
        elif selected_timeframe == "4 hours":
            # For 4-hourly: base_days × 6 periods per day (24/4 = 6)
            expected_periods = base_days * 6
        elif selected_timeframe == "1 day":
            # For daily: use base_days directly (days = data points)
            expected_periods = base_days
        elif selected_timeframe == "1 week":
            # For weekly: base_days ÷ 7 days per week
            expected_periods = max(1, base_days // 7)
        elif selected_timeframe == "1 month":
            # For monthly: base_days ÷ 30 days per month
            expected_periods = max(1, base_days // 30)
        else:
            # Fallback for any other timeframes
            expected_periods = base_days

        # For aggregated timeframes, check if we have enough data after aggregation
        # If we have fewer periods than expected, show all available (don't truncate)
        # If we have more periods than expected, show only the requested period
        if selected_timeframe in ["4 hours", "1 week", "1 month"]:
            if len(df) <= expected_periods:
                display_periods = None  # Show all available data (insufficient data)
            else:
                display_periods = expected_periods  # Show only requested period
        else:
            display_periods = expected_periods

        show_main_chart(df, selected_symbol, selected_timeframe, display_periods)

        # Signal analysis section
        show_signal_analysis(df, selected_symbol, selected_timeframe)
    else:
        st.warning(
            f"⚠️ Cannot create chart: Insufficient data ({len(df)} points, need ≥10)"
        )
        st.info("💡 **Possible Solutions:**")
        st.info("• Try a longer period (more days/months/years)")
        st.info("• Try a different timeframe")
        st.info("• Wait for background service to collect more historical data")
        st.info(f"• Check if {selected_symbol} has sufficient market data available")


def show_chart_controls(df: pd.DataFrame, symbol: str):
    """Show collapsible chart controls for indicator selection."""
    with st.expander("🎛️ Chart Controls", expanded=False):
        st.subheader("📊 Technical Indicators")

        # Create tabs for different indicator categories
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Moving Averages", "Oscillators", "Bollinger Bands", "Other"]
        )

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Simple Moving Averages (SMA)")
                show_sma = st.checkbox("Show SMA", value=True, key="show_sma")

                if show_sma:
                    # Initialize chart settings if not exists
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}

                    # Get saved number of SMAs or default to 2
                    saved_num_smas = (
                        st.session_state.settings.get("chart_num_smas", 2)
                        if "settings" in st.session_state
                        else 2
                    )
                    num_smas = st.session_state.chart_settings.get(
                        "num_smas", saved_num_smas
                    )

                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        num_smas = st.number_input(
                            "Number of SMAs",
                            min_value=1,
                            max_value=5,
                            value=num_smas,
                            key="num_smas_input",
                        )
                        st.session_state.chart_settings["num_smas"] = num_smas

                    with col_b:
                        st.write("SMA Periods:")
                        sma_periods = []
                        default_sma_values = [20, 50, 100, 150, 200]

                        for i in range(num_smas):
                            # Try to get saved period, fall back to chart_settings, then to default
                            saved_period = (
                                st.session_state.settings.get(
                                    f"chart_sma_period_{i}", default_sma_values[i]
                                )
                                if "settings" in st.session_state
                                else default_sma_values[i]
                            )
                            current_default = st.session_state.chart_settings.get(
                                f"sma_period_{i}", saved_period
                            )
                            period = st.number_input(
                                f"SMA {i+1}",
                                min_value=1,
                                max_value=200,
                                value=current_default,
                                key=f"sma_{i}",
                            )
                            sma_periods.append(period)
                            st.session_state.chart_settings[f"sma_period_{i}"] = period

                        st.session_state.chart_settings["sma_periods"] = sma_periods
                else:
                    # Set default SMA periods when not customized
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}
                    st.session_state.chart_settings["sma_periods"] = [20, 50]

            with col2:
                st.subheader("Exponential Moving Averages (EMA)")
                show_ema = st.checkbox("Show EMA", value=False, key="show_ema")

                if show_ema:
                    # Initialize chart settings if not exists
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}

                    # Get saved number of EMAs or default to 2
                    saved_num_emas = (
                        st.session_state.settings.get("chart_num_emas", 2)
                        if "settings" in st.session_state
                        else 2
                    )
                    num_emas = st.session_state.chart_settings.get(
                        "num_emas", saved_num_emas
                    )

                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        num_emas = st.number_input(
                            "Number of EMAs",
                            min_value=1,
                            max_value=5,
                            value=num_emas,
                            key="num_emas_input",
                        )
                        st.session_state.chart_settings["num_emas"] = num_emas

                    with col_b:
                        st.write("EMA Periods:")
                        ema_periods = []
                        default_ema_values = [12, 26, 50, 100, 200]

                        for i in range(num_emas):
                            # Try to get saved period, fall back to chart_settings, then to default
                            saved_period = (
                                st.session_state.settings.get(
                                    f"chart_ema_period_{i}", default_ema_values[i]
                                )
                                if "settings" in st.session_state
                                else default_ema_values[i]
                            )
                            current_default = st.session_state.chart_settings.get(
                                f"ema_period_{i}", saved_period
                            )
                            period = st.number_input(
                                f"EMA {i+1}",
                                min_value=1,
                                max_value=200,
                                value=current_default,
                                key=f"ema_{i}",
                            )
                            ema_periods.append(period)
                            st.session_state.chart_settings[f"ema_period_{i}"] = period

                        st.session_state.chart_settings["ema_periods"] = ema_periods
                else:
                    # Set default EMA periods when not shown (for fallback)
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}
                    st.session_state.chart_settings["ema_periods"] = (
                        [12, 26] if not show_ema else []
                    )

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                show_rsi = st.checkbox("Show RSI", value=True, key="show_rsi")
                if show_rsi:
                    # Get saved RSI period or default to 14
                    saved_rsi_period = (
                        st.session_state.settings.get("chart_rsi_period", 14)
                        if "settings" in st.session_state
                        else 14
                    )
                    rsi_period = st.number_input(
                        "RSI Period",
                        min_value=1,
                        max_value=100,
                        value=saved_rsi_period,
                        key="rsi_period",
                    )
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}
                    st.session_state.chart_settings["rsi_period"] = rsi_period

                show_stoch = st.checkbox(
                    "Show Stochastic", value=False, key="show_stoch"
                )
                if show_stoch:
                    # Get saved Stochastic parameters or defaults
                    saved_stoch_k = (
                        st.session_state.settings.get("chart_stoch_k", 14)
                        if "settings" in st.session_state
                        else 14
                    )
                    saved_stoch_d = (
                        st.session_state.settings.get("chart_stoch_d", 3)
                        if "settings" in st.session_state
                        else 3
                    )
                    stoch_k = st.number_input(
                        "Stochastic %K",
                        min_value=1,
                        max_value=100,
                        value=saved_stoch_k,
                        key="stoch_k",
                    )
                    stoch_d = st.number_input(
                        "Stochastic %D",
                        min_value=1,
                        max_value=100,
                        value=saved_stoch_d,
                        key="stoch_d",
                    )
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}
                    st.session_state.chart_settings["stoch_k"] = stoch_k
                    st.session_state.chart_settings["stoch_d"] = stoch_d

            with col2:
                show_macd = st.checkbox("Show MACD", value=True, key="show_macd")
                if show_macd:
                    # Get saved MACD parameters or defaults
                    saved_macd_fast = (
                        st.session_state.settings.get("chart_macd_fast", 12)
                        if "settings" in st.session_state
                        else 12
                    )
                    saved_macd_slow = (
                        st.session_state.settings.get("chart_macd_slow", 26)
                        if "settings" in st.session_state
                        else 26
                    )
                    saved_macd_signal = (
                        st.session_state.settings.get("chart_macd_signal", 9)
                        if "settings" in st.session_state
                        else 9
                    )
                    macd_fast = st.number_input(
                        "MACD Fast",
                        min_value=1,
                        max_value=100,
                        value=saved_macd_fast,
                        key="macd_fast",
                    )
                    macd_slow = st.number_input(
                        "MACD Slow",
                        min_value=1,
                        max_value=100,
                        value=saved_macd_slow,
                        key="macd_slow",
                    )
                    macd_signal = st.number_input(
                        "MACD Signal",
                        min_value=1,
                        max_value=100,
                        value=saved_macd_signal,
                        key="macd_signal",
                    )
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}
                    st.session_state.chart_settings["macd_fast"] = macd_fast
                    st.session_state.chart_settings["macd_slow"] = macd_slow
                    st.session_state.chart_settings["macd_signal"] = macd_signal

                show_williams = st.checkbox(
                    "Show Williams %R", value=False, key="show_williams"
                )
                if show_williams:
                    # Get saved Williams %R period or default
                    saved_williams_period = (
                        st.session_state.settings.get("chart_williams_period", 14)
                        if "settings" in st.session_state
                        else 14
                    )
                    williams_period = st.number_input(
                        "Williams %R Period",
                        min_value=1,
                        max_value=100,
                        value=saved_williams_period,
                        key="williams_period",
                    )
                    if "chart_settings" not in st.session_state:
                        st.session_state.chart_settings = {}
                    st.session_state.chart_settings["williams_period"] = williams_period

        with tab3:
            show_bb = st.checkbox("Show Bollinger Bands", value=False, key="show_bb")
            if show_bb:
                col1, col2 = st.columns(2)
                with col1:
                    # Get saved Bollinger Bands parameters or defaults
                    saved_bb_period = (
                        st.session_state.settings.get("chart_bb_period", 20)
                        if "settings" in st.session_state
                        else 20
                    )
                    bb_period = st.number_input(
                        "BB Period",
                        min_value=1,
                        max_value=100,
                        value=saved_bb_period,
                        key="bb_period",
                    )
                with col2:
                    saved_bb_std = (
                        st.session_state.settings.get("chart_bb_std", 2.0)
                        if "settings" in st.session_state
                        else 2.0
                    )
                    bb_std = st.number_input(
                        "BB Std Dev",
                        min_value=0.1,
                        max_value=5.0,
                        value=saved_bb_std,
                        step=0.1,
                        key="bb_std",
                    )

                shade_bb = st.checkbox(
                    "Shade area between bands", value=True, key="shade_bb"
                )
                if "chart_settings" not in st.session_state:
                    st.session_state.chart_settings = {}
                st.session_state.chart_settings["bb_period"] = bb_period
                st.session_state.chart_settings["bb_std"] = bb_std
                st.session_state.chart_settings["shade_bb"] = shade_bb

        with tab4:
            show_volume = st.checkbox("Show Volume", value=True, key="show_volume")

            # Volume trend analysis
            show_volume_trend = st.checkbox(
                "Show Volume Trend", value=False, key="show_volume_trend"
            )
            if show_volume_trend:
                # Get saved Volume MA period or default
                saved_volume_ma_period = (
                    st.session_state.settings.get("chart_volume_ma_period", 20)
                    if "settings" in st.session_state
                    else 20
                )
                volume_ma_period = st.number_input(
                    "Volume MA Period",
                    min_value=1,
                    max_value=100,
                    value=saved_volume_ma_period,
                    key="volume_ma",
                )
                if "chart_settings" not in st.session_state:
                    st.session_state.chart_settings = {}
                st.session_state.chart_settings["volume_ma_period"] = volume_ma_period

        # Store all checkbox states in session state
        if "chart_settings" not in st.session_state:
            st.session_state.chart_settings = {}
        st.session_state.chart_settings.update(
            {
                "show_sma": show_sma,
                "show_ema": show_ema,
                "show_rsi": show_rsi,
                "show_macd": show_macd,
                "show_stoch": show_stoch,
                "show_williams": show_williams,
                "show_bb": show_bb,
                "show_volume": show_volume,
                "show_volume_trend": show_volume_trend,
            }
        )

        # Add Save and Reset buttons
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Chart Parameters"):
                # Save all chart settings to user settings
                from components.shared import save_all_settings

                # Save chart parameters to settings
                if "settings" not in st.session_state:
                    st.session_state.settings = {}

                # Store all chart-related parameters in settings
                chart_params = {
                    "chart_rsi_period": st.session_state.chart_settings.get(
                        "rsi_period", 14
                    ),
                    "chart_macd_fast": st.session_state.chart_settings.get(
                        "macd_fast", 12
                    ),
                    "chart_macd_slow": st.session_state.chart_settings.get(
                        "macd_slow", 26
                    ),
                    "chart_macd_signal": st.session_state.chart_settings.get(
                        "macd_signal", 9
                    ),
                    "chart_stoch_k": st.session_state.chart_settings.get("stoch_k", 14),
                    "chart_stoch_d": st.session_state.chart_settings.get("stoch_d", 3),
                    "chart_williams_period": st.session_state.chart_settings.get(
                        "williams_period", 14
                    ),
                    "chart_bb_period": st.session_state.chart_settings.get(
                        "bb_period", 20
                    ),
                    "chart_bb_std": st.session_state.chart_settings.get("bb_std", 2.0),
                    "chart_volume_ma_period": st.session_state.chart_settings.get(
                        "volume_ma_period", 20
                    ),
                    "chart_num_smas": st.session_state.chart_settings.get(
                        "num_smas", 2
                    ),
                    "chart_num_emas": st.session_state.chart_settings.get(
                        "num_emas", 2
                    ),
                }

                # Save individual SMA and EMA periods
                for i in range(5):  # Max 5 SMAs/EMAs
                    sma_key = f"chart_sma_period_{i}"
                    ema_key = f"chart_ema_period_{i}"
                    if sma_key.replace("chart_", "") in st.session_state.chart_settings:
                        chart_params[sma_key] = st.session_state.chart_settings[
                            sma_key.replace("chart_", "")
                        ]
                    if ema_key.replace("chart_", "") in st.session_state.chart_settings:
                        chart_params[ema_key] = st.session_state.chart_settings[
                            ema_key.replace("chart_", "")
                        ]

                # Update settings with chart parameters
                st.session_state.settings.update(chart_params)

                # Save to database
                save_all_settings()

                # Clear chart cache to force recalculation with new parameters
                if hasattr(st.session_state, "historical_data"):
                    # Clear chart-related cache entries
                    keys_to_remove = [
                        key
                        for key in st.session_state.historical_data.keys()
                        if "_chart" in key
                    ]
                    for key in keys_to_remove:
                        del st.session_state.historical_data[key]

                st.success(
                    "Chart parameters saved! Settings will be applied to future chart loads."
                )

        with col2:
            if st.button("🔄 Reset to Defaults"):
                # Reset all chart parameters to default values
                default_chart_params = {
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "stoch_k": 14,
                    "stoch_d": 3,
                    "williams_period": 14,
                    "bb_period": 20,
                    "bb_std": 2.0,
                    "volume_ma_period": 20,
                    "num_smas": 2,
                    "num_emas": 2,
                    "sma_period_0": 20,
                    "sma_period_1": 50,
                    "sma_period_2": 100,
                    "sma_period_3": 150,
                    "sma_period_4": 200,
                    "ema_period_0": 12,
                    "ema_period_1": 26,
                    "ema_period_2": 50,
                    "ema_period_3": 100,
                    "ema_period_4": 200,
                }

                # Update chart_settings with defaults
                if "chart_settings" not in st.session_state:
                    st.session_state.chart_settings = {}
                st.session_state.chart_settings.update(default_chart_params)

                # Also save defaults to user settings
                from components.shared import save_all_settings

                if "settings" not in st.session_state:
                    st.session_state.settings = {}

                # Add chart_ prefix for settings storage
                chart_defaults_for_settings = {
                    f"chart_{key}": value for key, value in default_chart_params.items()
                }
                st.session_state.settings.update(chart_defaults_for_settings)

                # Save defaults to database
                save_all_settings()

                # Clear chart cache to force recalculation with default parameters
                if hasattr(st.session_state, "historical_data"):
                    keys_to_remove = [
                        key
                        for key in st.session_state.historical_data.keys()
                        if "_chart" in key
                    ]
                    for key in keys_to_remove:
                        del st.session_state.historical_data[key]

                st.success(
                    "Parameters reset to defaults! Refreshing with default values..."
                )
                st.rerun()


def show_main_chart(
    df: pd.DataFrame, symbol: str, timeframe: str, display_days: int = None
):
    """Create and display the main TradingView-style chart with optional period filtering for visualization."""
    settings = getattr(st.session_state, "chart_settings", {})

    # Apply period filtering for visualization (not for indicators)
    # Indicators are already calculated on full dataset, now we subset for display only
    chart_df = df.copy()
    if display_days is not None and len(df) > display_days:
        # Use the last N data points for visualization
        chart_df = df.tail(display_days).copy()
    elif display_days is not None:
        # display_days is specified but we have fewer data points than requested
        pass
    else:
        # display_days is None - show all available data (common for aggregated timeframes)
        pass

    # Use the filtered data for chart display
    df = chart_df

    # Indicators are already calculated, just use them based on settings

    # Create subplots
    rows = 1
    if settings.get("show_volume", True):
        rows += 1
    if settings.get("show_rsi", True):
        rows += 1
    if settings.get("show_macd", True):
        rows += 1
    if settings.get("show_stoch", False):
        rows += 1
    if settings.get("show_williams", False):
        rows += 1

    row_heights = [0.6] + [0.4 / (rows - 1)] * (rows - 1) if rows > 1 else [1.0]

    subplot_titles = [f"{symbol} - {timeframe}"]
    if settings.get("show_volume", True):
        subplot_titles.append("Volume")
    if settings.get("show_rsi", True):
        subplot_titles.append("RSI")
    if settings.get("show_macd", True):
        subplot_titles.append("MACD")
    if settings.get("show_stoch", False):
        subplot_titles.append("Stochastic")
    if settings.get("show_williams", False):
        subplot_titles.append("Williams %R")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # More distinctive colors for moving averages (SMA then EMA) - avoiding red to prevent confusion with candlesticks
    sma_colors = [
        "#2196F3",
        "#00CC66",
        "#FF9800",
        "#9C27B0",
        "#607D8B",
    ]  # Blue, Green, Orange, Purple, Blue Grey
    ema_colors = [
        "#64B5F6",
        "#66BB6A",
        "#FFB74D",
        "#BA68C8",
        "#90A4AE",
    ]  # Lighter shades for EMA
    color_index = 0

    # Moving averages on main chart
    if settings.get("show_sma", True):
        sma_periods = settings.get(
            "sma_periods", [20, 50]
        )  # Always default to [20, 50]
        for i, period in enumerate(sma_periods):
            if f"SMA_{period}" in df.columns:
                # Ensure we have valid data (not all NaN)
                sma_data = df[f"SMA_{period}"].dropna()
                if len(sma_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f"SMA_{period}"],
                            name=f"SMA {period}",
                            line=dict(color=sma_colors[i % len(sma_colors)], width=2),
                        ),
                        row=1,
                        col=1,
                    )

    if settings.get("show_ema", False):
        ema_periods = settings.get(
            "ema_periods", [12, 26]
        )  # Always default to [12, 26]
        for i, period in enumerate(ema_periods):
            if f"EMA_{period}" in df.columns:
                # Ensure we have valid data (not all NaN)
                ema_data = df[f"EMA_{period}"].dropna()
                if len(ema_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f"EMA_{period}"],
                            name=f"EMA {period}",
                            line=dict(
                                color=ema_colors[i % len(ema_colors)],
                                width=2,
                                dash="dash",
                            ),
                        ),
                        row=1,
                        col=1,
                    )

    # Bollinger Bands
    if settings.get("show_bb", False) and all(
        col in df.columns for col in ["BB_Upper", "BB_Middle", "BB_Lower"]
    ):
        # Add bands with purple color scheme
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="#8b5cf6", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="#8b5cf6", width=1),
                fill="tonexty" if settings.get("shade_bb", True) else None,
                fillcolor=(
                    "rgba(139, 92, 246, 0.1)"
                    if settings.get("shade_bb", True)
                    else None
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Middle"],
                name="BB Middle",
                line=dict(color="#a855f7", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    current_row = 1

    # Volume
    if settings.get("show_volume", True):
        current_row += 1
        colors_vol = [
            "#ef5350" if df["close"].iloc[i] < df["open"].iloc[i] else "#26a69a"
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=colors_vol,
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )

        # Volume trend line
        if settings.get("show_volume_trend", False) and "Volume_MA" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Volume_MA"],
                    name=f"Volume MA",
                    line=dict(color="orange", width=2),
                ),
                row=current_row,
                col=1,
            )

    # RSI
    if settings.get("show_rsi", True) and "RSI" in df.columns:
        current_row += 1
        # Only plot RSI if we have valid data (not all NaN)
        rsi_data = df["RSI"].dropna()
        if len(rsi_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")
                ),
                row=current_row,
                col=1,
            )
            fig.add_hline(
                y=70, line_dash="dash", line_color="red", row=current_row, col=1
            )
            fig.add_hline(
                y=30, line_dash="dash", line_color="green", row=current_row, col=1
            )
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        else:
            # Add a placeholder text annotation for missing RSI data
            fig.add_annotation(
                text=f"RSI: Insufficient data for {timeframe} timeframe",
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                font=dict(size=14, color="gray"),
                row=current_row,
                col=1,
            )
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)

    # MACD
    if settings.get("show_macd", True) and all(
        col in df.columns for col in ["MACD", "MACD_Signal", "MACD_Histogram"]
    ):
        current_row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue")),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="red")
            ),
            row=current_row,
            col=1,
        )

        colors_macd = ["red" if val < 0 else "green" for val in df["MACD_Histogram"]]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Histogram"],
                name="Histogram",
                marker_color=colors_macd,
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )

    # Stochastic
    if settings.get("show_stoch", False) and all(
        col in df.columns for col in ["Stoch_K", "Stoch_D"]
    ):
        current_row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Stoch_K"], name="%K", line=dict(color="blue")),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Stoch_D"], name="%D", line=dict(color="red")),
            row=current_row,
            col=1,
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(
            y=20, line_dash="dash", line_color="green", row=current_row, col=1
        )
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)

    # Williams %R
    if settings.get("show_williams", False) and "Williams_R" in df.columns:
        current_row += 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Williams_R"],
                name="Williams %R",
                line=dict(color="orange"),
            ),
            row=current_row,
            col=1,
        )
        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(
            y=-80, line_dash="dash", line_color="green", row=current_row, col=1
        )
        fig.update_yaxes(range=[-100, 0], row=current_row, col=1)

    # Configure x-axis formatting based on timeframe
    xaxis_config = {}
    if timeframe == "1 month":
        # For monthly charts, show months clearly
        xaxis_config.update(
            {
                "dtick": "M1",  # Show every month
                "tickformat": "%b %Y",  # Format as "Jan 2024"
                "tickangle": -45,
            }
        )
    elif timeframe == "1 week":
        # For weekly charts, show weeks
        xaxis_config.update(
            {
                "dtick": 604800000,  # 7 days in milliseconds
                "tickformat": "%b %d",
                "tickangle": -45,
            }
        )
    elif timeframe in ["1 hour", "4 hours"]:
        # For hourly charts, show dates and times
        xaxis_config.update({"tickformat": "%m/%d %H:%M", "tickangle": -45})
    else:
        # Daily charts
        xaxis_config.update({"tickformat": "%b %d", "tickangle": -45})

    # Set default x-axis range to focus on recent data if we have a lot of periods
    xaxis_range = None
    if (
        len(df) > 50
    ):  # If we have more than 50 periods, focus on the most recent portion
        # Show approximately the last 25% of the data by default, but at least 20 periods
        recent_periods = max(20, len(df) // 4)
        recent_start_idx = len(df) - recent_periods
        xaxis_range = [df.index[recent_start_idx], df.index[-1]]

    # Update layout
    fig.update_layout(
        height=600 + (rows - 1) * 150,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        xaxis={**xaxis_config, "range": xaxis_range} if xaxis_range else xaxis_config,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
    )

    # Remove date label from x-axis as requested
    fig.update_xaxes(title_text="")

    st.plotly_chart(fig, width="stretch")


def get_current_indicators_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Get current (latest) values of indicators from DataFrame columns."""
    if len(df) == 0:
        return {
            "current_price": np.nan,
            "rsi": np.nan,
            "sma_20": np.nan,
            "sma_50": np.nan,
            "sma_200": np.nan,
            "ema_12": np.nan,
            "ema_21": np.nan,
            "ema_26": np.nan,
            "macd": np.nan,
            "macd_signal": np.nan,
            "macd_histogram": np.nan,
            "bb_upper": np.nan,
            "bb_middle": np.nan,
            "bb_lower": np.nan,
        }

    # Get latest values from DataFrame columns (already calculated for selected timeframe)
    indicators = {
        "current_price": df["close"].iloc[-1] if "close" in df.columns else np.nan,
        "rsi": (
            df["RSI"].iloc[-1]
            if "RSI" in df.columns and not df["RSI"].isna().all()
            else np.nan
        ),
        "sma_20": (
            df["SMA_20"].iloc[-1]
            if "SMA_20" in df.columns and not df["SMA_20"].isna().all()
            else np.nan
        ),
        "sma_50": (
            df["SMA_50"].iloc[-1]
            if "SMA_50" in df.columns and not df["SMA_50"].isna().all()
            else np.nan
        ),
        "sma_200": (
            df["SMA_200"].iloc[-1]
            if "SMA_200" in df.columns and not df["SMA_200"].isna().all()
            else np.nan
        ),
        "ema_12": (
            df["EMA_12"].iloc[-1]
            if "EMA_12" in df.columns and not df["EMA_12"].isna().all()
            else np.nan
        ),
        "ema_21": (
            df["EMA_21"].iloc[-1]
            if "EMA_21" in df.columns and not df["EMA_21"].isna().all()
            else np.nan
        ),
        "ema_26": (
            df["EMA_26"].iloc[-1]
            if "EMA_26" in df.columns and not df["EMA_26"].isna().all()
            else np.nan
        ),
        "macd": (
            df["MACD"].iloc[-1]
            if "MACD" in df.columns and not df["MACD"].isna().all()
            else np.nan
        ),
        "macd_signal": (
            df["MACD_Signal"].iloc[-1]
            if "MACD_Signal" in df.columns and not df["MACD_Signal"].isna().all()
            else np.nan
        ),
        "macd_histogram": (
            df["MACD_Histogram"].iloc[-1]
            if "MACD_Histogram" in df.columns and not df["MACD_Histogram"].isna().all()
            else np.nan
        ),
        "bb_upper": (
            df["BB_Upper"].iloc[-1]
            if "BB_Upper" in df.columns and not df["BB_Upper"].isna().all()
            else np.nan
        ),
        "bb_middle": (
            df["BB_Middle"].iloc[-1]
            if "BB_Middle" in df.columns and not df["BB_Middle"].isna().all()
            else np.nan
        ),
        "bb_lower": (
            df["BB_Lower"].iloc[-1]
            if "BB_Lower" in df.columns and not df["BB_Lower"].isna().all()
            else np.nan
        ),
    }

    return indicators


def get_colored_status(status: str, indicator_type: str = "general") -> str:
    """Get color-coded status string for indicators."""
    if indicator_type == "rsi":
        if "Overbought" in status:
            return f"🔴 **:red[{status}]**"
        elif "Oversold" in status:
            return f"🟢 **:green[{status}]**"
        else:
            return f"🟡 **:orange[{status}]**"
    elif indicator_type == "ma_position":
        if "Above" in status:
            return f"🟢 **:green[{status}]**"
        elif "Below" in status:
            return f"🔴 **:red[{status}]**"
        else:
            return f"🟡 **:orange[{status}]**"
    elif indicator_type == "macd":
        if "Bullish" in status:
            return f"🟢 **:green[{status}]**"
        elif "Bearish" in status:
            return f"🔴 **:red[{status}]**"
        else:
            return f"🟡 **:orange[{status}]**"
    elif indicator_type == "bb":
        if "Above Upper Band" in status:
            return f"🔴 **:red[{status}]**"
        elif "Below Lower Band" in status:
            return f"🟢 **:green[{status}]**"
        else:
            return f"🟡 **:orange[{status}]**"
    elif indicator_type == "volatility":
        if "High" in status:
            return f"🔴 **:red[{status}]**"
        elif "Low" in status:
            return f"🟢 **:green[{status}]**"
        else:
            return f"🟡 **:orange[{status}]**"
    else:
        return status


def show_signal_analysis(df: pd.DataFrame, symbol: str, timeframe: str = "Unknown"):
    """Show comprehensive signal analysis section."""
    st.subheader("📊 Signal Analysis")

    with st.spinner("Analyzing signals..."):
        # Get current indicators from the DataFrame (already calculated for selected timeframe)
        indicators = get_current_indicators_from_df(df)

        # Calculate trends using the same DataFrame
        trend_analysis = calculate_trend_analysis(df)

        # Calculate volatility using the DataFrame data
        volatility_analysis = TechnicalIndicators.calculate_volatility_details(
            df["close"].values
        )

        # Debugging info to understand the discrepancy
        if len(df) >= 2 and "SMA_20" in df.columns and "SMA_50" in df.columns:
            sma_20_data = df["SMA_20"].dropna()
            sma_50_data = df["SMA_50"].dropna()
            if len(sma_20_data) >= 2 and len(sma_50_data) >= 2:
                pass  # Debug code can be added here if needed

    # Current metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_price = indicators.get("current_price", 0)

        # Calculate 1-period price change
        price_change_delta = None
        price_change_color = None
        if len(df) >= 2:
            prev_price = df["close"].iloc[-2]
            price_diff = current_price - prev_price
            price_pct = (price_diff / prev_price) * 100 if prev_price > 0 else 0

            # Format the delta with color
            if price_diff > 0:
                price_change_delta = f"+${price_diff:.4f} (+{price_pct:.2f}%)"
                price_change_color = "normal"  # Streamlit will show green
            elif price_diff < 0:
                price_change_delta = f"-${abs(price_diff):.4f} ({price_pct:.2f}%)"
                price_change_color = "inverse"  # Streamlit will show red
            else:
                price_change_delta = f"${price_diff:.4f} ({price_pct:.2f}%)"
                price_change_color = "off"

        st.metric(
            "Current Price",
            f"${current_price:.4f}",
            delta=price_change_delta,
            delta_color=price_change_color,
        )

    with col2:
        rsi = indicators.get("rsi", 50)
        if not np.isnan(rsi):
            rsi_status = (
                "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            )
            rsi_color, rsi_emoji = get_signal_color_and_emoji(rsi_status)
            st.metric("RSI", f"{rsi:.1f}", delta=f"{rsi_emoji} {rsi_status}")
        else:
            st.metric("RSI", "N/A", delta="📊 Insufficient Data")

    with col3:
        # Determine overall signal
        signal = determine_overall_signal(indicators, trend_analysis)
        signal_color, signal_emoji = get_signal_color_and_emoji(signal)
        st.metric("Signal", f"{signal_emoji} {signal}")

    with col4:
        volatility_ratio = volatility_analysis.get("volatility_ratio", np.nan)
        if not np.isnan(volatility_ratio):
            vol_status = (
                "High"
                if volatility_ratio > 1.3
                else "Low" if volatility_ratio < 0.7 else "Moderate"
            )
            vol_emoji = (
                "🔴" if vol_status == "High" else "🟢" if vol_status == "Low" else "🟡"
            )
            st.metric(
                "Volatility",
                f"{vol_emoji} {vol_status}",
                delta=f"Ratio: {volatility_ratio:.2f}",
            )
        else:
            st.metric("Volatility", "N/A")

    # Detailed trend analysis
    st.subheader("📈 Trend Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**EMA Trends (10 periods)**")
        for trend_name, trend_data in trend_analysis.get("ema_trends", {}).items():
            direction = trend_data["direction"]
            if direction == "up":
                emoji = "↗️"
                color = "🟢"
                st.markdown(
                    f"{color} {trend_name}: {emoji} **:green[{direction.title()}]**"
                )
            elif direction == "down":
                emoji = "↘️"
                color = "🔴"
                st.markdown(
                    f"{color} {trend_name}: {emoji} **:red[{direction.title()}]**"
                )
            else:
                emoji = "➡️"
                color = "🟡"
                st.markdown(
                    f"{color} {trend_name}: {emoji} **:orange[{direction.title()}]**"
                )

    with col2:
        st.write("**SMA Trends (10 periods)**")
        for trend_name, trend_data in trend_analysis.get("sma_trends", {}).items():
            direction = trend_data["direction"]
            if direction == "up":
                emoji = "↗️"
                color = "🟢"
                st.markdown(
                    f"{color} {trend_name}: {emoji} **:green[{direction.title()}]**"
                )
            elif direction == "down":
                emoji = "↘️"
                color = "🔴"
                st.markdown(
                    f"{color} {trend_name}: {emoji} **:red[{direction.title()}]**"
                )
            else:
                emoji = "➡️"
                color = "🟡"
                st.markdown(
                    f"{color} {trend_name}: {emoji} **:orange[{direction.title()}]**"
                )

    with col3:
        st.write("**MACD Trend**")
        macd_trend = trend_analysis.get("macd_trend", {})
        if macd_trend:
            direction = macd_trend["direction"]
            if direction == "up":
                emoji = "↗️"
                color = "🟢"
                st.markdown(f"{color} MACD: {emoji} **:green[{direction.title()}]**")
            elif direction == "down":
                emoji = "↘️"
                color = "🔴"
                st.markdown(f"{color} MACD: {emoji} **:red[{direction.title()}]**")
            else:
                emoji = "➡️"
                color = "🟡"
                st.markdown(f"{color} MACD: {emoji} **:orange[{direction.title()}]**")

        st.write("**Volume Trend**")
        volume_trend = trend_analysis.get("volume_trend", {})
        if volume_trend:
            direction = volume_trend["direction"]
            if direction == "up":
                emoji = "↗️"
                color = "🟢"
                st.markdown(f"{color} Volume: {emoji} **:green[{direction.title()}]**")
            elif direction == "down":
                emoji = "↘️"
                color = "🔴"
                st.markdown(f"{color} Volume: {emoji} **:red[{direction.title()}]**")
            else:
                emoji = "➡️"
                color = "🟡"
                st.markdown(f"{color} Volume: {emoji} **:orange[{direction.title()}]**")

    # Detailed indicator information
    with st.expander("📋 Detailed Indicators Information", expanded=False):
        st.subheader("All Current Indicator Values")

        col1, col2 = st.columns(2)

        with col1:
            # RSI
            rsi = indicators.get("rsi", np.nan)
            if not np.isnan(rsi):
                rsi_status = (
                    "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                )
                colored_rsi = get_colored_status(rsi_status, "rsi")
                st.markdown(f"**RSI (14):** {rsi:.2f} - {colored_rsi}")
            else:
                st.markdown(f"**RSI (14):** N/A - 🟡 **:orange[Insufficient Data]**")

            # Moving Averages
            sma_20 = indicators.get("sma_20", np.nan)
            sma_50 = indicators.get("sma_50", np.nan)
            sma_200 = indicators.get("sma_200", np.nan)
            if not np.isnan(sma_20):
                trend_20 = "Above" if current_price > sma_20 else "Below"
                colored_trend_20 = get_colored_status(
                    f"Price {trend_20}", "ma_position"
                )
                st.markdown(f"**SMA (20):** ${sma_20:.4f} - {colored_trend_20}")
            if not np.isnan(sma_50):
                trend_50 = "Above" if current_price > sma_50 else "Below"
                colored_trend_50 = get_colored_status(
                    f"Price {trend_50}", "ma_position"
                )
                st.markdown(f"**SMA (50):** ${sma_50:.4f} - {colored_trend_50}")
            if not np.isnan(sma_200):
                trend_200 = "Above" if current_price > sma_200 else "Below"
                colored_trend_200 = get_colored_status(
                    f"Price {trend_200}", "ma_position"
                )
                st.markdown(f"**SMA (200):** ${sma_200:.4f} - {colored_trend_200}")

            # EMA
            ema_12 = indicators.get("ema_12", np.nan)
            ema_21 = indicators.get("ema_21", np.nan)
            ema_26 = indicators.get("ema_26", np.nan)
            if not np.isnan(ema_12):
                trend_12 = "Above" if current_price > ema_12 else "Below"
                colored_trend_12 = get_colored_status(
                    f"Price {trend_12}", "ma_position"
                )
                st.markdown(f"**EMA (12):** ${ema_12:.4f} - {colored_trend_12}")
            if not np.isnan(ema_21):
                trend_21 = "Above" if current_price > ema_21 else "Below"
                colored_trend_21 = get_colored_status(
                    f"Price {trend_21}", "ma_position"
                )
                st.markdown(f"**EMA (21):** ${ema_21:.4f} - {colored_trend_21}")
            if not np.isnan(ema_26):
                trend_26 = "Above" if current_price > ema_26 else "Below"
                colored_trend_26 = get_colored_status(
                    f"Price {trend_26}", "ma_position"
                )
                st.markdown(f"**EMA (26):** ${ema_26:.4f} - {colored_trend_26}")

        with col2:
            # MACD
            macd = indicators.get("macd", np.nan)
            macd_signal = indicators.get("macd_signal", np.nan)
            if not np.isnan(macd) and not np.isnan(macd_signal):
                macd_status = "Bullish" if macd > macd_signal else "Bearish"
                colored_macd = get_colored_status(macd_status, "macd")
                st.markdown(f"**MACD:** {macd:.6f} - {colored_macd}")
                st.markdown(f"**MACD Signal:** {macd_signal:.6f}")

            # Bollinger Bands
            bb_upper = indicators.get("bb_upper", np.nan)
            bb_lower = indicators.get("bb_lower", np.nan)
            if not np.isnan(bb_upper) and not np.isnan(bb_lower):
                if current_price > bb_upper:
                    bb_status = "Above Upper Band"
                elif current_price < bb_lower:
                    bb_status = "Below Lower Band"
                else:
                    bb_status = "Within Bands"
                colored_bb = get_colored_status(bb_status, "bb")
                st.markdown(
                    f"**Bollinger Bands:** ${bb_upper:.4f} / ${bb_lower:.4f} - {colored_bb}"
                )

            # Volatility
            short_vol = volatility_analysis.get("short_term_vol", np.nan)
            long_vol = volatility_analysis.get("long_term_vol", np.nan)
            if not np.isnan(short_vol) and not np.isnan(long_vol):
                st.markdown(f"**Short-term Volatility:** {short_vol:.2%} (Annualized)")
                st.markdown(f"**Long-term Volatility:** {long_vol:.2%} (Annualized)")
                colored_vol = get_colored_status(vol_status, "volatility")
                st.markdown(
                    f"**Volatility Ratio:** {volatility_ratio:.2f} - {colored_vol}"
                )


def calculate_trend_analysis(df: pd.DataFrame) -> Dict:
    """Calculate trend analysis for EMA, SMA, MACD, and Volume."""
    try:
        trends = {}

        # EMA trends - Use DataFrame columns for consistency with selected timeframe
        trends["ema_trends"] = {}

        # EMA 12 trend - compare current to 10 periods ago
        if "EMA_12" in df.columns and len(df) >= 11:
            ema_12_data = df["EMA_12"].dropna()
            if len(ema_12_data) >= 11:
                ema_12_direction = (
                    "up"
                    if ema_12_data.iloc[-1] > ema_12_data.iloc[-11]
                    else (
                        "down"
                        if ema_12_data.iloc[-1] < ema_12_data.iloc[-11]
                        else "neutral"
                    )
                )
                trends["ema_trends"]["EMA 12"] = {"direction": ema_12_direction}

        # EMA 21 trend - compare current to 10 periods ago
        if "EMA_21" in df.columns and len(df) >= 11:
            ema_21_data = df["EMA_21"].dropna()
            if len(ema_21_data) >= 11:
                ema_21_direction = (
                    "up"
                    if ema_21_data.iloc[-1] > ema_21_data.iloc[-11]
                    else (
                        "down"
                        if ema_21_data.iloc[-1] < ema_21_data.iloc[-11]
                        else "neutral"
                    )
                )
                trends["ema_trends"]["EMA 21"] = {"direction": ema_21_direction}

        # EMA 26 trend - compare current to 10 periods ago
        if "EMA_26" in df.columns and len(df) >= 11:
            ema_26_data = df["EMA_26"].dropna()
            if len(ema_26_data) >= 11:
                ema_26_direction = (
                    "up"
                    if ema_26_data.iloc[-1] > ema_26_data.iloc[-11]
                    else (
                        "down"
                        if ema_26_data.iloc[-1] < ema_26_data.iloc[-11]
                        else "neutral"
                    )
                )
                trends["ema_trends"]["EMA 26"] = {"direction": ema_26_direction}

        # SMA trends - Use DataFrame columns for consistency with selected timeframe
        # This is different from price position relative to SMA (which shows in detailed indicators)
        trends["sma_trends"] = {}

        # SMA 20 trend - compare current to 10 periods ago
        if "SMA_20" in df.columns and len(df) >= 11:
            sma_20_data = df["SMA_20"].dropna()
            if len(sma_20_data) >= 11:
                sma_20_direction = (
                    "up"
                    if sma_20_data.iloc[-1] > sma_20_data.iloc[-11]
                    else (
                        "down"
                        if sma_20_data.iloc[-1] < sma_20_data.iloc[-11]
                        else "neutral"
                    )
                )
                trends["sma_trends"]["SMA 20"] = {"direction": sma_20_direction}

        # SMA 50 trend - compare current to 10 periods ago
        if "SMA_50" in df.columns and len(df) >= 11:
            sma_50_data = df["SMA_50"].dropna()
            if len(sma_50_data) >= 11:
                sma_50_direction = (
                    "up"
                    if sma_50_data.iloc[-1] > sma_50_data.iloc[-11]
                    else (
                        "down"
                        if sma_50_data.iloc[-1] < sma_50_data.iloc[-11]
                        else "neutral"
                    )
                )
                trends["sma_trends"]["SMA 50"] = {"direction": sma_50_direction}

        # SMA 200 trend - compare current to 10 periods ago
        if "SMA_200" in df.columns and len(df) >= 11:
            sma_200_data = df["SMA_200"].dropna()
            if len(sma_200_data) >= 11:
                sma_200_direction = (
                    "up"
                    if sma_200_data.iloc[-1] > sma_200_data.iloc[-11]
                    else (
                        "down"
                        if sma_200_data.iloc[-1] < sma_200_data.iloc[-11]
                        else "neutral"
                    )
                )
                trends["sma_trends"]["SMA 200"] = {"direction": sma_200_direction}

        # MACD trend - Use DataFrame columns for consistency
        if "MACD" in df.columns and len(df) >= 2:
            macd_data = df["MACD"].dropna()
            if len(macd_data) >= 2:
                macd_direction = (
                    "up"
                    if macd_data.iloc[-1] > macd_data.iloc[-2]
                    else (
                        "down" if macd_data.iloc[-1] < macd_data.iloc[-2] else "neutral"
                    )
                )
                trends["macd_trend"] = {"direction": macd_direction}

        # Volume trend - Use DataFrame columns for consistency
        if "volume" in df.columns and len(df) >= 10:
            # Calculate volume MA for trend analysis
            volume_ma = TechnicalIndicators.calculate_sma(df["volume"].values, 10)
            if (
                len(volume_ma) >= 2
                and not np.isnan(volume_ma[-1])
                and not np.isnan(volume_ma[-2])
            ):
                volume_direction = (
                    "up"
                    if volume_ma[-1] > volume_ma[-2]
                    else "down" if volume_ma[-1] < volume_ma[-2] else "neutral"
                )
                trends["volume_trend"] = {"direction": volume_direction}

        return trends

    except Exception as e:
        st.error(f"Error calculating trend analysis: {e}")
        return {}


def determine_overall_signal(indicators: Dict, trends: Dict) -> str:
    """Determine overall signal based on indicators and trends."""
    try:
        rsi = indicators.get("rsi", 50)
        current_price = indicators.get("current_price", 0)
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)

        # RSI signals
        rsi_signal = "NEUTRAL"
        if rsi > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi < 30:
            rsi_signal = "OVERSOLD"

        # Moving average signals
        ma_signal = "NEUTRAL"
        if current_price > sma_20 and current_price > sma_50 and sma_20 > sma_50:
            ma_signal = "BUY"
        elif current_price < sma_20 and current_price < sma_50 and sma_20 < sma_50:
            ma_signal = "SELL"

        # Combined signal logic
        if rsi_signal == "OVERSOLD" and ma_signal == "BUY":
            return "BUY"
        elif rsi_signal == "OVERBOUGHT" and ma_signal == "SELL":
            return "SELL"
        elif rsi_signal == "OVERSOLD":
            return "OVERSOLD"
        elif rsi_signal == "OVERBOUGHT":
            return "OVERBOUGHT"
        else:
            return "NEUTRAL"

    except Exception as e:
        return "NEUTRAL"
