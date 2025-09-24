"""
Background Data Service - Handles continuous data fetching and portfolio history calculations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from database.models import (
    get_session,
    HistoricalPrice,
    PortfolioValueHistory,
    AlertHistory,
    TechnicalIndicatorCache,
    TrackedAsset,
)
from database.utils import (
    get_db_session_with_retry,
    safe_db_operation,
    retry_db_operation,
)
from data_providers.data_fetcher import CryptoPriceFetcher
from services.portfolio_manager import PortfolioManager
from services.technical_indicators import TechnicalIndicators
from services.notification_service import NotificationService

logger = logging.getLogger(__name__)


class BackgroundDataService:
    """Service for continuous background data fetching and processing."""

    def __init__(self):
        self.price_fetcher = CryptoPriceFetcher()
        self.portfolio_manager = PortfolioManager()
        self.notification_service = NotificationService()
        self.is_running = False
        self.task = None

    async def start_background_process(self, refresh_interval_seconds: int = 60):
        """Start the background data collection process.

        Args:
            refresh_interval_seconds: Interval between data collection cycles
        """
        if self.is_running:
            logger.info("Background data service is already running")
            return

        self.is_running = True
        logger.info(
            f"Starting background data service with {refresh_interval_seconds}s interval"
        )

        while self.is_running:
            await self.run_data_collection_cycle()
            await asyncio.sleep(refresh_interval_seconds)

    def stop_background_process(self):
        """Stop the background data collection process."""
        self.is_running = False
        logger.info("Stopped background data service")

    async def run_data_collection_cycle(self):
        """Run a complete data collection cycle using tracked assets."""
        cycle_start = datetime.now()
        logger.info(
            f"🔄 Starting data collection cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        try:
            # First, synchronize tracked assets with current portfolios/watchlist (with retry logic)
            try:
                self.portfolio_manager.sync_tracked_assets()
            except Exception as e:
                logger.warning(f"Could not sync tracked assets: {e}")

            # Get symbols from tracked assets table
            tracked_assets = self.portfolio_manager.get_tracked_assets(active_only=True)
            symbols = [asset.symbol for asset in tracked_assets]

            if symbols:
                logger.info(
                    f"📊 Processing {len(symbols)} tracked symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}"
                )

                # Fetch current prices
                await self.update_current_prices(symbols, tracked_assets)

                # Update portfolio values (calculate total values from current prices)
                self.update_portfolio_values()

                # Update historical data (essential for charts and period changes)
                await self.update_historical_data(symbols, tracked_assets)

                # Check for notification triggers
                await self.check_notification_triggers(symbols)

                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                logger.info(
                    f"✅ Data collection cycle completed in {cycle_duration:.1f}s for {len(symbols)} symbols"
                )
            else:
                logger.info("No symbols to track")

        except Exception as e:
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.error(
                f"❌ Error in data collection cycle after {cycle_duration:.1f}s: {e}"
            )

    def get_all_tracked_symbols(self) -> List[str]:
        """Get all symbols that need to be tracked."""
        try:
            tracked_assets = self.portfolio_manager.get_tracked_assets(active_only=True)
            return [asset.symbol for asset in tracked_assets]
        except Exception as e:
            logger.error(f"Error getting tracked symbols: {e}")
            return []

    async def update_current_prices(
        self, symbols: List[str], tracked_assets: List[TrackedAsset]
    ):
        """Fetch and cache current prices for symbols using smart provider selection."""
        try:
            # Log provider preferences for debugging
            provider_stats = {}
            for tracked_asset in tracked_assets:
                if tracked_asset.preferred_data_provider:
                    provider = tracked_asset.preferred_data_provider
                    if provider not in provider_stats:
                        provider_stats[provider] = 0
                    provider_stats[provider] += 1

            if provider_stats:
                logger.info(f"Provider preferences: {provider_stats}")

            # Fetch current prices from exchanges using smart provider selection
            current_prices = await self.price_fetcher.get_current_prices(symbols)

            if current_prices:
                # Store in database cache
                self.portfolio_manager.update_cached_prices(current_prices)

                # Update tracked asset timestamps
                for symbol in current_prices.keys():
                    self.portfolio_manager.update_tracked_asset_timestamp(
                        symbol, "price"
                    )

                logger.info(
                    f"Updated {len(current_prices)} current prices using smart provider selection"
                )

        except Exception as e:
            logger.error(f"Error updating current prices: {e}")

    async def get_realistic_start_date(self, symbol: str, session) -> datetime:
        """Determine a realistic start date for historical data based on existing data."""
        try:
            # Check if we have any existing data for this symbol
            earliest_data = (
                session.query(HistoricalPrice)
                .filter(
                    HistoricalPrice.symbol == symbol,
                    HistoricalPrice.interval == "1d",
                )
                .order_by(HistoricalPrice.date.asc())
                .first()
            )

            if earliest_data:
                # If we have data, start from a bit before our earliest point
                buffer_days = 60  # Larger buffer to catch any missing early data
                return earliest_data.date - timedelta(days=buffer_days)
            else:
                # No existing data - try aggressive approach to get maximum available data
                # Start from early 2020 for most tokens to capture maximum possible data
                aggressive_start = datetime(2020, 1, 1)

                # For major tokens, go back even further
                major_tokens = [
                    "BTC",
                    "ETH",
                    "LTC",
                    "XRP",
                    "ADA",
                    "DOT",
                    "LINK",
                    "BCH",
                    "XLM",
                    "DOGE",
                ]
                if symbol in major_tokens:
                    return datetime(2017, 1, 1)  # Major tokens have longer history
                else:
                    # For newer tokens, still try from 2020 - exchanges will return what's available
                    return aggressive_start

        except Exception as e:
            logger.warning(
                f"Could not determine realistic start date for {symbol}: {e}"
            )
            # Fall back to conservative date for new tokens
            return datetime(2023, 1, 1)

    async def update_historical_data(
        self, symbols: List[str], tracked_assets: List[TrackedAsset]
    ):
        """Fetch maximum historical data every 5 minutes with generous fixed windows."""
        try:
            with get_db_session_with_retry() as session:
                for symbol in symbols:
                    try:
                        # RECENT DATA APPROACH: Always fetch most recent data every cycle
                        # Uses recent_only=True to get latest 1000 candles, not historical ranges
                        logger.info(f"Fetching most recent available data for {symbol}")
                        await self.fetch_maximum_available_data(symbol, session)

                        # Update tracked asset timestamp
                        self.portfolio_manager.update_tracked_asset_timestamp(
                            symbol, "historical"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Could not update historical data for {symbol}: {e}"
                        )

        except Exception as e:
            logger.error(f"Error updating historical data: {e}")

    async def fetch_maximum_available_data(self, symbol: str, session):
        """Fetch most recent data points for any asset using smart provider selection and recent_only mode."""
        try:
            end_date = datetime.now()

            # REQUEST MOST RECENT DATA POINTS using recent_only mode
            # This ignores date ranges and gets the latest available data from exchanges
            # Typically returns 1000 most recent candles ending at current time
            daily_days = 1000  # Not used in recent_only mode, kept for logging
            daily_start = end_date - timedelta(
                days=daily_days
            )  # Not used in recent_only mode

            hourly_hours = 1000  # Not used in recent_only mode, kept for logging
            hourly_start = end_date - timedelta(
                hours=hourly_hours
            )  # Not used in recent_only mode

            logger.info(
                f"Fetching most recent data for {symbol} (recent_only mode - gets latest ~1000 candles)"
            )

            # DAILY DATA - fetch most recent available
            try:
                daily_data = await self.price_fetcher.get_historical_data(
                    symbol, daily_start, end_date, "1d", recent_only=True
                )
                if daily_data:
                    self.replace_historical_data(session, symbol, daily_data, "1d")
                    first_point = datetime.fromtimestamp(
                        daily_data[0]["timestamp"] / 1000
                    )
                    last_point = datetime.fromtimestamp(
                        daily_data[-1]["timestamp"] / 1000
                    )
                    data_age_hours = (end_date - last_point).total_seconds() / 3600
                    logger.info(
                        f"✅ {len(daily_data)} daily candles for {symbol} ({first_point.date()} to {last_point.date()}, age: {data_age_hours:.1f}h)"
                    )
            except Exception as e:
                logger.warning(f"Could not fetch daily data for {symbol}: {e}")

            # HOURLY DATA - fetch most recent available
            try:
                hourly_data = await self.price_fetcher.get_historical_data(
                    symbol, hourly_start, end_date, "1h", recent_only=True
                )
                if hourly_data:
                    self.replace_historical_data(session, symbol, hourly_data, "1h")
                    last_point = datetime.fromtimestamp(
                        hourly_data[-1]["timestamp"] / 1000
                    )
                    data_age_hours = (end_date - last_point).total_seconds() / 3600
                    logger.info(
                        f"✅ {len(hourly_data)} hourly candles for {symbol} (latest: {last_point}, age: {data_age_hours:.1f}h)"
                    )
            except Exception as e:
                logger.warning(f"Could not fetch hourly data for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")

    def replace_historical_data(
        self, session: Session, symbol: str, data: List[Dict], interval: str
    ):
        """Replace all historical data for symbol/interval to maintain constant database size."""
        try:
            # DELETE existing data for this symbol/interval
            deleted_count = (
                session.query(HistoricalPrice)
                .filter(
                    HistoricalPrice.symbol == symbol,
                    HistoricalPrice.interval == interval,
                )
                .delete()
            )

            # INSERT all fresh data
            stored_count = 0
            for item in data:
                timestamp = datetime.fromtimestamp(item["timestamp"] / 1000)
                historical_price = HistoricalPrice(
                    symbol=symbol,
                    date=timestamp,
                    open_price=item["open"],
                    high_price=item["high"],
                    low_price=item["low"],
                    close_price=item["close"],
                    price=item["close"],  # Backward compatibility
                    volume=item.get("volume", 0),
                    interval=interval,
                )
                session.add(historical_price)
                stored_count += 1

            session.commit()
            logger.info(
                f"Replaced {deleted_count} old with {stored_count} fresh {interval} candles for {symbol}"
            )

        except Exception as e:
            logger.error(f"Error replacing historical data for {symbol}: {e}")
            session.rollback()

    def store_historical_data(
        self, session: Session, symbol: str, data: List[Dict], interval: str
    ):
        """Store historical price data in database."""
        try:
            stored_count = 0
            for item in data:
                timestamp = datetime.fromtimestamp(item["timestamp"] / 1000)

                # Check if this data point already exists
                existing = (
                    session.query(HistoricalPrice)
                    .filter(
                        HistoricalPrice.symbol == symbol,
                        HistoricalPrice.date == timestamp,
                        HistoricalPrice.interval == interval,
                    )
                    .first()
                )

                if not existing:
                    # Create new historical price record
                    historical_price = HistoricalPrice(
                        symbol=symbol,
                        date=timestamp,
                        open_price=item["open"],
                        high_price=item["high"],
                        low_price=item["low"],
                        close_price=item["close"],
                        price=item["close"],  # Backward compatibility
                        volume=item.get("volume", 0),
                        interval=interval,
                    )
                    session.add(historical_price)
                    stored_count += 1
                elif existing.volume == 0 and item.get("volume", 0) > 0:
                    # Update existing record if it has missing volume data but new data has volume
                    existing.volume = item.get("volume", 0)
                    existing.open_price = item["open"]
                    existing.high_price = item["high"]
                    existing.low_price = item["low"]
                    existing.close_price = item["close"]
                    existing.price = item["close"]
                    stored_count += 1

            if stored_count > 0:
                logger.info(
                    f"Stored {stored_count} new historical price points for {symbol}"
                )

        except Exception as e:
            logger.error(f"Error storing historical data for {symbol}: {e}")

    def update_portfolio_values(self):
        """Calculate and store portfolio value history."""
        try:
            session = get_session()

            # Get all portfolios
            portfolios = self.portfolio_manager.get_all_portfolios()
            current_prices = self.portfolio_manager.get_cached_prices()

            for portfolio in portfolios:
                try:
                    # Calculate current portfolio value with invested and PnL
                    summary = self.portfolio_manager.get_portfolio_summary(
                        portfolio.id, current_prices
                    )
                    total_value = summary.get("total_value", 0)
                    total_invested = summary.get(
                        "total_spent", 0
                    )  # total_spent = total_invested
                    total_pnl = summary.get("total_return", 0)  # total_return = PnL

                    # Check if we already have today's value
                    today = datetime.now().date()
                    existing = (
                        session.query(PortfolioValueHistory)
                        .filter(
                            PortfolioValueHistory.portfolio_id == portfolio.id,
                            PortfolioValueHistory.date >= today,
                            PortfolioValueHistory.date < today + timedelta(days=1),
                        )
                        .first()
                    )

                    if existing:
                        # Update existing record with all required fields
                        existing.total_value = total_value
                        existing.total_invested = total_invested
                        existing.total_pnl = total_pnl
                        existing.updated_at = datetime.now()
                    else:
                        # Create new record with all required fields
                        history_record = PortfolioValueHistory(
                            portfolio_id=portfolio.id,
                            date=datetime.now(),
                            total_value=total_value,
                            total_invested=total_invested,
                            total_pnl=total_pnl,
                        )
                        session.add(history_record)

                except Exception as e:
                    logger.warning(
                        f"Could not update portfolio {portfolio.id} value: {e}"
                    )

            session.commit()
            logger.info(f"Updated portfolio values for {len(portfolios)} portfolios")

        except Exception as e:
            logger.error(f"Error updating portfolio values: {e}")
        finally:
            if session:
                session.close()

    def calculate_portfolio_history_from_db(
        self, session, assets, days: int = 90, max_days: int = 1825
    ) -> List[Dict]:
        """Calculate portfolio history from database using 5-year dataset, then subset by period."""
        try:
            from database.models import HistoricalPrice

            end_date = datetime.now()
            # Always fetch max_days (1825 = 5 years) worth of data
            full_start_date = end_date - timedelta(days=max_days)
            # But only return the requested period
            selected_start_date = end_date - timedelta(days=days)

            # Collect all symbols
            symbols = [asset.symbol for asset in assets]

            # Get historical data for each symbol from database (full 5 years)
            historical_data = {}
            for symbol in symbols:
                if symbol in ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]:
                    # For stablecoins, create dummy data at $1.00 for the selected period only
                    dates = pd.date_range(selected_start_date, end_date, freq="D")
                    historical_data[symbol] = [
                        {"date": date.strftime("%Y-%m-%d"), "close": 1.0}
                        for date in dates
                    ]
                else:
                    # Query database for 5 years of historical prices
                    historical_prices = (
                        session.query(HistoricalPrice)
                        .filter(
                            HistoricalPrice.symbol == symbol,
                            HistoricalPrice.interval == "1d",
                            HistoricalPrice.date >= full_start_date,
                            HistoricalPrice.date <= end_date,
                        )
                        .order_by(HistoricalPrice.date.asc())
                        .all()
                    )

                    if historical_prices:
                        # Convert and filter to selected period only
                        filtered_data = []
                        for price in historical_prices:
                            try:
                                price_date = (
                                    price.date.date()
                                    if hasattr(price.date, "date")
                                    else price.date
                                )

                                # Only include data within the selected period
                                if (
                                    selected_start_date.date()
                                    <= price_date
                                    <= end_date.date()
                                ):
                                    filtered_data.append(
                                        {
                                            "date": price_date.strftime("%Y-%m-%d"),
                                            "close": float(price.price),
                                        }
                                    )
                            except Exception:
                                continue  # Skip problematic dates

                        historical_data[symbol] = filtered_data

            # Calculate portfolio value for each day
            portfolio_history = []

            # Get all unique dates from selected period
            all_dates = set()
            for symbol_data in historical_data.values():
                for item in symbol_data:
                    all_dates.add(item["date"])

            sorted_dates = sorted(all_dates)

            for date in sorted_dates:
                daily_portfolio_value = 0
                assets_with_data = 0

                # Calculate portfolio value using dot product
                for asset in assets:
                    symbol = asset.symbol
                    quantity = asset.quantity

                    # Handle stablecoins
                    if symbol in ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]:
                        daily_portfolio_value += quantity * 1.0
                        assets_with_data += 1
                    elif symbol in historical_data:
                        # Find close price for this specific date
                        day_data = next(
                            (
                                item
                                for item in historical_data[symbol]
                                if item["date"] == date
                            ),
                            None,
                        )
                        if day_data:
                            close_price = day_data["close"]
                            daily_portfolio_value += quantity * close_price
                            assets_with_data += 1
                        else:
                            # Use backward fill
                            available_dates = [
                                item["date"]
                                for item in historical_data[symbol]
                                if item["date"] <= date
                            ]
                            if available_dates:
                                latest_date = max(available_dates)
                                latest_data = next(
                                    (
                                        item
                                        for item in historical_data[symbol]
                                        if item["date"] == latest_date
                                    ),
                                    None,
                                )
                                if latest_data:
                                    close_price = latest_data["close"]
                                    daily_portfolio_value += quantity * close_price
                                    assets_with_data += 1

                # Only add this day's data if we have price data for at least some assets
                if assets_with_data > 0:
                    portfolio_history.append(
                        {"date": date, "total_value": daily_portfolio_value}
                    )

            return portfolio_history

        except Exception as e:
            logger.error(f"Error calculating portfolio history from DB: {e}")
            return []

    async def check_notification_triggers(self, symbols: List[str]):
        """Check notification triggers for all tracked symbols."""
        try:
            current_prices = self.portfolio_manager.get_cached_prices()

            for symbol in symbols:
                if symbol in current_prices:
                    current_price = current_prices[symbol]

                    # Get historical data for technical analysis
                    historical_data = await self.get_historical_data_for_notifications(
                        symbol
                    )

                    # Get take profit levels for this symbol if it's in portfolio
                    take_profit_levels = self.get_take_profit_levels_for_symbol(symbol)

                    # Process all alerts for this symbol
                    self.notification_service.process_symbol_alerts(
                        symbol, current_price, historical_data, take_profit_levels
                    )

        except Exception as e:
            logger.error(f"Error checking notification triggers: {e}")

    async def get_historical_data_for_notifications(self, symbol: str) -> pd.DataFrame:
        """Get historical data formatted for notification analysis."""
        try:
            session = get_session()

            # Get last 200 daily candles for proper SMA calculation
            historical_records = (
                session.query(HistoricalPrice)
                .filter(
                    HistoricalPrice.symbol == symbol, HistoricalPrice.interval == "1d"
                )
                .order_by(HistoricalPrice.date.desc())
                .limit(200)
                .all()
            )

            session.close()

            if len(historical_records) < 50:  # Need minimum data for analysis
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for record in reversed(
                historical_records
            ):  # Reverse to get chronological order
                data.append(
                    {
                        "date": record.date,
                        "open": record.open_price or record.price,
                        "high": record.high_price or record.price,
                        "low": record.low_price or record.price,
                        "close": record.close_price or record.price,
                        "volume": record.volume or 0,
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(
                f"Error getting historical data for notifications for {symbol}: {e}"
            )
            return pd.DataFrame()

    def get_take_profit_levels_for_symbol(self, symbol: str):
        """Get active take profit levels for a symbol."""
        try:
            session = get_session()

            # Get all active TP levels for assets with this symbol with eager loading
            from database.models import TakeProfitLevel, Asset
            from sqlalchemy.orm import joinedload

            tp_levels = (
                session.query(TakeProfitLevel)
                .options(joinedload(TakeProfitLevel.asset))
                .join(Asset)
                .filter(
                    Asset.symbol == symbol,
                    TakeProfitLevel.is_active == True,
                    TakeProfitLevel.triggered_at == None,
                )
                .all()
            )

            # Detach from session to avoid lazy loading issues
            for tp_level in tp_levels:
                session.expunge(tp_level)
                session.expunge(tp_level.asset)

            session.close()
            return tp_levels

        except Exception as e:
            logger.error(f"Error getting take profit levels for {symbol}: {e}")
            return []


# Create global instance
background_service = BackgroundDataService()
