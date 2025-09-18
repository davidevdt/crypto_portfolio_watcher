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

    async def start_background_process(self, refresh_interval_seconds: int = 300):
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
        logger.info("Starting data collection cycle...")

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
                # Fetch current prices
                await self.update_current_prices(symbols, tracked_assets)

                # Update portfolio values (calculate total values from current prices)
                self.update_portfolio_values()

                # Update historical data (essential for charts and period changes)
                await self.update_historical_data(symbols, tracked_assets)

                # Check for notification triggers
                await self.check_notification_triggers(symbols)

                logger.info(
                    f"Data collection cycle completed for {len(symbols)} symbols"
                )
            else:
                logger.info("No symbols to track")

        except Exception as e:
            logger.error(f"Error in data collection cycle: {e}")

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
        """Fetch and cache current prices for symbols."""
        try:
            # Fetch current prices from exchanges
            current_prices = await self.price_fetcher.get_current_prices(symbols)

            if current_prices:
                # Store in database cache
                self.portfolio_manager.update_cached_prices(current_prices)

                # Update tracked asset timestamps
                for symbol in current_prices.keys():
                    self.portfolio_manager.update_tracked_asset_timestamp(
                        symbol, "price"
                    )

                logger.info(f"Updated {len(current_prices)} current prices")

        except Exception as e:
            logger.error(f"Error updating current prices: {e}")

    async def update_historical_data(
        self, symbols: List[str], tracked_assets: List[TrackedAsset]
    ):
        """Fetch and store historical data for charts and portfolio progress."""
        try:
            with get_db_session_with_retry() as session:
                end_date = datetime.now()

                for symbol in symbols:
                    try:
                        # Check if we need to update daily historical data
                        latest_daily_data = (
                            session.query(HistoricalPrice)
                            .filter(
                                HistoricalPrice.symbol == symbol,
                                HistoricalPrice.interval == "1d",
                            )
                            .order_by(HistoricalPrice.date.desc())
                            .first()
                        )

                        # Check for missing volume data in daily data
                        volume_data_check = (
                            session.query(HistoricalPrice)
                            .filter(
                                HistoricalPrice.symbol == symbol,
                                HistoricalPrice.interval == "1d",
                                HistoricalPrice.volume == 0,
                            )
                            .count()
                        )

                        total_data_points = (
                            session.query(HistoricalPrice)
                            .filter(
                                HistoricalPrice.symbol == symbol,
                                HistoricalPrice.interval == "1d",
                            )
                            .count()
                        )

                        missing_volume_ratio = (
                            (volume_data_check / total_data_points)
                            if total_data_points > 0
                            else 0
                        )

                        # Check if we need to update hourly historical data
                        latest_hourly_data = (
                            session.query(HistoricalPrice)
                            .filter(
                                HistoricalPrice.symbol == symbol,
                                HistoricalPrice.interval == "1h",
                            )
                            .order_by(HistoricalPrice.date.desc())
                            .first()
                        )

                        # Check if we have sufficient historical data range (5 years = 1825 days)
                        required_start_date = end_date - timedelta(days=1825)
                        earliest_daily_data = (
                            session.query(HistoricalPrice)
                            .filter(
                                HistoricalPrice.symbol == symbol,
                                HistoricalPrice.interval == "1d",
                            )
                            .order_by(HistoricalPrice.date.asc())
                            .first()
                        )

                        insufficient_historical_range = (
                            not earliest_daily_data
                            or earliest_daily_data.date.date()
                            > required_start_date.date()
                        )

                        # Only fetch daily if we don't have recent data OR volume data is missing OR insufficient historical range
                        needs_daily_update = (
                            not latest_daily_data
                            or latest_daily_data.date.date() < datetime.now().date()
                            or missing_volume_ratio
                            > 0.3  # Re-fetch if >30% of volume data is missing
                            or insufficient_historical_range
                        )  # Re-fetch if we don't have 5 years of data

                        # Only fetch hourly if we don't have recent hourly data (within last 2 hours)
                        needs_hourly_update = (
                            not latest_hourly_data
                            or latest_hourly_data.date
                            < datetime.now() - timedelta(hours=2)
                        )

                        # Update daily data if needed
                        if needs_daily_update:
                            # Log reason for update
                            if missing_volume_ratio > 0.3:
                                logger.info(
                                    f"Re-fetching {symbol} daily data due to missing volume data ({missing_volume_ratio:.1%} missing)"
                                )
                            if insufficient_historical_range:
                                current_range_days = (
                                    (
                                        latest_daily_data.date.date()
                                        - earliest_daily_data.date.date()
                                    ).days
                                    if earliest_daily_data and latest_daily_data
                                    else 0
                                )
                                logger.info(
                                    f"Re-fetching {symbol} daily data due to insufficient historical range (have {current_range_days} days, need 1825 days)"
                                )

                            # Fetch 5 years of daily data (needed for weekly/monthly 5-year charts and long-term analysis)
                            # Due to API 1000-point limit, fetch in two chunks: older data first, then recent data
                            full_start_date = end_date - timedelta(
                                days=1825
                            )  # 5 years = 1825 days

                            all_historical_data = []

                            # Chunk 1: Fetch older data (5 years ago to ~3 years ago, ~1000 points)
                            chunk1_start = full_start_date
                            chunk1_end = end_date - timedelta(
                                days=825
                            )  # Leave ~825 days for recent data

                            try:
                                logger.info(
                                    f"Fetching older {symbol} data: {chunk1_start.strftime('%Y-%m-%d')} to {chunk1_end.strftime('%Y-%m-%d')}"
                                )
                                older_data = (
                                    await self.price_fetcher.get_historical_data(
                                        symbol, chunk1_start, chunk1_end, "1d"
                                    )
                                )
                                if older_data:
                                    all_historical_data.extend(older_data)
                                    logger.info(
                                        f"Got {len(older_data)} older data points for {symbol}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not fetch older data for {symbol}: {e}"
                                )

                            # Chunk 2: Fetch recent data (from where older data ends to now)
                            chunk2_start = chunk1_end + timedelta(days=1)
                            chunk2_end = end_date

                            try:
                                logger.info(
                                    f"Fetching recent {symbol} data: {chunk2_start.strftime('%Y-%m-%d')} to {chunk2_end.strftime('%Y-%m-%d')}"
                                )
                                recent_data = (
                                    await self.price_fetcher.get_historical_data(
                                        symbol, chunk2_start, chunk2_end, "1d"
                                    )
                                )
                                if recent_data:
                                    all_historical_data.extend(recent_data)
                                    logger.info(
                                        f"Got {len(recent_data)} recent data points for {symbol}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not fetch recent data for {symbol}: {e}"
                                )

                            # Store all collected data
                            if all_historical_data:
                                self.store_historical_data(
                                    session, symbol, all_historical_data, "1d"
                                )
                                logger.info(
                                    f"Updated daily historical data for {symbol} ({len(all_historical_data)} total points)"
                                )

                        # Update hourly data if needed
                        if needs_hourly_update:
                            # Fetch 90 days of hourly data (needed for 1h and 4h timeframes in Asset Charts)
                            # Due to API 1000-point limit, fetch in chunks if needed
                            # 90 days = ~2160 hours, but APIs limit to ~1000 points

                            all_hourly_data = []

                            # Chunk 1: Most recent 30 days (priority for 1h timeframe)
                            recent_start = end_date - timedelta(days=30)
                            try:
                                logger.info(
                                    f"Fetching recent {symbol} hourly data: last 30 days"
                                )
                                recent_hourly = (
                                    await self.price_fetcher.get_historical_data(
                                        symbol, recent_start, end_date, "1h"
                                    )
                                )
                                if recent_hourly:
                                    all_hourly_data.extend(recent_hourly)
                                    logger.info(
                                        f"Got {len(recent_hourly)} recent hourly points for {symbol}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not fetch recent hourly data for {symbol}: {e}"
                                )

                            # Chunk 2: Older hourly data (for 4h timeframe coverage)
                            older_start = end_date - timedelta(days=90)
                            older_end = recent_start - timedelta(hours=1)

                            try:
                                logger.info(
                                    f"Fetching older {symbol} hourly data: 30-90 days ago"
                                )
                                older_hourly = (
                                    await self.price_fetcher.get_historical_data(
                                        symbol, older_start, older_end, "1h"
                                    )
                                )
                                if older_hourly:
                                    all_hourly_data.extend(older_hourly)
                                    logger.info(
                                        f"Got {len(older_hourly)} older hourly points for {symbol}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not fetch older hourly data for {symbol}: {e}"
                                )

                            # Store all collected hourly data
                            if all_hourly_data:
                                self.store_historical_data(
                                    session, symbol, all_hourly_data, "1h"
                                )
                                logger.info(
                                    f"Updated hourly historical data for {symbol} ({len(all_hourly_data)} total points)"
                                )

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
