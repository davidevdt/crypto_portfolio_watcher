import asyncio
import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from .mexc import MEXCProvider
from .bybit import BybitProvider
from .binance import BinanceProvider
from .kraken import KrakenProvider
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoPriceFetcher:
    """
    Fetches cryptocurrency prices from multiple providers with fallback.

    Manages multiple data providers and implements fallback logic for
    reliable price fetching and historical data retrieval.
    """

    STABLECOINS = {
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "USDD",
        "TUSD",
        "USDTUSDT",
        "USDCUSDT",
        "USDTUSD",
        "USDCUSD",
        "BUSDUSDT",
        "DAIUSDT",
    }

    def __init__(self, providers: Optional[List[Any]] = None):
        """Initialize CryptoPriceFetcher with providers.

        Args:
            providers: List of crypto data providers. Uses defaults if None.
        """
        if providers is None:
            self.providers = [
                BinanceProvider(),
                BybitProvider(),
                MEXCProvider(),
                KrakenProvider(),
            ]
        else:
            self.providers = providers
        self.scheduler = AsyncIOScheduler()
        self.rate_limiters = {
            provider.__class__.__name__: asyncio.Semaphore(
                provider.get_rate_limit().get("requests_per_second", 1)
            )
            for provider in self.providers
        }

        # Create provider lookup dictionary
        self.provider_by_name = {
            provider.__class__.__name__: provider for provider in self.providers
        }

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to trading pair format.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            Normalized trading pair symbol
        """
        if symbol.endswith("USDT") or symbol.endswith("USDC") or symbol.endswith("USD"):
            return symbol

        if symbol in self.STABLECOINS:
            return symbol

        return f"{symbol}USDT"

    def _is_stablecoin(self, symbol: str) -> bool:
        """Check if symbol is a known stablecoin.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            True if symbol is a stablecoin
        """
        return symbol.upper() in self.STABLECOINS

    def _get_preferred_provider(self, symbol: str) -> Optional[str]:
        """Get the preferred data provider for a symbol from database.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            Provider class name if found, None otherwise
        """
        try:
            from database.models import get_session, TrackedAsset

            session = get_session()
            tracked_asset = (
                session.query(TrackedAsset)
                .filter(TrackedAsset.symbol == symbol)
                .first()
            )

            if tracked_asset and tracked_asset.preferred_data_provider:
                provider_name = tracked_asset.preferred_data_provider
                session.close()
                return provider_name

            session.close()
            return None
        except Exception as e:
            logger.warning(f"Could not get preferred provider for {symbol}: {e}")
            return None

    def _update_provider_success(self, symbol: str, provider_name: str, success: bool):
        """Update provider success/failure statistics for a symbol.

        Args:
            symbol: Cryptocurrency symbol
            provider_name: Name of the provider class
            success: Whether the operation was successful
        """
        try:
            from database.models import get_session, TrackedAsset

            session = get_session()
            tracked_asset = (
                session.query(TrackedAsset)
                .filter(TrackedAsset.symbol == symbol)
                .first()
            )

            if tracked_asset:
                # Initialize counts if they're None (for backward compatibility)
                if tracked_asset.provider_success_count is None:
                    tracked_asset.provider_success_count = 0
                if tracked_asset.provider_fail_count is None:
                    tracked_asset.provider_fail_count = 0

                # Update preferred provider if this was successful
                if success:
                    tracked_asset.preferred_data_provider = provider_name
                    tracked_asset.provider_success_count += 1
                    tracked_asset.last_provider_success = datetime.datetime.utcnow()
                else:
                    tracked_asset.provider_fail_count += 1

                session.commit()

            session.close()
        except Exception as e:
            logger.warning(f"Could not update provider stats for {symbol}: {e}")
            try:
                session.close()
            except:
                pass

    def _get_ordered_providers(self, symbol: str) -> List[Any]:
        """Get providers ordered by preference for a symbol.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            List of providers ordered by preference
        """
        preferred_provider_name = self._get_preferred_provider(symbol)

        if preferred_provider_name and preferred_provider_name in self.provider_by_name:
            # Start with preferred provider, then add others
            preferred_provider = self.provider_by_name[preferred_provider_name]
            other_providers = [p for p in self.providers if p != preferred_provider]
            return [preferred_provider] + other_providers
        else:
            # No preference, use default order
            return self.providers.copy()

    async def get_real_time_price(self, symbol: str) -> float:
        """Get current price for cryptocurrency symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')

        Returns:
            Current price as float

        Raises:
            ValueError: If price cannot be fetched from any provider
        """
        if self._is_stablecoin(symbol):
            return 1.0

        normalized_symbol = self._normalize_symbol(symbol)
        ordered_providers = self._get_ordered_providers(symbol)

        # Log when using preferred provider
        preferred_provider_name = self._get_preferred_provider(symbol)
        if (
            preferred_provider_name
            and ordered_providers[0].__class__.__name__ == preferred_provider_name
        ):
            logger.debug(
                f"Using preferred provider {preferred_provider_name} for {symbol} price"
            )

        for provider in ordered_providers:
            async with self.rate_limiters[provider.__class__.__name__]:
                try:
                    price = await provider.get_real_time_price(normalized_symbol)
                    # Update success statistics
                    self._update_provider_success(
                        symbol, provider.__class__.__name__, True
                    )
                    return price
                except Exception as e:
                    # Update failure statistics
                    self._update_provider_success(
                        symbol, provider.__class__.__name__, False
                    )
                    logger.error(
                        f"Error from {provider.__class__.__name__} for {normalized_symbol}: {e}"
                    )
                    continue
        raise ValueError(
            f"Could not fetch real-time price for {symbol} from any provider."
        )

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols using smart provider selection.

        Args:
            symbols: List of cryptocurrency symbols

        Returns:
            Dictionary mapping symbols to their current prices
        """
        prices = {}
        remaining_symbols = set(symbols)

        # Handle stablecoins first
        for symbol in list(remaining_symbols):
            if self._is_stablecoin(symbol):
                prices[symbol] = 1.0
                remaining_symbols.remove(symbol)

        # Group symbols by their preferred provider to batch requests
        provider_symbol_groups = {}
        symbol_mapping = {}

        for symbol in remaining_symbols:
            normalized = self._normalize_symbol(symbol)
            symbol_mapping[normalized] = symbol

            # Get preferred provider for this symbol
            preferred_provider_name = self._get_preferred_provider(symbol)

            if (
                preferred_provider_name
                and preferred_provider_name in self.provider_by_name
            ):
                if preferred_provider_name not in provider_symbol_groups:
                    provider_symbol_groups[preferred_provider_name] = []
                provider_symbol_groups[preferred_provider_name].append(normalized)
            else:
                # No preference, add to default provider group
                default_provider_name = self.providers[0].__class__.__name__
                if default_provider_name not in provider_symbol_groups:
                    provider_symbol_groups[default_provider_name] = []
                provider_symbol_groups[default_provider_name].append(normalized)

        # Try each provider group
        unfetched_symbols = []

        for provider_name, normalized_symbols in provider_symbol_groups.items():
            if not normalized_symbols:
                continue

            provider = self.provider_by_name.get(provider_name, self.providers[0])

            async with self.rate_limiters[provider.__class__.__name__]:
                try:
                    provider_prices = await provider.get_real_time_prices(
                        normalized_symbols
                    )

                    for normalized_symbol, price in provider_prices.items():
                        if normalized_symbol in symbol_mapping:
                            original_symbol = symbol_mapping[normalized_symbol]
                            prices[original_symbol] = price
                            # Update success statistics
                            self._update_provider_success(
                                original_symbol, provider.__class__.__name__, True
                            )

                    # Track symbols that couldn't be fetched
                    fetched_symbols = set(provider_prices.keys())
                    for normalized_symbol in normalized_symbols:
                        if normalized_symbol not in fetched_symbols:
                            original_symbol = symbol_mapping[normalized_symbol]
                            unfetched_symbols.append(original_symbol)
                            # Update failure statistics
                            self._update_provider_success(
                                original_symbol, provider.__class__.__name__, False
                            )

                except Exception as e:
                    logger.error(f"Error from {provider.__class__.__name__}: {e}")
                    # All symbols in this group failed
                    for normalized_symbol in normalized_symbols:
                        original_symbol = symbol_mapping[normalized_symbol]
                        unfetched_symbols.append(original_symbol)
                        self._update_provider_success(
                            original_symbol, provider.__class__.__name__, False
                        )

        # Fallback: try remaining providers for unfetched symbols
        if unfetched_symbols:
            for symbol in unfetched_symbols:
                try:
                    price = await self.get_real_time_price(
                        symbol
                    )  # This uses smart provider selection
                    prices[symbol] = price
                except Exception:
                    logger.warning(
                        f"Could not fetch price for {symbol} from any provider"
                    )

        return prices

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval: str,
        recent_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol using smart provider selection.

        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
            start_date (datetime.datetime): Start date for the data.
            end_date (datetime.datetime): End date for the data.
            interval (str): Time interval (e.g., '1d', '1h').
            recent_only (bool): If True, fetch only the most recent data points,
                ignoring date parameters.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with OHLCV data.

        Raises:
            ValueError: If data cannot be fetched from any provider.
        """
        if self._is_stablecoin(symbol):
            # Return dummy historical data for stablecoins
            return [
                {
                    "timestamp": int(datetime.datetime.now().timestamp() * 1000),
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 0.0,
                }
            ]

        last_error = None
        preferred_provider_name = self._get_preferred_provider(symbol)
        ordered_providers = self._get_ordered_providers(symbol)

        # MEXC special handling - only move to end if it's first AND not the preferred provider
        # (historical data is often unreliable, but if it's preferred, user wants it tried first)
        if (
            ordered_providers
            and ordered_providers[0].__class__.__name__ == "MEXCProvider"
            and preferred_provider_name != "MEXCProvider"
        ):
            mexc_provider = ordered_providers.pop(0)
            ordered_providers.append(mexc_provider)
            logger.info(
                f"Moved MEXCProvider to end for {symbol} (not preferred for historical data)"
            )

        normalized_symbol = self._normalize_symbol(symbol)

        # Check if we're using a preferred provider and log appropriately
        if (
            preferred_provider_name
            and ordered_providers[0].__class__.__name__ == preferred_provider_name
        ):
            logger.info(
                f"Using preferred provider {preferred_provider_name} for {symbol} historical data"
            )
        elif preferred_provider_name:
            logger.info(
                f"Provider order for {symbol}: {[p.__class__.__name__ for p in ordered_providers]} (preferred: {preferred_provider_name})"
            )

        for provider in ordered_providers:
            async with self.rate_limiters[provider.__class__.__name__]:
                try:
                    data = await provider.get_historical_data(
                        normalized_symbol, start_date, end_date, interval, recent_only
                    )
                    if data:
                        # Update success statistics
                        self._update_provider_success(
                            symbol, provider.__class__.__name__, True
                        )
                        if provider.__class__.__name__ != preferred_provider_name:
                            logger.info(
                                f"Found working provider {provider.__class__.__name__} for {symbol} (was {preferred_provider_name})"
                            )
                        return data
                except Exception as e:
                    last_error = e
                    # Update failure statistics
                    self._update_provider_success(
                        symbol, provider.__class__.__name__, False
                    )
                    logger.error(f"Error from {provider.__class__.__name__}: {e}")
                    continue

        raise ValueError(
            f"Could not fetch historical data for {symbol}. Last error: {last_error}"
        )

    async def get_historical_data_for_symbols(
        self,
        symbols: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get historical data for multiple symbols.

        Args:
            symbols (List[str]): List of cryptocurrency symbols.
            start_date (datetime.datetime): Start date for the data.
            end_date (datetime.datetime): End date for the data.
            interval (str): Time interval (e.g., '1d', '1h').

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of symbol to OHLCV data.
        """
        tasks = [
            self.get_historical_data(symbol, start_date, end_date, interval)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        historical_data = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                historical_data[symbol] = result
            else:
                logger.error(f"Failed to fetch historical data for {symbol}: {result}")
        return historical_data

    async def is_symbol_valid(self, symbol: str) -> bool:
        """
        Check if a symbol exists in any provider.

        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').

        Returns:
            bool: True if the symbol is valid, False otherwise.
        """
        if self._is_stablecoin(symbol):
            return True

        normalized_symbol = self._normalize_symbol(symbol)

        for provider in self.providers:
            async with self.rate_limiters[provider.__class__.__name__]:
                try:
                    exchange_info = await provider.get_exchange_info()
                    available_symbols = (
                        [s["symbol"] for s in exchange_info["symbols"]]
                        if "symbols" in exchange_info
                        else list(exchange_info.keys())
                    )
                    if normalized_symbol in available_symbols:
                        return True
                except Exception as e:
                    logger.error(
                        f"Error checking symbol {normalized_symbol} with {provider.__class__.__name__}: {e}"
                    )
                    continue
        return False

    def start_price_daemon(self, symbols: List[str], interval_minutes: float = 5):
        """
        Start a daemon process to periodically fetch prices for symbols.

        Args:
            symbols (List[str]): List of symbols to fetch prices for.
            interval_minutes (float): Interval between updates in minutes.
        """
        self.scheduler.add_job(
            self.get_current_prices,
            "interval",
            minutes=interval_minutes,
            args=[symbols],
            id="price_update",
        )
        self.scheduler.start()
