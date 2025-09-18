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

        for provider in self.providers:
            async with self.rate_limiters[provider.__class__.__name__]:
                try:
                    price = await provider.get_real_time_price(normalized_symbol)
                    return price
                except Exception as e:
                    logger.error(
                        f"Error from {provider.__class__.__name__} for {normalized_symbol}: {e}"
                    )
                    continue
        raise ValueError(
            f"Could not fetch real-time price for {symbol} from any provider."
        )

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols.

        Args:
            symbols: List of cryptocurrency symbols

        Returns:
            Dictionary mapping symbols to their current prices
        """
        prices = {}
        remaining_symbols = set(symbols)

        for symbol in list(remaining_symbols):
            if self._is_stablecoin(symbol):
                prices[symbol] = 1.0
                remaining_symbols.remove(symbol)

        symbol_mapping = {}
        normalized_symbols = []
        for symbol in remaining_symbols:
            normalized = self._normalize_symbol(symbol)
            symbol_mapping[normalized] = symbol
            normalized_symbols.append(normalized)

        for provider in self.providers:
            if not normalized_symbols:
                break
            async with self.rate_limiters[provider.__class__.__name__]:
                try:
                    provider_prices = await provider.get_real_time_prices(
                        normalized_symbols
                    )
                    for normalized_symbol, price in provider_prices.items():
                        if normalized_symbol in symbol_mapping:
                            original_symbol = symbol_mapping[normalized_symbol]
                            prices[original_symbol] = price
                            if normalized_symbol in normalized_symbols:
                                normalized_symbols.remove(normalized_symbol)
                except Exception as e:
                    logger.error(f"Error from {provider.__class__.__name__}: {e}")
                    continue

        if remaining_symbols:
            logger.warning(f"Could not find prices for: {', '.join(remaining_symbols)}")

        return prices

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval: str,
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol, trying providers in order.

        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
            start_date (datetime.datetime): Start date for the data.
            end_date (datetime.datetime): End date for the data.
            interval (str): Time interval (e.g., '1d', '1h').

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
        providers = self.providers.copy()
        if providers and providers[0].__class__.__name__ == "MEXCProvider":
            providers.pop(0)
            providers.append(MEXCProvider())

        normalized_symbol = self._normalize_symbol(symbol)

        for provider in providers:
            async with self.rate_limiters[provider.__class__.__name__]:
                try:
                    data = await provider.get_historical_data(
                        normalized_symbol, start_date, end_date, interval
                    )
                    if data:
                        return data
                except Exception as e:
                    last_error = e
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
