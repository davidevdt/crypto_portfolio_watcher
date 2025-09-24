import aiohttp
import pandas as pd
from typing import List, Dict, Any
import asyncio
import logging
import datetime
from .base_provider import CryptoDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BybitProvider(CryptoDataProvider):
    """
    Bybit API provider for fetching cryptocurrency data.

    Implements the CryptoDataProvider interface for Bybit exchange API.
    Supports both mainnet and testnet environments.
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize the BybitProvider.

        Args:
            testnet: Whether to use testnet endpoints
        """
        self.base_url = (
            "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        )
        self.ticker_endpoint = f"{self.base_url}/v5/market/tickers"
        self.kline_endpoint = f"{self.base_url}/v5/market/kline"
        self.instruments_endpoint = f"{self.base_url}/v5/market/instruments-info"
        self.rate_limit = {"requests_per_second": 10}

    async def get_real_time_price(self, symbol: str) -> float:
        """
        Get the current price of a cryptocurrency from Bybit.

        Args:
            symbol (str): The symbol of the cryptocurrency (e.g., 'BTCUSDT').

        Returns:
            float: The current price in USDT.

        Raises:
            ValueError: If the price cannot be fetched.
        """
        async with aiohttp.ClientSession() as session:
            params = {"category": "linear", "symbol": symbol}
            try:
                async with session.get(self.ticker_endpoint, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data["retCode"] == 0 and data["result"]["list"]:
                        return float(data["result"]["list"][0]["lastPrice"])
                    raise ValueError(f"No price data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                raise ValueError(f"Unable to fetch real-time price for {symbol}")

    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get the current prices of multiple cryptocurrencies from Bybit.

        Args:
            symbols (List[str]): A list of cryptocurrency symbols.

        Returns:
            Dict[str, float]: A dictionary with symbols as keys and prices as values.
        """
        prices = {}
        tasks = [self.get_real_time_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                prices[symbol] = result
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
        Get historical price data for a cryptocurrency from Bybit.

        Args:
            symbol (str): The symbol of the cryptocurrency.
            start_date (datetime.datetime): The start date for the data.
            end_date (datetime.datetime): The end date for the data.
            interval (str): The time interval (e.g., '1m', '1h', '1d').
            recent_only (bool): If True, fetch only the most recent 1000 candles.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with OHLCV data.

        Raises:
            ValueError: If historical data cannot be fetched.
        """
        bybit_interval = self._convert_interval(interval)

        if recent_only:
            # Fetch only the most recent data without date constraints
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": 1000,  # Bybit max limit - gets most recent 1000 candles
            }
        else:
            # Use date range as before
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "start": int(start_date.timestamp() * 1000),
                "end": int(end_date.timestamp() * 1000),
                "limit": 1000,  # Bybit max limit
            }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.kline_endpoint, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data["retCode"] == 0 and data["result"]["list"]:
                        return self._format_kline_data(data["result"]["list"])
                    raise ValueError(f"No historical data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                raise ValueError(f"Unable to fetch historical data for {symbol}")

    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about available trading pairs from Bybit.

        Returns:
            Dict[str, Any]: A dictionary with symbol information.

        Raises:
            ValueError: If exchange info cannot be fetched.
        """
        params = {"category": "linear"}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    self.instruments_endpoint, params=params
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data["retCode"] == 0:
                        return {item["symbol"]: item for item in data["result"]["list"]}
                    raise ValueError("No exchange info available")
            except Exception as e:
                logger.error(f"Error fetching exchange info: {e}")
                raise ValueError("Unable to fetch exchange info")

    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Get the rate limit configuration for Bybit.

        Returns:
            Dict[str, Any]: Rate limit details.
        """
        return self.rate_limit

    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Bybit format."""
        interval_map = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }
        return interval_map.get(interval, "D")  # Default to daily

    def _format_kline_data(self, data: List[List]) -> List[Dict[str, Any]]:
        """Format Bybit kline data into standard OHLCV format."""
        result = []
        for entry in data:
            result.append(
                {
                    "timestamp": int(entry[0]),
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                }
            )
        return result
