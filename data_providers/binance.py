import datetime
from typing import List, Dict, Any
import aiohttp
import pandas as pd
import numpy as np
import asyncio
import logging
from .base_provider import CryptoDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceProvider(CryptoDataProvider):
    """
    Binance API provider for fetching cryptocurrency data.

    Implements the CryptoDataProvider interface for Binance exchange API.
    Supports real-time prices, historical data, and exchange information.
    """

    def __init__(self):
        """
        Initialize the BinanceProvider.

        Sets up API endpoints and rate limiting configuration.
        """
        self.base_url = "https://api.binance.com/api/v3"
        self.klines_endpoint = f"{self.base_url}/klines"
        self.ticker_endpoint = f"{self.base_url}/ticker/price"
        self.exchange_info_endpoint = f"{self.base_url}/exchangeInfo"
        self.rate_limit = {
            "requests_per_minute": 1200,
            "requests_per_second": 20,
        }

    async def get_real_time_price(self, symbol: str) -> float:
        """
        Get the current price of a cryptocurrency from Binance.

        Args:
            symbol (str): The symbol of the cryptocurrency (e.g., 'BTCUSDT').

        Returns:
            float: The current price in USDT.

        Raises:
            ValueError: If the price cannot be fetched.
        """
        async with aiohttp.ClientSession() as session:
            params = {"symbol": symbol}
            try:
                async with session.get(self.ticker_endpoint, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return float(data["price"])
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                raise ValueError(f"Unable to fetch real-time price for {symbol}")

    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get the current prices of multiple cryptocurrencies from Binance.

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
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data for a cryptocurrency from Binance.

        Args:
            symbol (str): The symbol of the cryptocurrency.
            start_date (datetime.datetime): The start date for the data.
            end_date (datetime.datetime): The end date for the data.
            interval (str): The time interval (e.g., '1m', '1h', '1d').

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with OHLCV data.

        Raises:
            ValueError: If historical data cannot be fetched.
        """
        binance_interval = self._convert_interval(interval)
        params = {
            "symbol": symbol,
            "interval": binance_interval,
            "startTime": int(start_date.timestamp() * 1000),
            "endTime": int(end_date.timestamp() * 1000),
            "limit": 1000,  # Binance max limit
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.klines_endpoint, params=params) as response:
                    response.raise_for_status()
                    klines = await response.json()
                    return self._format_kline_data(klines)
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                raise ValueError(f"Unable to fetch historical data for {symbol}")

    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about available trading pairs from Binance.

        Returns:
            Dict[str, Any]: A dictionary with symbol information.

        Raises:
            ValueError: If exchange info cannot be fetched.
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.exchange_info_endpoint) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                logger.error(f"Error fetching exchange info: {e}")
                raise ValueError("Unable to fetch exchange info")

    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Get the rate limit configuration for Binance.

        Returns:
            Dict[str, Any]: Rate limit details.
        """
        return self.rate_limit

    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Binance format."""
        interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M",
        }
        return interval_map.get(interval, "1d")

    def _format_kline_data(self, data: List[List]) -> List[Dict[str, Any]]:
        """Format Binance kline data into standard OHLCV format."""
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
