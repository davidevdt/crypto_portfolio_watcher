import datetime
from typing import List, Dict, Any
import aiohttp
import pandas as pd
import asyncio
import logging
from .base_provider import CryptoDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KrakenProvider(CryptoDataProvider):
    """
    Kraken API provider for fetching cryptocurrency data.

    Implements the CryptoDataProvider interface for Kraken exchange API.
    Includes symbol normalization for Kraken's naming conventions.
    """

    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        self.ohlc_endpoint = f"{self.base_url}/OHLC"
        self.ticker_endpoint = f"{self.base_url}/Ticker"
        self.asset_pairs_endpoint = f"{self.base_url}/AssetPairs"
        self.rate_limit = {"requests_per_second": 1}
        self._symbol_map = {
            "BTCUSD": "XXBTZUSD",
            "ETHUSD": "XETHZUSD",
            "XDGUSD": "XDGUSD",
        }

    def _normalize_pair_name(self, symbol: str) -> str:
        """
        Normalize symbol to Kraken's format.

        Args:
            symbol: Standard cryptocurrency symbol

        Returns:
            Kraken-formatted symbol
        """
        if not symbol.endswith("USD") and not symbol.endswith("USDT"):
            symbol += "USD"
        return self._symbol_map.get(symbol, symbol)

    def _clean_symbol(self, symbol: str) -> str:
        """Convert Kraken symbol to standard format."""
        reverse_map = {v: k for k, v in self._symbol_map.items()}
        return reverse_map.get(symbol, symbol)

    async def get_real_time_price(self, symbol: str) -> float:
        """
        Get the current price of a cryptocurrency from Kraken.

        Args:
            symbol (str): The symbol of the cryptocurrency (e.g., 'BTCUSD').

        Returns:
            float: The current price in USD.

        Raises:
            ValueError: If the price cannot be fetched.
        """
        symbol = self._normalize_pair_name(symbol)
        async with aiohttp.ClientSession() as session:
            params = {"pair": symbol}
            try:
                async with session.get(self.ticker_endpoint, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if "error" in data and data["error"]:
                        raise ValueError(f"Kraken API error: {data['error']}")
                    return float(data["result"][list(data["result"].keys())[0]]["c"][0])
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                raise ValueError(f"Unable to fetch real-time price for {symbol}")

    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get the current prices of multiple cryptocurrencies from Kraken.

        Args:
            symbols (List[str]): A list of cryptocurrency symbols.

        Returns:
            Dict[str, float]: A dictionary with symbols as keys and prices as values.
        """
        normalized_symbols = [self._normalize_pair_name(s) for s in symbols]
        async with aiohttp.ClientSession() as session:
            params = {"pair": ",".join(normalized_symbols)}
            try:
                async with session.get(self.ticker_endpoint, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if "error" in data and data["error"]:
                        raise ValueError(f"Kraken API error: {data['error']}")
                    prices = {}
                    for s in data["result"].keys():
                        try:
                            prices[self._clean_symbol(s)] = float(
                                data["result"][s]["c"][0]
                            )
                        except Exception:
                            continue
                    return prices
            except Exception as e:
                logger.error(f"Error fetching prices: {e}")
                return {}

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval: str,
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data for a cryptocurrency from Kraken.

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
        symbol = self._normalize_pair_name(symbol)
        interval_minutes = self._convert_interval_to_minutes(interval)
        params = {
            "pair": symbol,
            "interval": interval_minutes,
            "since": int(start_date.timestamp()),
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.ohlc_endpoint, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if "error" in data and data["error"]:
                        raise ValueError(f"Kraken API error: {data['error']}")
                    ohlc_data = data["result"][list(data["result"].keys())[0]]
                    return self._format_ohlc_data(ohlc_data)
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                raise ValueError(f"Unable to fetch historical data for {symbol}")

    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about available asset pairs from Kraken.

        Returns:
            Dict[str, Any]: A dictionary with asset pair information.

        Raises:
            ValueError: If exchange info cannot be fetched.
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.asset_pairs_endpoint) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if "error" in data and data["error"]:
                        raise ValueError(f"Kraken API error: {data['error']}")
                    return data["result"]
            except Exception as e:
                logger.error(f"Error fetching exchange info: {e}")
                raise ValueError("Unable to fetch exchange info")

    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Get the rate limit configuration for Kraken.

        Returns:
            Dict[str, Any]: Rate limit details.
        """
        return self.rate_limit

    def _convert_interval_to_minutes(self, interval: str) -> int:
        """Convert standard interval to Kraken's minute-based format."""
        interval_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
            "2w": 21600,
        }
        return interval_map.get(interval, 1)

    def _format_ohlc_data(self, data: List[List]) -> List[Dict[str, Any]]:
        """Format Kraken OHLC data into standard OHLCV format."""
        result = []
        for entry in data:
            result.append(
                {
                    "timestamp": int(entry[0]) * 1000,
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[6]),
                }
            )
        return result
