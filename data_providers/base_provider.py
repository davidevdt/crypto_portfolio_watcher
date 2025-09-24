from abc import ABC, abstractmethod
from typing import List, Dict, Any
import datetime


class CryptoDataProvider(ABC):
    """Abstract base class for cryptocurrency data providers."""

    @abstractmethod
    async def get_real_time_price(self, symbol: str) -> float:
        """
        Get the current price of a cryptocurrency.

        Args:
            symbol (str): The symbol of the cryptocurrency (e.g., 'BTCUSDT', 'ETHUSDT').

        Returns:
            float: The current price of the cryptocurrency in USDT.

        Raises:
            ValueError: If the price cannot be fetched for the given symbol.
        """
        pass

    @abstractmethod
    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get the current prices of multiple cryptocurrencies.

        Args:
            symbols (List[str]): A list of cryptocurrency symbols (e.g., ['BTCUSDT', 'ETHUSDT']).

        Returns:
            Dict[str, float]: A dictionary with symbols as keys and current prices as values.

        Raises:
            ValueError: If prices cannot be fetched for any symbols.
        """
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval: str,
        recent_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data for a cryptocurrency.

        Args:
            symbol (str): The symbol of the cryptocurrency (e.g., 'BTCUSDT').
            start_date (datetime.datetime): The start date for the historical data.
            end_date (datetime.datetime): The end date for the historical data.
            interval (str): The time interval for data points (e.g., '1m', '1h', '1d').
            recent_only (bool): If True, ignore date parameters and fetch only the most
                recent data points available from the exchange (typically 1000 candles).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with keys 'timestamp', 'open',
                'high', 'low', 'close', 'volume' for each data point.

        Raises:
            ValueError: If historical data cannot be fetched.
        """
        pass

    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about the exchange (e.g., supported symbols, trading pairs).

        Returns:
            Dict[str, Any]: A dictionary containing exchange information (e.g., available symbols).

        Raises:
            ValueError: If exchange information cannot be fetched.
        """
        pass

    @abstractmethod
    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Get the rate limit configuration for the provider.

        Returns:
            Dict[str, Any]: A dictionary with rate limit details (e.g., requests per minute).

        Raises:
            NotImplementedError: If the provider does not support rate limit information.
        """
        pass
