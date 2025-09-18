import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for cryptocurrency price data.

    Provides static methods for calculating various technical analysis indicators
    including moving averages, RSI, MACD, Bollinger Bands, and more.
    """

    @staticmethod
    def calculate_sma(prices: List[float], window: int) -> List[float]:
        """
        Calculate Simple Moving Average.

        Args:
            prices: List of price values
            window: Period for SMA calculation

        Returns:
            List of SMA values
        """
        if len(prices) < window:
            return [np.nan] * len(prices)

        df = pd.DataFrame({"price": prices})
        sma = df["price"].rolling(window=window).mean()
        return sma.tolist()

    @staticmethod
    def calculate_ema(prices: List[float], window: int) -> List[float]:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: List of price values
            window: Period for EMA calculation

        Returns:
            List of EMA values
        """
        if len(prices) < window:
            return [np.nan] * len(prices)

        df = pd.DataFrame({"price": prices})
        ema = df["price"].ewm(span=window, adjust=False).mean()
        return ema.tolist()

    @staticmethod
    def calculate_rsi(prices: List[float], window: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index.

        Args:
            prices: List of price values
            window: Period for RSI calculation (default: 14)

        Returns:
            List of RSI values (0-100)
        """
        if len(prices) < window + 1:
            return [np.nan] * len(prices)

        df = pd.DataFrame({"price": prices})
        delta = df["price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.tolist()

    @staticmethod
    def calculate_macd(
        prices: List[float],
        fast_window: int = 12,
        slow_window: int = 26,
        signal_window: int = 9,
    ) -> Dict[str, List[float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: List of price values
            fast_window: Fast EMA period (default: 12)
            slow_window: Slow EMA period (default: 26)
            signal_window: Signal line EMA period (default: 9)

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' lists
        """
        if len(prices) < max(fast_window, slow_window, signal_window):
            nan_list = [np.nan] * len(prices)
            return {"macd": nan_list, "signal": nan_list, "histogram": nan_list}

        df = pd.DataFrame({"price": prices})

        ema_fast = df["price"].ewm(span=fast_window, adjust=False).mean()
        ema_slow = df["price"].ewm(span=slow_window, adjust=False).mean()

        macd_line = ema_fast - ema_slow

        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        histogram = macd_line - signal_line

        return {
            "macd": macd_line.tolist(),
            "signal": signal_line.tolist(),
            "histogram": histogram.tolist(),
        }

    @staticmethod
    def calculate_bollinger_bands(
        prices: List[float], window: int = 20, std_dev: float = 2
    ) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: List of price values
            window: Period for moving average (default: 20)
            std_dev: Standard deviation multiplier (default: 2)

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' band lists
        """
        if len(prices) < window:
            nan_list = [np.nan] * len(prices)
            return {"upper": nan_list, "middle": nan_list, "lower": nan_list}

        df = pd.DataFrame({"price": prices})

        middle = df["price"].rolling(window=window).mean()

        std = df["price"].rolling(window=window).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            "upper": upper.tolist(),
            "middle": middle.tolist(),
            "lower": lower.tolist(),
        }

    @staticmethod
    def calculate_stochastic(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        k_window: int = 14,
        d_window: int = 3,
    ) -> Tuple[List[float], List[float]]:
        """Calculate Stochastic Oscillator (%K and %D)."""
        if len(closes) < k_window:
            nan_list = [np.nan] * len(closes)
            return nan_list, nan_list

        df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

        # Calculate %K
        lowest_low = df["low"].rolling(window=k_window).min()
        highest_high = df["high"].rolling(window=k_window).max()
        k_percent = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)

        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_window).mean()

        return k_percent.tolist(), d_percent.tolist()

    @staticmethod
    def calculate_williams_r(
        highs: List[float], lows: List[float], closes: List[float], window: int = 14
    ) -> List[float]:
        """Calculate Williams %R."""
        if len(closes) < window:
            return [np.nan] * len(closes)

        df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

        # Calculate Williams %R
        highest_high = df["high"].rolling(window=window).max()
        lowest_low = df["low"].rolling(window=window).min()
        williams_r = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

        return williams_r.tolist()

    @staticmethod
    def calculate_volatility_ratio(
        prices: List[float], short_window: int = 10, long_window: int = 30
    ) -> float:
        """Calculate volatility ratio (short-term vs long-term volatility)."""
        if len(prices) < long_window:
            return np.nan

        df = pd.DataFrame({"price": prices})

        # Calculate returns
        returns = df["price"].pct_change().dropna()

        if len(returns) < long_window:
            return np.nan

        # Calculate short-term volatility (recent)
        short_term_vol = returns.tail(short_window).std() * np.sqrt(252)  # Annualized

        # Calculate long-term volatility (historical average)
        long_term_vol = returns.tail(long_window).std() * np.sqrt(252)  # Annualized

        if long_term_vol == 0 or np.isnan(long_term_vol):
            return np.nan

        # Return ratio: >1 means higher recent volatility, <1 means lower recent volatility
        return short_term_vol / long_term_vol

    @staticmethod
    def calculate_volatility_details(
        prices: List[float], short_window: int = 10, long_window: int = 30
    ) -> Dict[str, float]:
        """Calculate detailed volatility metrics (short-term, long-term, and ratio)."""
        if len(prices) < long_window:
            return {
                "short_term_vol": np.nan,
                "long_term_vol": np.nan,
                "volatility_ratio": np.nan,
            }

        df = pd.DataFrame({"price": prices})

        # Calculate returns
        returns = df["price"].pct_change().dropna()

        if len(returns) < long_window:
            return {
                "short_term_vol": np.nan,
                "long_term_vol": np.nan,
                "volatility_ratio": np.nan,
            }

        # Calculate short-term volatility (recent)
        short_term_vol = returns.tail(short_window).std() * np.sqrt(252)  # Annualized

        # Calculate long-term volatility (historical average)
        long_term_vol = returns.tail(long_window).std() * np.sqrt(252)  # Annualized

        if long_term_vol == 0 or np.isnan(long_term_vol):
            volatility_ratio = np.nan
        else:
            volatility_ratio = short_term_vol / long_term_vol

        return {
            "short_term_vol": short_term_vol,
            "long_term_vol": long_term_vol,
            "volatility_ratio": volatility_ratio,
        }

    @staticmethod
    def get_volatility_status(
        volatility_ratio: float, low_threshold: float = 0.7, high_threshold: float = 1.3
    ) -> Dict[str, str]:
        """Get volatility status with color coding and configurable thresholds."""
        if np.isnan(volatility_ratio):
            return {"status": "N/A", "color": "gray", "emoji": "⚫"}

        # Configurable thresholds for volatility classification
        if (
            volatility_ratio < low_threshold
        ):  # Recent volatility much lower than average
            return {"status": "Low", "color": "green", "emoji": "🟢"}
        elif volatility_ratio < high_threshold:  # Recent volatility similar to average
            return {"status": "Moderate", "color": "yellow", "emoji": "🟡"}
        else:  # Recent volatility much higher than average
            return {"status": "High", "color": "red", "emoji": "🔴"}

    @staticmethod
    def get_current_indicators(
        prices: List[float], volumes: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Get current (latest) values of all indicators."""
        if len(prices) < 50:  # Need enough data for reliable indicators
            return {
                "rsi": np.nan,
                "sma_20": np.nan,
                "sma_50": np.nan,
                "ema_12": np.nan,
                "ema_26": np.nan,
                "macd": np.nan,
                "macd_signal": np.nan,
                "macd_histogram": np.nan,
                "bb_upper": np.nan,
                "bb_middle": np.nan,
                "bb_lower": np.nan,
                "volatility_ratio": np.nan,
                "current_price": prices[-1] if len(prices) > 0 else np.nan,
            }

        # Calculate all indicators
        rsi = TechnicalIndicators.calculate_rsi(prices, 14)
        sma_20 = TechnicalIndicators.calculate_sma(prices, 20)
        sma_50 = TechnicalIndicators.calculate_sma(prices, 50)
        ema_12 = TechnicalIndicators.calculate_ema(prices, 12)
        ema_26 = TechnicalIndicators.calculate_ema(prices, 26)
        macd_data = TechnicalIndicators.calculate_macd(prices)
        bb_data = TechnicalIndicators.calculate_bollinger_bands(prices)
        volatility_ratio = TechnicalIndicators.calculate_volatility_ratio(
            prices, 10, 30
        )

        return {
            "rsi": rsi[-1] if not np.isnan(rsi[-1]) else np.nan,
            "sma_20": sma_20[-1] if not np.isnan(sma_20[-1]) else np.nan,
            "sma_50": sma_50[-1] if not np.isnan(sma_50[-1]) else np.nan,
            "ema_12": ema_12[-1] if not np.isnan(ema_12[-1]) else np.nan,
            "ema_26": ema_26[-1] if not np.isnan(ema_26[-1]) else np.nan,
            "macd": (
                macd_data["macd"][-1] if not np.isnan(macd_data["macd"][-1]) else np.nan
            ),
            "macd_signal": (
                macd_data["signal"][-1]
                if not np.isnan(macd_data["signal"][-1])
                else np.nan
            ),
            "macd_histogram": (
                macd_data["histogram"][-1]
                if not np.isnan(macd_data["histogram"][-1])
                else np.nan
            ),
            "bb_upper": (
                bb_data["upper"][-1] if not np.isnan(bb_data["upper"][-1]) else np.nan
            ),
            "bb_middle": (
                bb_data["middle"][-1] if not np.isnan(bb_data["middle"][-1]) else np.nan
            ),
            "bb_lower": (
                bb_data["lower"][-1] if not np.isnan(bb_data["lower"][-1]) else np.nan
            ),
            "volatility_ratio": (
                volatility_ratio if not np.isnan(volatility_ratio) else np.nan
            ),
            "current_price": prices[-1] if len(prices) > 0 else np.nan,
        }

    @staticmethod
    def get_trading_signal(
        indicators: Dict[str, float], config: Dict = None
    ) -> Dict[str, str]:
        """Determine trading signals based on technical indicators with configurable parameters."""
        if config is None:
            config = {
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "sma_short_key": "sma_20",
                "sma_long_key": "sma_50",
            }

        signals = {
            "overall": "NEUTRAL",
            "rsi_signal": "NEUTRAL",
            "macd_signal": "NEUTRAL",
            "moving_average_signal": "NEUTRAL",
            "color": "yellow",  # neutral
        }

        current_price = indicators.get("current_price", 0)
        rsi = indicators.get("rsi", 50)
        sma_short_key = config.get("sma_short_key", "sma_20")
        sma_long_key = config.get("sma_long_key", "sma_50")
        sma_short = indicators.get(sma_short_key, 0)
        sma_long = indicators.get(sma_long_key, 0)
        macd = indicators.get("macd", 0)
        macd_signal_val = indicators.get("macd_signal", 0)

        # RSI signals with configurable thresholds
        rsi_overbought = config.get("rsi_overbought", 70)
        rsi_oversold = config.get("rsi_oversold", 30)
        if not np.isnan(rsi):
            if rsi > rsi_overbought:
                signals["rsi_signal"] = "SELL"
            elif rsi < rsi_oversold:
                signals["rsi_signal"] = "BUY"

        # Moving Average signals with configurable keys
        if not np.isnan(sma_short) and not np.isnan(sma_long) and current_price > 0:
            if current_price > sma_short and current_price > sma_long:
                signals["moving_average_signal"] = "SELL"
            elif current_price < sma_short and current_price < sma_long:
                signals["moving_average_signal"] = "BUY"

        # MACD signals
        if not np.isnan(macd) and not np.isnan(macd_signal_val):
            if macd > macd_signal_val and macd > 0:
                signals["macd_signal"] = "SELL"
            elif macd < macd_signal_val and macd < 0:
                signals["macd_signal"] = "BUY"

        # Overall signal based on requirements:
        # Red: Price above both moving averages and RSI above 70
        # Green: Price below both moving averages and RSI below 30
        if (
            signals["rsi_signal"] == "SELL"
            and signals["moving_average_signal"] == "SELL"
        ):
            signals["overall"] = "SELL"
            signals["color"] = "red"
        elif (
            signals["rsi_signal"] == "BUY" and signals["moving_average_signal"] == "BUY"
        ):
            signals["overall"] = "BUY"
            signals["color"] = "green"

        return signals

    @staticmethod
    def calculate_fibonacci_retracement(
        highs: List[float], lows: List[float], lookback_period: int = 50
    ) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels based on recent swing high and low."""
        if len(highs) < lookback_period or len(lows) < lookback_period:
            return {}

        # Get recent data for swing calculation
        recent_highs = highs[-lookback_period:]
        recent_lows = lows[-lookback_period:]

        # Find swing high and low
        swing_high = max(recent_highs)
        swing_low = min(recent_lows)

        # Calculate the range
        price_range = swing_high - swing_low

        if price_range == 0:
            return {}

        # Standard Fibonacci retracement levels
        fib_levels = {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "fib_0": swing_high,  # 0% (swing high)
            "fib_23_6": swing_high - (price_range * 0.236),  # 23.6%
            "fib_38_2": swing_high - (price_range * 0.382),  # 38.2%
            "fib_50": swing_high - (price_range * 0.5),  # 50%
            "fib_61_8": swing_high - (price_range * 0.618),  # 61.8%
            "fib_78_6": swing_high - (price_range * 0.786),  # 78.6%
            "fib_100": swing_low,  # 100% (swing low)
            # Extension levels
            "fib_127_2": swing_high - (price_range * 1.272),  # 127.2%
            "fib_161_8": swing_high - (price_range * 1.618),  # 161.8%
        }

        return fib_levels

    @staticmethod
    def prepare_chart_data(
        historical_data: List[Dict], indicators_config: Dict
    ) -> Dict:
        """Prepare data for TradingView-style charts with selected indicators."""
        if not historical_data:
            return {}

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Extract OHLCV data
        timestamps = df["timestamp"].tolist()
        opens = df["open"].tolist()
        highs = df["high"].tolist()
        lows = df["low"].tolist()
        closes = df["close"].tolist()
        volumes = df["volume"].tolist()

        chart_data = {
            "timestamps": timestamps,
            "ohlc": {"open": opens, "high": highs, "low": lows, "close": closes},
            "volume": volumes,
            "indicators": {},
        }

        # Calculate requested indicators
        if indicators_config.get("sma_enabled"):
            for window in indicators_config.get("sma_windows", [20]):
                sma = TechnicalIndicators.calculate_sma(closes, window)
                chart_data["indicators"][f"SMA_{window}"] = sma

        if indicators_config.get("ema_enabled"):
            for window in indicators_config.get("ema_windows", [12, 26]):
                ema = TechnicalIndicators.calculate_ema(closes, window)
                chart_data["indicators"][f"EMA_{window}"] = ema

        if indicators_config.get("rsi_enabled"):
            rsi_window = indicators_config.get("rsi_window", 14)
            rsi = TechnicalIndicators.calculate_rsi(closes, rsi_window)
            chart_data["indicators"][f"RSI_{rsi_window}"] = rsi

        if indicators_config.get("macd_enabled"):
            fast = indicators_config.get("macd_fast", 12)
            slow = indicators_config.get("macd_slow", 26)
            signal = indicators_config.get("macd_signal", 9)
            macd_data = TechnicalIndicators.calculate_macd(closes, fast, slow, signal)
            chart_data["indicators"]["MACD"] = macd_data["macd"]
            chart_data["indicators"]["MACD_Signal"] = macd_data["signal"]
            chart_data["indicators"]["MACD_Histogram"] = macd_data["histogram"]

        if indicators_config.get("bb_enabled"):
            bb_window = indicators_config.get("bb_window", 20)
            bb_std = indicators_config.get("bb_std", 2)
            bb_data = TechnicalIndicators.calculate_bollinger_bands(
                closes, bb_window, bb_std
            )
            chart_data["indicators"]["BB_Upper"] = bb_data["upper"]
            chart_data["indicators"]["BB_Middle"] = bb_data["middle"]
            chart_data["indicators"]["BB_Lower"] = bb_data["lower"]

        fib_lookback = indicators_config.get("fib_lookback", 50)
        fib_levels = TechnicalIndicators.calculate_fibonacci_retracement(
            highs, lows, fib_lookback
        )
        chart_data["fibonacci"] = fib_levels

        return chart_data
