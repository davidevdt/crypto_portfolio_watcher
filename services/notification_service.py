"""
Notification Service - Handles all notification triggers and delivery
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from database.models import get_session, AlertHistory, NotificationSettings
from services.technical_indicators import TechnicalIndicators
import pandas as pd

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications and alert triggers."""

    def __init__(self):
        self.last_notifications = {}  # Track last notification times to prevent spam

    def check_oversold_overbought_conditions(
        self, symbol: str, current_price: float, historical_data: pd.DataFrame
    ) -> List[str]:
        """
        Check if asset meets oversold/overbought conditions.

        Conditions:
        - Oversold: Price < ALL SMAs (20, 50, 200) AND/OR RSI < 30
        - Overbought: Price > ALL SMAs (20, 50, 200) AND/OR RSI > 70

        Returns list of alerts to send.
        """
        alerts = []

        if historical_data.empty or len(historical_data) < 200:
            return alerts

        try:
            # Calculate SMAs
            sma_20_series = TechnicalIndicators.calculate_sma(
                historical_data["close"], window=20
            )
            sma_50_series = TechnicalIndicators.calculate_sma(
                historical_data["close"], window=50
            )
            sma_200_series = TechnicalIndicators.calculate_sma(
                historical_data["close"], window=200
            )

            # Get the last values
            sma_20 = (
                sma_20_series.iloc[-1]
                if hasattr(sma_20_series, "iloc")
                else sma_20_series[-1]
            )
            sma_50 = (
                sma_50_series.iloc[-1]
                if hasattr(sma_50_series, "iloc")
                else sma_50_series[-1]
            )
            sma_200 = (
                sma_200_series.iloc[-1]
                if hasattr(sma_200_series, "iloc")
                else sma_200_series[-1]
            )

            # Calculate RSI
            rsi_series = TechnicalIndicators.calculate_rsi(
                historical_data["close"], window=14
            )
            rsi = rsi_series.iloc[-1] if hasattr(rsi_series, "iloc") else rsi_series[-1]

            # Check oversold conditions
            price_below_all_smas = (
                current_price < sma_20
                and current_price < sma_50
                and current_price < sma_200
            )
            rsi_oversold = rsi < 30

            if price_below_all_smas or rsi_oversold:
                condition_text = []
                if price_below_all_smas:
                    condition_text.append("Price below all SMAs")
                if rsi_oversold:
                    condition_text.append(f"RSI: {rsi:.1f}")

                alert_message = f"🟢 OVERSOLD: {symbol} - {' & '.join(condition_text)}"
                alerts.append(("OVERSOLD", alert_message))

            # Check overbought conditions
            price_above_all_smas = (
                current_price > sma_20
                and current_price > sma_50
                and current_price > sma_200
            )
            rsi_overbought = rsi > 70

            if price_above_all_smas or rsi_overbought:
                condition_text = []
                if price_above_all_smas:
                    condition_text.append("Price above all SMAs")
                if rsi_overbought:
                    condition_text.append(f"RSI: {rsi:.1f}")

                alert_message = (
                    f"🔴 OVERBOUGHT: {symbol} - {' & '.join(condition_text)}"
                )
                alerts.append(("OVERBOUGHT", alert_message))

        except Exception as e:
            logger.error(
                f"Error checking oversold/overbought conditions for {symbol}: {e}"
            )

        return alerts

    def check_take_profit_triggered(
        self, symbol: str, current_price: float, take_profit_levels: List
    ) -> List[str]:
        """Check if any take profit levels have been triggered."""
        alerts = []

        # Get a fresh session for this operation
        from database.models import get_session

        session = get_session()

        try:
            for tp_level in take_profit_levels:
                if tp_level.is_active and not tp_level.triggered_at:
                    if current_price >= tp_level.target_price:
                        # Get the TP level from current session to avoid lazy load issues
                        current_tp = session.merge(tp_level)

                        # Mark as triggered
                        current_tp.triggered_at = datetime.utcnow()

                        # Calculate profit with asset data
                        asset = current_tp.asset  # Now safely loaded in this session
                        profit_amount = (current_price - asset.average_buy_price) * (
                            asset.quantity * current_tp.percentage_to_sell / 100
                        )
                        profit_percentage = (
                            (current_price / asset.average_buy_price) - 1
                        ) * 100

                        alert_message = (
                            f"💰 TAKE PROFIT TRIGGERED: {symbol} - "
                            f"Target: ${current_tp.target_price:.4f}, "
                            f"Current: ${current_price:.4f}, "
                            f"Profit: ${profit_amount:.2f} ({profit_percentage:.1f}%)"
                        )

                        alerts.append(("TP_TRIGGERED", alert_message))

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error checking take profit levels for {symbol}: {e}")

        finally:
            session.close()

        return alerts

    def should_send_notification(
        self, symbol: str, alert_type: str, cooldown_minutes: int = 60
    ) -> bool:
        """Check if we should send a notification or if we're in cooldown period."""
        key = f"{symbol}_{alert_type}"
        now = datetime.utcnow()

        if key in self.last_notifications:
            last_sent = self.last_notifications[key]
            if (now - last_sent).total_seconds() < cooldown_minutes * 60:
                return False  # Still in cooldown period

        return True

    def send_notifications(self, alert_type: str, message: str, symbol: str = None):
        """Send notifications through all enabled channels."""
        try:
            from components.shared import (
                send_desktop_notification,
                send_whatsapp_notification,
            )

            # Get settings from database instead of session state for background compatibility
            settings = self.get_notification_settings_dict()

            # Send desktop notification
            if settings.get("desktop_notifications", False):
                send_desktop_notification("Crypto Portfolio Alert", message)

            # Send WhatsApp notification
            if settings.get("whatsapp_notifications", False) and settings.get(
                "whatsapp_number"
            ):
                send_whatsapp_notification(message, settings["whatsapp_number"])

            # Store in alert history
            self.store_alert_history(symbol, alert_type, message)

            # Update last notification time
            if symbol:
                key = f"{symbol}_{alert_type}"
                self.last_notifications[key] = datetime.utcnow()

            logger.info(f"Notifications sent for {alert_type}: {message}")

        except Exception as e:
            logger.error(f"Error sending notifications: {e}")

    def get_notification_settings_dict(self) -> dict:
        """Get notification settings as dictionary, compatible with background threads."""
        try:
            # First try to get from session state (if in Streamlit context)
            import streamlit as st

            if hasattr(st, "session_state") and hasattr(st.session_state, "settings"):
                return st.session_state.settings
        except:
            pass

        # Fallback to database settings for background threads
        try:
            from database.models import get_session, UserSettings

            session = get_session()

            # Get basic notification settings from database
            settings = {}

            # Query individual settings
            desktop_setting = (
                session.query(UserSettings)
                .filter_by(setting_key="desktop_notifications")
                .first()
            )
            whatsapp_setting = (
                session.query(UserSettings)
                .filter_by(setting_key="whatsapp_notifications")
                .first()
            )
            whatsapp_number = (
                session.query(UserSettings)
                .filter_by(setting_key="whatsapp_number")
                .first()
            )
            whatsapp_api_url = (
                session.query(UserSettings)
                .filter_by(setting_key="whatsapp_api_url")
                .first()
            )
            whatsapp_api_key = (
                session.query(UserSettings)
                .filter_by(setting_key="whatsapp_api_key")
                .first()
            )

            # Convert to dict
            settings["desktop_notifications"] = (
                desktop_setting.setting_value.lower() == "true"
                if desktop_setting
                else False
            )
            settings["whatsapp_notifications"] = (
                whatsapp_setting.setting_value.lower() == "true"
                if whatsapp_setting
                else False
            )
            settings["whatsapp_number"] = (
                whatsapp_number.setting_value if whatsapp_number else ""
            )
            settings["whatsapp_api_url"] = (
                whatsapp_api_url.setting_value if whatsapp_api_url else ""
            )
            settings["whatsapp_api_key"] = (
                whatsapp_api_key.setting_value if whatsapp_api_key else ""
            )

            session.close()
            return settings

        except Exception as e:
            logger.error(f"Error getting notification settings: {e}")
            # Return default settings
            return {
                "desktop_notifications": False,
                "whatsapp_notifications": False,
                "whatsapp_number": "",
                "whatsapp_api_url": "",
                "whatsapp_api_key": "",
            }

    def store_alert_history(self, symbol: str, alert_type: str, message: str):
        """Store alert in database for history tracking."""
        try:
            session = get_session()

            alert = AlertHistory(
                symbol=symbol or "SYSTEM",
                alert_type=alert_type,
                message=message,
                severity="MEDIUM",
                triggered_at=datetime.utcnow(),
            )

            session.add(alert)
            session.commit()
            session.close()

        except Exception as e:
            logger.error(f"Error storing alert history: {e}")

    def process_symbol_alerts(
        self,
        symbol: str,
        current_price: float,
        historical_data: pd.DataFrame,
        take_profit_levels: List = None,
    ):
        """Process all alert conditions for a symbol."""
        try:
            all_alerts = []

            # Check oversold/overbought conditions
            technical_alerts = self.check_oversold_overbought_conditions(
                symbol, current_price, historical_data
            )
            all_alerts.extend(technical_alerts)

            # Check take profit levels if provided
            if take_profit_levels:
                tp_alerts = self.check_take_profit_triggered(
                    symbol, current_price, take_profit_levels
                )
                all_alerts.extend(tp_alerts)

            # Send notifications for all triggered alerts
            for alert_type, message in all_alerts:
                if self.should_send_notification(symbol, alert_type):
                    self.send_notifications(alert_type, message, symbol)

        except Exception as e:
            logger.error(f"Error processing alerts for {symbol}: {e}")

    def get_notification_settings(self) -> Optional[NotificationSettings]:
        """Get notification settings from database."""
        try:
            session = get_session()
            settings = (
                session.query(NotificationSettings).filter_by(user_id="default").first()
            )
            session.close()
            return settings
        except Exception as e:
            logger.error(f"Error getting notification settings: {e}")
            return None
