"""
Settings Page - Comprehensive configuration and notifications management
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any

from components.shared import (
    format_currency,
    format_percentage,
    format_number,
    send_desktop_notification,
    send_whatsapp_notification,
    save_all_settings,
)


def show():
    """Main settings and configuration page."""
    st.title("⚙️ Settings & Configuration")

    # Settings tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔔 Notifications", "📊 Display", "🔧 Technical", "💾 Data Management"]
    )

    with tab1:
        show_notification_settings()

    with tab2:
        show_display_settings()

    with tab3:
        show_technical_settings()

    with tab4:
        show_data_management()

    # Save and Reset buttons (always visible)
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save All Settings", type="primary"):
            save_all_settings()

    with col2:
        if st.button("🔄 Reset Settings"):
            reset_all_settings()


def show_notification_settings():
    """Show notification configuration settings."""
    st.subheader("🔔 Notification Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Desktop Notifications")
        st.session_state.settings["desktop_notifications"] = st.checkbox(
            "Enable Desktop Notifications",
            value=st.session_state.settings.get("desktop_notifications", False),
            help="Show system notifications for price alerts and signals",
        )

        if st.session_state.settings["desktop_notifications"]:
            if st.button("🧪 Test Desktop Notification"):
                send_desktop_notification(
                    "Crypto Portfolio Tracker",
                    "Desktop notifications are working correctly!",
                )
                st.success("Test notification sent!")

    with col2:
        st.markdown("### WhatsApp Notifications")
        st.session_state.settings["whatsapp_notifications"] = st.checkbox(
            "Enable WhatsApp Notifications",
            value=st.session_state.settings.get("whatsapp_notifications", False),
            help="Send WhatsApp alerts (requires API setup)",
        )

        if st.session_state.settings["whatsapp_notifications"]:
            st.session_state.settings["whatsapp_number"] = st.text_input(
                "WhatsApp Number",
                value=st.session_state.settings.get("whatsapp_number", ""),
                help="Your WhatsApp number with country code (e.g., +1234567890)",
                placeholder="+1234567890",
            )

            st.markdown("**WhatsApp API Configuration**")
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.session_state.settings["whatsapp_api_url"] = st.text_input(
                    "WhatsApp API URL",
                    value=st.session_state.settings.get("whatsapp_api_url", ""),
                    help="API endpoint URL for WhatsApp service (e.g., https://api.whatsapp.example.com/send)",
                    placeholder="https://api.whatsapp.example.com/send",
                )

            with col1_2:
                st.session_state.settings["whatsapp_api_key"] = st.text_input(
                    "WhatsApp API Key",
                    value=st.session_state.settings.get("whatsapp_api_key", ""),
                    type="password",
                    help="API key or token for WhatsApp service authentication",
                )

            # Test button
            if st.button("📱 Test WhatsApp Notification"):
                if st.session_state.settings.get("whatsapp_number"):
                    send_whatsapp_notification(
                        "Test notification from Crypto Portfolio Tracker!",
                        st.session_state.settings["whatsapp_number"],
                    )
                    st.success("WhatsApp test message sent! Check status below.")
                else:
                    st.error("Please enter a WhatsApp number first.")

            st.info(
                "📋 **WhatsApp API Setup Instructions:**\n\n"
                "**Supported Services:**\n"
                "• Twilio WhatsApp API\n"
                "• WhatsApp Business API\n"
                "• Third-party WhatsApp services\n\n"
                "**Setup:**\n"
                "1. Sign up for a WhatsApp API service\n"
                "2. Get your API URL and key\n"
                "3. Enter credentials above\n"
                "4. Test to verify connection\n\n"
                "**Without API:** Messages are logged for testing"
            )

        st.markdown("### Alert Thresholds")

        st.session_state.settings["signal_change_alerts"] = st.checkbox(
            "Signal Change Alerts",
            value=st.session_state.settings.get("signal_change_alerts", True),
            help="Get notified when trading signals change",
        )


def show_display_settings():
    """Show display and UI configuration."""
    st.subheader("📊 Display Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Formatting")

        st.session_state.settings["decimal_places"] = st.slider(
            "Price Decimal Places",
            min_value=2,
            max_value=8,
            value=st.session_state.settings.get("decimal_places", 4),
            help="Number of decimal places to show for prices",
        )

        st.markdown("### Auto Refresh")
        st.session_state.settings["auto_refresh"] = st.checkbox(
            "Enable Auto Refresh",
            value=st.session_state.settings.get("auto_refresh", True),
            help="Automatically refresh prices at regular intervals",
        )

        if st.session_state.settings["auto_refresh"]:
            st.session_state.settings["refresh_interval"] = st.slider(
                "Refresh Interval (seconds)",
                min_value=30,
                max_value=600,
                value=st.session_state.settings.get("refresh_interval", 300),
                step=30,
                help="How often to automatically update prices",
            )

            st.session_state.settings["background_refresh_interval"] = st.slider(
                "Background Data Refresh (seconds)",
                min_value=30,
                max_value=600,
                value=st.session_state.settings.get("background_refresh_interval", 60),
                step=30,
                help="How often the background process fetches data from providers",
            )

    with col2:

        st.markdown("### Data Display")
        show_portfolio_in_charts = st.checkbox(
            "Show Portfolio Column in Asset Tables",
            value=True,
            help="Display which portfolio each asset belongs to",
        )

        group_small_allocations = st.checkbox(
            "Group Small Allocations as 'Other'",
            value=True,
            help="Group allocations smaller than 5% into 'Other' category in pie charts",
        )

        other_threshold = st.slider(
            "'Other' Threshold (%)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Assets with allocation below this percentage are grouped as 'Other'",
        )


def show_technical_settings():
    """Show technical analysis configuration."""
    st.subheader("🔧 Technical Analysis Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Moving Averages")
        default_sma_periods = st.text_input(
            "Default SMA Periods",
            value="20, 50, 100, 150, 200",
            help="Comma-separated list of SMA periods for charts",
        )

        default_ema_periods = st.text_input(
            "Default EMA Periods",
            value="12, 26, 50, 100, 200",
            help="Comma-separated list of EMA periods for charts",
        )

        st.markdown("### Oscillators")
        default_rsi_period = st.number_input(
            "Default RSI Period",
            min_value=5,
            max_value=50,
            value=14,
            help="Default period for RSI calculations",
        )

        rsi_overbought = st.slider(
            "RSI Overbought Level",
            min_value=60,
            max_value=90,
            value=70,
            help="RSI level considered overbought",
        )

        rsi_oversold = st.slider(
            "RSI Oversold Level",
            min_value=10,
            max_value=40,
            value=30,
            help="RSI level considered oversold",
        )

    with col2:
        st.markdown("### MACD Settings")
        macd_fast = st.number_input(
            "MACD Fast Period",
            min_value=5,
            max_value=30,
            value=12,
            help="Fast EMA period for MACD",
        )

        macd_slow = st.number_input(
            "MACD Slow Period",
            min_value=15,
            max_value=50,
            value=26,
            help="Slow EMA period for MACD",
        )

        macd_signal = st.number_input(
            "MACD Signal Period",
            min_value=5,
            max_value=20,
            value=9,
            help="Signal line period for MACD",
        )

        st.markdown("### Bollinger Bands")
        bb_period = st.number_input(
            "BB Period",
            min_value=10,
            max_value=50,
            value=20,
            help="Period for Bollinger Bands calculation",
        )

        bb_std = st.number_input(
            "BB Standard Deviation",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Standard deviation multiplier for BB",
        )


def show_data_management():
    """Show simplified data management and maintenance options."""
    st.subheader("💾 Data Management")

    # Show basic stats
    try:
        cached_prices_count = len(st.session_state.current_prices)
        historical_cache_count = len(getattr(st.session_state, "historical_data", {}))
        active_alerts = [a for a in st.session_state.alerts if a.get("active", True)]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cached Prices", cached_prices_count)
        with col2:
            st.metric("Historical Data", historical_cache_count)
        with col3:
            st.metric("Active Alerts", len(active_alerts))

    except Exception as e:
        st.warning(f"Could not load data stats: {e}")

    # Simple maintenance buttons
    st.markdown("### 🧹 Cache Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear All Cache", help="Clear price cache and historical data"):
            st.session_state.current_prices = {}
            st.session_state.historical_data = {}
            st.session_state.alerts = []
            st.success("✅ All cache cleared!")
            st.rerun()

    with col2:
        if st.button("🔄 Refresh Data", help="Reload fresh price data"):
            try:
                from components.shared import update_prices

                update_prices()
                st.success("✅ Data refreshed!")
            except Exception as e:
                st.error(f"❌ Error refreshing data: {e}")

    # Database location configuration
    st.markdown("### 📁 Database Location")
    show_database_location_settings()

    # Database operations
    st.markdown("### 🗄️ Database Operations")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Show Database Stats"):
            show_database_stats()

    with col2:
        if st.button("🧹 Clean Empty Portfolios"):
            try:
                deleted_count = (
                    st.session_state.portfolio_manager.clean_empty_portfolios()
                )
                if deleted_count > 0:
                    st.success(f"✅ Cleaned {deleted_count} empty portfolios")
                else:
                    st.info("ℹ️ No empty portfolios to clean")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Dangerous operations - simplified
    st.markdown("---")
    st.markdown("### ⚠️ **Danger Zone**")

    # Simple delete all records button
    st.warning(
        "⚠️ **Delete All Records** will permanently remove all portfolios, transactions, and data!"
    )

    confirm_delete = st.checkbox("I understand this will delete ALL data permanently")

    if confirm_delete:
        confirmation_text = st.text_input(
            "Type 'DELETE ALL' to confirm:",
            help="Type exactly 'DELETE ALL' to enable deletion",
        )

        if confirmation_text == "DELETE ALL":
            if st.button("🗑️ **Delete All Records from Database**", type="primary"):
                wipe_entire_database()
        else:
            st.info("📝 Type 'DELETE ALL' to enable the delete button")


def save_all_settings():
    """Save all settings to database using the new comprehensive persistence system."""
    from components.shared import save_all_settings_to_database

    try:
        # Use the new comprehensive settings save system
        save_all_settings_to_database()
        saved_count = len(st.session_state.settings)
        st.success(f"✅ Saved {saved_count} settings successfully to database!")

        # Send notification if desktop notifications are enabled
        if st.session_state.settings.get("desktop_notifications"):
            send_desktop_notification(
                "Settings Saved", f"Successfully saved {saved_count} settings"
            )

    except Exception as e:
        st.error(f"❌ Error saving settings: {e}")


def reset_all_settings():
    """Reset all settings to default values."""
    try:
        # Define default settings values
        default_settings = {
            # Notification settings
            "desktop_notifications": False,
            "whatsapp_notifications": False,
            "whatsapp_number": "",
            "whatsapp_api_url": "",
            "whatsapp_api_key": "",
            "signal_change_alerts": True,
            # Display settings
            "decimal_places": 4,
            "auto_refresh": True,
            "refresh_interval": 300,
            "background_refresh_interval": 60,
            # Technical analysis settings (default values from Asset Charts)
            "chart_rsi_period": 14,
            "chart_macd_fast": 12,
            "chart_macd_slow": 26,
            "chart_macd_signal": 9,
            "chart_stoch_k": 14,
            "chart_stoch_d": 3,
            "chart_williams_period": 14,
            "chart_bb_period": 20,
            "chart_bb_std": 2.0,
            "chart_volume_ma_period": 20,
            "chart_num_smas": 2,
            "chart_num_emas": 2,
            "chart_sma_period_0": 20,
            "chart_sma_period_1": 50,
            "chart_sma_period_2": 100,
            "chart_sma_period_3": 150,
            "chart_sma_period_4": 200,
            "chart_ema_period_0": 12,
            "chart_ema_period_1": 26,
            "chart_ema_period_2": 50,
            "chart_ema_period_3": 100,
            "chart_ema_period_4": 200,
            # Monitoring settings (from monitoring page defaults)
            "volatility_short_periods": 20,
            "volatility_long_periods": 50,
            "volatility_low_threshold": 0.02,
            "volatility_high_threshold": 0.05,
            "ema_trend_periods": 20,
            "sma_trend_periods": 50,
            "macd_trend_periods": 12,
            # Watchlist settings
            "watchlist_volatility_short_periods": 20,
            "watchlist_volatility_long_periods": 50,
            "watchlist_volatility_low_threshold": 0.02,
            "watchlist_volatility_high_threshold": 0.05,
            "watchlist_ema_trend_periods": 20,
            "watchlist_sma_trend_periods": 50,
            "watchlist_macd_trend_periods": 12,
            # Database settings
            "database_folder": "db_data",
        }

        # Reset settings in session state
        if "settings" not in st.session_state:
            st.session_state.settings = {}
        st.session_state.settings.update(default_settings)

        # Also reset chart_settings if it exists
        if "chart_settings" in st.session_state:
            default_chart_settings = {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "stoch_k": 14,
                "stoch_d": 3,
                "williams_period": 14,
                "bb_period": 20,
                "bb_std": 2.0,
                "volume_ma_period": 20,
                "num_smas": 2,
                "num_emas": 2,
                "sma_period_0": 20,
                "sma_period_1": 50,
                "sma_period_2": 100,
                "sma_period_3": 150,
                "sma_period_4": 200,
                "ema_period_0": 12,
                "ema_period_1": 26,
                "ema_period_2": 50,
                "ema_period_3": 100,
                "ema_period_4": 200,
            }
            st.session_state.chart_settings.update(default_chart_settings)

        # Save defaults to database
        from components.shared import save_all_settings_to_database

        save_all_settings_to_database()

        # Clear relevant caches to force refresh with new settings
        if hasattr(st.session_state, "historical_data"):
            st.session_state.historical_data.clear()

        reset_count = len(default_settings)
        st.success(
            f"✅ Reset {reset_count} settings to default values and saved to database!"
        )

        # Send notification if desktop notifications were enabled before reset
        try:
            send_desktop_notification(
                "Settings Reset", f"All settings have been reset to default values"
            )
        except:
            pass  # Ignore if notifications fail after reset

        # Force page rerun to reflect changes
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error resetting settings: {e}")


def show_database_stats():
    """Show database statistics and information."""
    try:
        # Get database statistics
        portfolios = st.session_state.portfolio_manager.get_all_portfolios()
        all_assets = st.session_state.portfolio_manager.get_all_assets()
        profit_history = st.session_state.portfolio_manager.get_profit_history()
        watchlist = st.session_state.portfolio_manager.get_watchlist()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Portfolios", len(portfolios))

        with col2:
            st.metric("Assets", len(all_assets))

        with col3:
            st.metric("Profit Records", len(profit_history))

        with col4:
            st.metric("Watchlist Items", len(watchlist))

        # Additional stats
        st.markdown("### 📊 Detailed Statistics")

        if all_assets:
            unique_symbols = len(set(asset.symbol for asset in all_assets))
            st.write(f"**Unique Symbols:** {unique_symbols}")

            total_value = sum(
                asset.quantity
                * st.session_state.current_prices.get(
                    asset.symbol, asset.average_buy_price
                )
                for asset in all_assets
            )
            st.write(f"**Total Portfolio Value:** {format_currency(total_value)}")

        if profit_history:
            total_realized = sum(p.realized_profit for p in profit_history)
            st.write(f"**Total Realized Profit:** {format_currency(total_realized)}")

    except Exception as e:
        st.error(f"Error getting database statistics: {e}")


def wipe_entire_database():
    """Completely wipe the entire database and all data."""
    try:
        st.warning("💲 Starting complete database wipe...")

        # Clear session state first
        st.session_state.current_prices = {}
        st.session_state.alerts = []
        if hasattr(st.session_state, "historical_data"):
            st.session_state.historical_data.clear()

        # Get database session
        from database.models import (
            get_session,
            Portfolio,
            Asset,
            Transaction,
            ProfitHistory,
            TakeProfitLevel,
            Watchlist,
            UserSettings,
            CachedPrice,
            TrackedAsset,
            HistoricalPrice,
            PortfolioValueHistory,
            AlertHistory,
            NotificationSettings,
            TechnicalIndicatorCache,
        )

        session = get_session()

        try:
            # Delete all data from all tables
            tables_to_clear = [
                TechnicalIndicatorCache,
                AlertHistory,
                NotificationSettings,
                PortfolioValueHistory,
                HistoricalPrice,
                TrackedAsset,
                CachedPrice,
                UserSettings,
                TakeProfitLevel,
                ProfitHistory,
                Transaction,
                Asset,
                Portfolio,
                Watchlist,
            ]

            deleted_counts = {}

            for table in tables_to_clear:
                count = session.query(table).count()
                session.query(table).delete()
                deleted_counts[table.__name__] = count

            session.commit()

            # Show what was deleted
            st.success("✅ Database completely wiped!")
            st.info("📈 **Deleted Records:**")

            for table_name, count in deleted_counts.items():
                if count > 0:
                    st.write(f"• {table_name}: {count} records")

            # Reinitialize with default data
            st.info("🔄 Creating default portfolio...")
            from database.models import create_database

            create_database()

            st.success("✅ Database wiped and reinitialized successfully!")
            st.info(
                "♾️ **Please restart the application** to ensure proper initialization."
            )

            # Clear session state completely
            for key in list(st.session_state.keys()):
                del st.session_state[key]

            st.balloons()

        except Exception as e:
            session.rollback()
            st.error(f"❌ Error during database wipe: {e}")
            raise

        finally:
            session.close()

    except Exception as e:
        st.error(f"❌ Critical error during database wipe: {e}")
        st.info(
            "⚠️ You may need to manually delete the database file and restart the application."
        )


def show_database_location_settings():
    """Show database location configuration settings."""
    try:
        from database.models import get_database_info

        # Get current database info
        db_info = get_database_info()

        # Current database info display
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Current Database Folder:** `{db_info['database_folder']}`")
            if db_info["db_exists"]:
                st.success(f"**Database Size:** {db_info['db_size_human']}")
            else:
                st.warning("Database not yet created")

        with col2:
            st.code(f"Location: {db_info['db_path']}", language=None)

        # Database folder setting
        current_folder = st.session_state.settings.get("database_folder", "db_data")
        new_folder = st.text_input(
            "Database Folder Name",
            value=current_folder,
            help="Folder name where the database will be stored (relative to project root)",
            key="database_folder_input",
        )

        # Validate folder name
        import re
        import os

        if new_folder:
            # Check for valid folder name
            if not re.match(r"^[a-zA-Z0-9_-]+$", new_folder):
                st.error(
                    "❌ Folder name can only contain letters, numbers, hyphens, and underscores"
                )
                return

            # Check if folder name changed
            if new_folder != current_folder:
                st.warning(
                    "⚠️ **Changing database location requires application restart**"
                )

                # Show what will happen
                st.info(f"Database will be moved to: `{new_folder}/portfolio.db`")

                # Migration option if current database exists
                if db_info["db_exists"] or db_info["legacy_db_exists"]:
                    migrate_data = st.checkbox(
                        "Migrate existing database to new location",
                        value=True,
                        help="Copy current database to the new location",
                    )
                else:
                    migrate_data = False

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("💾 Apply New Location", type="primary"):
                        try:
                            # Update setting
                            st.session_state.settings["database_folder"] = new_folder

                            # Create new folder
                            project_root = os.path.dirname(
                                os.path.dirname(os.path.dirname(__file__))
                            )
                            new_folder_path = os.path.join(project_root, new_folder)
                            os.makedirs(new_folder_path, exist_ok=True)

                            # Handle migration
                            if migrate_data and (
                                db_info["db_exists"] or db_info["legacy_db_exists"]
                            ):
                                import shutil

                                # Determine source database
                                if db_info["db_exists"]:
                                    source_db = db_info["db_path"]
                                else:
                                    source_db = os.path.join(
                                        project_root, "portfolio.db"
                                    )

                                dest_db = os.path.join(new_folder_path, "portfolio.db")

                                # Copy database files
                                if os.path.exists(source_db):
                                    shutil.copy2(source_db, dest_db)
                                    st.success(f"✅ Database migrated to {new_folder}/")

                                # Copy WAL and SHM files if they exist
                                for suffix in ["-wal", "-shm"]:
                                    source_file = source_db + suffix
                                    dest_file = dest_db + suffix
                                    if os.path.exists(source_file):
                                        shutil.copy2(source_file, dest_file)

                            st.success("✅ Database location updated!")
                            st.info(
                                "🔄 **Please restart the application** for changes to take effect"
                            )

                        except Exception as e:
                            st.error(f"❌ Error updating database location: {e}")

                with col2:
                    if st.button("❌ Cancel"):
                        st.rerun()

        # Legacy database migration option
        if db_info["legacy_db_exists"] and not db_info["db_exists"]:
            st.markdown("---")
            st.warning("🔄 **Legacy Database Detected**")
            st.info(f"Found database in project root: `portfolio.db`")

            if st.button(
                "📁 Migrate to Current Folder",
                help="Move legacy database to current database folder",
            ):
                try:
                    import shutil

                    project_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(__file__))
                    )
                    legacy_path = os.path.join(project_root, "portfolio.db")

                    # Create target folder
                    os.makedirs(db_info["db_folder_path"], exist_ok=True)

                    # Copy database
                    shutil.copy2(legacy_path, db_info["db_path"])

                    # Copy associated files
                    for suffix in ["-wal", "-shm"]:
                        legacy_file = legacy_path + suffix
                        target_file = db_info["db_path"] + suffix
                        if os.path.exists(legacy_file):
                            shutil.copy2(legacy_file, target_file)

                    st.success("✅ Legacy database migrated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error migrating legacy database: {e}")

    except Exception as e:
        st.error(f"❌ Error loading database info: {e}")
