"""
Portfolio Overview Page - Enhanced portfolio management and visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

from components.shared import (
    smart_price_refresh,
    format_currency,
    format_percentage,
    format_crypto_quantity,
    format_smart_currency,
    format_smart_quantity,
    create_portfolio_allocation_chart,
    create_performance_bar_chart,
    poll_database_continuously,
    refresh_portfolio_data_after_operation,
    is_operation_in_progress,
    set_operation_in_progress,
    show_empty_state,
    show_empty_chart,
)
from services.background_data_service import background_service


def show_portfolio_24h_change(summary: Dict, selected_portfolio):
    """
    Show 24h portfolio change based on actual asset price changes.

    Args:
        summary: Portfolio summary dictionary
        selected_portfolio: Selected portfolio ID or 'all'
    """
    try:
        from database.models import get_session, PortfolioValueHistory, HistoricalPrice

        current_value = summary.get("total_value", 0)

        session = get_session()
        yesterday = datetime.now() - timedelta(days=1)
        portfolio_history_value = None

        if selected_portfolio == "all":
            yesterday_records = (
                session.query(PortfolioValueHistory)
                .filter(
                    PortfolioValueHistory.date >= yesterday - timedelta(hours=12),
                    PortfolioValueHistory.date <= yesterday + timedelta(hours=12),
                )
                .all()
            )
            portfolio_history_value = (
                sum(record.total_value for record in yesterday_records)
                if yesterday_records
                else None
            )
        else:
            yesterday_record = (
                session.query(PortfolioValueHistory)
                .filter(
                    PortfolioValueHistory.portfolio_id == selected_portfolio,
                    PortfolioValueHistory.date >= yesterday - timedelta(hours=12),
                    PortfolioValueHistory.date <= yesterday + timedelta(hours=12),
                )
                .first()
            )
            portfolio_history_value = (
                yesterday_record.total_value if yesterday_record else None
            )

        # If no portfolio history, calculate from individual asset price changes
        if portfolio_history_value is None:
            # Get assets for calculation
            if selected_portfolio == "all":
                assets = st.session_state.portfolio_manager.get_all_assets()
            else:
                assets = st.session_state.portfolio_manager.get_portfolio_assets(
                    selected_portfolio
                )

            # Calculate yesterday's portfolio value using historical prices
            yesterday_portfolio_value = 0
            for asset in assets:
                # Get yesterday's price for this asset
                yesterday_price_record = (
                    session.query(HistoricalPrice)
                    .filter(
                        HistoricalPrice.symbol == asset.symbol,
                        HistoricalPrice.interval == "1d",
                        HistoricalPrice.date >= yesterday - timedelta(hours=12),
                        HistoricalPrice.date <= yesterday + timedelta(hours=12),
                    )
                    .first()
                )

                if yesterday_price_record:
                    yesterday_asset_value = (
                        asset.quantity * yesterday_price_record.price
                    )
                    yesterday_portfolio_value += yesterday_asset_value
                else:
                    current_price = st.session_state.current_prices.get(
                        asset.symbol, asset.average_buy_price
                    )
                    yesterday_portfolio_value += asset.quantity * current_price

            portfolio_history_value = yesterday_portfolio_value

        session.close()

        if portfolio_history_value and portfolio_history_value > 0:
            change_dollar = current_value - portfolio_history_value
            change_pct = (change_dollar / portfolio_history_value) * 100

            color = "green" if change_dollar >= 0 else "red"
            sign = "+" if change_dollar >= 0 else ""

            st.markdown(
                f"""
            <span style="color: {color}; font-weight: bold;">
            24h Change: {sign}${change_dollar:,.2f} ({sign}{change_pct:.2f}%)
            </span>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='font-weight: bold; color: orange;'>24h Change: Calculating...</span>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.markdown(
            f"<span style='font-weight: bold; color: red;'>24h Change: Error ({str(e)})</span>",
            unsafe_allow_html=True,
        )


def show():
    """
    Main portfolio overview page.

    Displays portfolio overview with asset tables, charts, and management tools.
    Handles empty states for first-time users and provides comprehensive
    portfolio analytics and visualization.
    """
    st.title("📊 Portfolio Overview")

    if not hasattr(st.session_state, "current_prices"):
        st.session_state.current_prices = {}
    if not hasattr(st.session_state, "historical_data"):
        st.session_state.historical_data = {}
    if not hasattr(st.session_state, "portfolio_manager"):
        from components.shared import init_session_state

        init_session_state()

    # Portfolio selector is now handled in the main app navigation for this page
    # Get the current selected portfolio from session state
    selected_portfolio = st.session_state.get("selected_portfolio", None)

    # Debug: Check if we have any portfolios at all
    try:
        all_portfolios = st.session_state.portfolio_manager.get_all_portfolios()
        if not all_portfolios:
            # Force selected_portfolio to None when no portfolios exist
            st.session_state.selected_portfolio = None
            selected_portfolio = None
    except Exception as e:
        st.error(f"Error checking portfolios: {e}")

    if selected_portfolio is None:
        st.markdown("---")
        st.subheader("🎉 Welcome! Let's get started")
        st.markdown(
            "**Add your first crypto asset and give your portfolio a name to start tracking your investments.**"
        )

        with st.expander("💡 Examples & Ideas", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                **Popular Assets:**
                • BTC (Bitcoin)
                • ETH (Ethereum)
                • BNB (Binance Coin)
                • ADA (Cardano)
                • SOL (Solana)
                """
                )
            with col2:
                st.markdown(
                    """
                **Portfolio Names:**
                • Long-term Holdings
                • Main Portfolio
                • DeFi Investments
                • Trading Fund
                • Crypto Savings
                """
                )

        with st.form("add_first_asset_form"):
            st.markdown("### 🚀 Add Your First Asset")

            # Portfolio name input
            st.markdown("**Portfolio Name:**")
            portfolio_name = st.text_input(
                "Give your portfolio a name",
                value="Main Portfolio",
                placeholder="e.g., Long-term Holdings, Main Portfolio, Trading Fund...",
                help="Choose a meaningful name that reflects your investment goals or strategy",
            )

            st.markdown("**Asset Details:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                symbol = st.text_input(
                    "Symbol",
                    value="",
                    placeholder="e.g., BTC, ETH, ADA",
                    help="Enter the crypto symbol (e.g., BTC for Bitcoin)",
                ).upper()

            with col2:
                quantity = st.number_input(
                    "Quantity",
                    min_value=0.0,
                    step=0.000001,
                    format="%.6f",
                    value=0.0,
                    help="How much of this crypto do you own?",
                )

            with col3:
                buy_price = st.number_input(
                    "Average Buy Price ($)",
                    min_value=0.0,
                    step=0.01,
                    format="%.4f",
                    value=0.0,
                    help="The average price you paid per unit",
                )

            # Show investment summary
            if quantity > 0 and buy_price > 0:
                total_invested = quantity * buy_price
                st.info(f"💰 Total Investment: ${total_invested:,.2f}")

            # Submit button
            col1, col2 = st.columns(2)
            with col1:
                if (
                    portfolio_name.strip()
                    and symbol.strip()
                    and quantity > 0
                    and buy_price > 0
                ):
                    submit_clicked = st.form_submit_button(
                        f"🎯 Create '{portfolio_name.strip()}' & Add {symbol}",
                        type="primary",
                    )
                else:
                    submit_clicked = st.form_submit_button(
                        "Add First Asset", type="primary"
                    )

            with col2:
                # Option to just create empty portfolio
                empty_portfolio_clicked = st.form_submit_button(
                    "📁 Create Empty Portfolio Only"
                )

            # Handle form submission
            if submit_clicked:
                if not portfolio_name.strip():
                    st.error("❌ Portfolio name cannot be empty")
                elif not symbol.strip():
                    st.error("❌ Asset symbol cannot be empty")
                elif quantity <= 0:
                    st.error("❌ Quantity must be greater than 0")
                elif buy_price <= 0:
                    st.error("❌ Buy price must be greater than 0")
                else:
                    try:
                        # Create portfolio first
                        new_portfolio = (
                            st.session_state.portfolio_manager.create_portfolio(
                                portfolio_name.strip()
                            )
                        )
                        st.success(f"✅ Created portfolio '{new_portfolio.name}'!")

                        # Add the asset to the new portfolio
                        st.session_state.portfolio_manager.add_asset(
                            portfolio_id=new_portfolio.id,
                            symbol=symbol.strip(),
                            quantity=quantity,
                            buy_price=buy_price,
                        )
                        st.success(
                            f"✅ Added {quantity:,.6f} {symbol} to your portfolio!"
                        )
                        st.success(
                            f"🎉 Your crypto portfolio is now live and tracking ${quantity * buy_price:,.2f} in {symbol}!"
                        )

                        # Clear any cached portfolio data to ensure fresh reload
                        for key in list(st.session_state.keys()):
                            if (
                                "cached_portfolios" in key
                                or "portfolio_cache" in key
                                or "portfolio_selector_cache" in key
                            ):
                                del st.session_state[key]

                        time.sleep(2)  # Brief pause for user to see success
                        st.rerun()  # Refresh to show the new portfolio with asset

                    except Exception as e:
                        if "UNIQUE constraint failed" in str(e):
                            st.error(
                                f"❌ Portfolio named '{portfolio_name.strip()}' already exists. Please choose a different name."
                            )
                        else:
                            st.error(f"❌ Error creating portfolio and asset: {e}")

            elif empty_portfolio_clicked and portfolio_name.strip():
                try:
                    new_portfolio = st.session_state.portfolio_manager.create_portfolio(
                        portfolio_name.strip()
                    )
                    st.success(f"✅ Created empty portfolio '{new_portfolio.name}'!")
                    st.info(
                        "💡 You can now add assets to your portfolio using the Asset Management section."
                    )

                    # Clear any cached portfolio data to ensure fresh reload
                    for key in list(st.session_state.keys()):
                        if (
                            "cached_portfolios" in key
                            or "portfolio_cache" in key
                            or "portfolio_selector_cache" in key
                        ):
                            del st.session_state[key]

                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    if "UNIQUE constraint failed" in str(e):
                        st.error(
                            f"❌ Portfolio named '{portfolio_name.strip()}' already exists. Please choose a different name."
                        )
                    else:
                        st.error(f"❌ Error creating portfolio: {e}")
            elif empty_portfolio_clicked and not portfolio_name.strip():
                st.error("❌ Portfolio name cannot be empty")

        return  # Return after handling empty state forms

    # Get assets based on selection - ALWAYS fresh from database
    # Clear any cached assets to ensure fresh data
    for key in list(st.session_state.keys()):
        if "cached_assets" in key or "cached_portfolios" in key:
            del st.session_state[key]

    if selected_portfolio == "all":
        # Use aggregated assets for proper "All Portfolios" view
        aggregated_assets = st.session_state.portfolio_manager.get_aggregated_assets()
        assets = aggregated_assets
        portfolio_name = "All Portfolios"
        use_aggregated = True
    else:
        assets = st.session_state.portfolio_manager.get_portfolio_assets(
            selected_portfolio
        )
        portfolio = st.session_state.portfolio_manager.get_portfolio_by_id(
            selected_portfolio
        )
        portfolio_name = portfolio.name if portfolio else "Unknown Portfolio"
        use_aggregated = False

    if not assets:
        st.info(f"No assets in {portfolio_name}. Add some assets to get started!")
        show_add_asset_form(selected_portfolio)
        return

    # Database-first approach: continuous polling handles data freshness
    # No need for manual API calls - data comes from database cache
    poll_database_continuously()

    # CRITICAL: Always ensure we have fresh database data loaded on every page load
    # This ensures the dashboard displays current data immediately
    smart_price_refresh(force_refresh=True)  # Load from database cache only

    # Use session state current_prices which were just refreshed above
    consistent_prices = st.session_state.current_prices

    if selected_portfolio == "all":
        # Use aggregated summary for proper asset aggregation
        summary = (
            st.session_state.portfolio_manager.get_all_portfolios_summary_aggregated(
                consistent_prices
            )
        )
    else:
        summary = st.session_state.portfolio_manager.get_portfolio_summary(
            selected_portfolio, consistent_prices
        )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", format_currency(summary.get("total_value", 0)))
    with col2:
        st.metric("Total Invested", format_currency(summary.get("total_spent", 0)))
    with col3:
        pnl = summary.get("total_return", 0)
        pnl_pct = summary.get("total_return_percentage", 0)
        st.metric("P&L", format_currency(pnl), delta=format_percentage(pnl_pct))
    with col4:
        st.metric("Assets Count", len(assets))

    # Debug section removed for production

    # Show last refresh time
    col1, col2 = st.columns([2, 2])
    with col1:
        if "last_portfolio_price_update" in st.session_state:
            last_update = st.session_state.last_portfolio_price_update
            time_ago = int((datetime.now() - last_update).total_seconds())
            if time_ago < 60:
                st.markdown(f"**Last Refresh**: {time_ago}s ago")
            else:
                st.markdown(f"**Last Refresh**: {time_ago//60}m ago")
        else:
            st.markdown("**Last Refresh**: Never")

    with col2:
        # Show 24h portfolio change
        show_portfolio_24h_change(summary, selected_portfolio)

    # 90-day portfolio progress chart
    if use_aggregated:
        original_assets = st.session_state.portfolio_manager.get_all_assets()
        show_portfolio_progress_chart(original_assets, selected_portfolio, summary)
    else:
        show_portfolio_progress_chart(assets, selected_portfolio, summary)

    # Show portfolios summary table when "All Portfolios" is selected
    if selected_portfolio == "all":
        show_portfolios_summary_table(consistent_prices)

    # Portfolio Assets Table (with real-time updated prices)
    st.subheader("Portfolio Assets")

    show_portfolio_assets_table(assets, summary, consistent_prices, use_aggregated)

    # Charts section - Collapsible expandable box
    with st.expander("📊 Charts", expanded=False):
        # Show portfolio charts when "All Portfolios" is selected
        if selected_portfolio == "all":
            show_portfolio_charts(consistent_prices)

        # First row: Portfolio allocation pie charts
        show_allocation_charts(assets, summary, use_aggregated)

        # Second row: Asset performance bar charts
        st.subheader("Asset Performance")
        show_performance_bar_charts(assets, use_aggregated)

        # Third row: Value scatter plot
        show_value_scatter_plot(assets, use_aggregated)

    # Asset management section - Collapsible expandable box
    with st.expander("🔧 Asset Management", expanded=False):
        if selected_portfolio != "all":
            show_asset_management(assets, selected_portfolio)
        else:
            # Show asset management for "All Portfolios" view
            show_asset_management_all_portfolios()


def show_portfolio_progress_chart(assets: List, selected_portfolio, summary: Dict):
    """Show 90-day portfolio progress chart."""
    st.subheader("📈 Portfolio Progress")

    # Time period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        time_periods = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 365 days": 365,
            "Last 3 years": 1095,
            "Last 5 years": 1825,
        }

        selected_period = st.selectbox(
            "Time Period",
            list(time_periods.keys()),
            index=2,
            key=f"portfolio_period_{selected_portfolio}",
        )
        days = time_periods[selected_period]

        # Create a unique key to track period changes per portfolio
        period_tracker_key = f"period_tracker_{selected_portfolio}"

        # If period changed, clear relevant cache and force recalculation
        if (
            period_tracker_key not in st.session_state
            or st.session_state[period_tracker_key] != selected_period
        ):

            # Track the new period
            st.session_state[period_tracker_key] = selected_period

            # Clear all portfolio history cache for this portfolio (all time periods)
            keys_to_remove = []
            for key in list(st.session_state.historical_data.keys()):
                # Clear portfolio history cache for this specific portfolio
                if key.startswith(f"portfolio_{selected_portfolio}_") and key.endswith(
                    "_history"
                ):
                    keys_to_remove.append(key)
                # Also clear individual symbol portfolio cache to force fresh data
                if "_portfolio" in key and not key.endswith("_history"):
                    keys_to_remove.append(key)

            # Remove the keys
            for key in keys_to_remove:
                if key in st.session_state.historical_data:
                    del st.session_state.historical_data[key]

            # Force immediate rerun to recalculate with new period
            if keys_to_remove:
                st.rerun()

    with col2:
        # Auto-refresh enabled but no status message displayed
        pass

    # Calculate portfolio value over time using database cache
    cache_key = f"portfolio_{selected_portfolio}_{days}_history"

    # Force recalculation if cache key changed or data is missing
    if (
        cache_key not in st.session_state.historical_data
        or not st.session_state.historical_data.get(cache_key)
    ):

        with st.spinner(
            f"Calculating {days}-day portfolio history using database cache..."
        ):

            portfolio_history = calculate_portfolio_history(assets, days, summary)

            if portfolio_history:
                st.session_state.historical_data[cache_key] = portfolio_history
            else:
                st.session_state.historical_data[cache_key] = []

    portfolio_history = st.session_state.historical_data.get(cache_key, [])

    # Check if we have empty assets first
    if not assets:
        show_empty_chart(
            title="No Portfolio Data",
            message="Add some assets to your portfolio to see historical value progression.",
            height=400,
        )
        return

    if portfolio_history and len(portfolio_history) > 0:

        # Validate data structure before creating DataFrame
        if not portfolio_history:
            st.error("No portfolio history data available")
            return

        # Check if data has required columns
        required_columns = ["date", "value"]
        first_item = portfolio_history[0] if portfolio_history else {}
        missing_columns = [col for col in required_columns if col not in first_item]

        if missing_columns:
            st.error(
                f"Portfolio history data missing required columns: {missing_columns}"
            )
            st.error(f"Available columns: {list(first_item.keys())}")
            return

        try:
            df = pd.DataFrame(portfolio_history)
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            st.error(f"Error creating DataFrame from portfolio history: {e}")
            st.error(
                f"Data structure: {portfolio_history[:2] if len(portfolio_history) >= 2 else portfolio_history}"
            )
            return

        fig = go.Figure()

        # Add line plot
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#007bff", width=2),
                fill="tonexty",
                fillcolor="rgba(0, 123, 255, 0.1)",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>"
                + "Portfolio Value: $%{y:,.2f}<br>"
                + "<extra></extra>",
            )
        )

        # Add baseline at 0 for fill
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=[0] * len(df),
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=f"Portfolio Value - {selected_period}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            showlegend=False,
            hovermode="x unified",
            height=400,
        )

        st.plotly_chart(fig, width="stretch")

        # Show value change - ensure we have enough data and use period-specific values
        if len(df) >= 2:
            start_value = df.iloc[0]["value"]

            # CRITICAL: Use EXACT SAME prices as top summary to ensure perfect consistency
            # This ensures "Current Value" matches "Total Value" at the top exactly
            end_value = summary.get(
                "total_value", 0
            )  # Use the same summary calculated above

            value_change = end_value - start_value
            pct_change = (value_change / start_value * 100) if start_value > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Period Start",
                    format_currency(start_value),
                    help=f"Portfolio value at the beginning of {selected_period.lower()} ({df.iloc[0]['date'].strftime('%Y-%m-%d')})",
                )
            with col2:
                st.metric(
                    "Current Value",
                    format_currency(end_value),
                    help=f"Portfolio value as of {df.iloc[-1]['date'].strftime('%Y-%m-%d')}",
                )
            with col3:
                st.metric(
                    "Period Change",
                    format_currency(value_change),
                    delta=format_percentage(pct_change),
                    help=f"Value change over {selected_period.lower()} ({days} days)",
                )
        else:
            # Fallback for insufficient data - use same summary as top metrics
            current_value = summary.get("total_value", 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Period Start", "N/A", help="Insufficient historical data")
            with col2:
                st.metric(
                    "Current Value",
                    format_currency(current_value),
                    help="Current portfolio value",
                )
            with col3:
                st.metric("Period Change", "N/A", help="Insufficient historical data")
    else:
        show_empty_chart(
            title="No Historical Data",
            message=f"Unable to calculate portfolio history for {selected_period}. Historical price data is being collected in the background.",
            height=400,
        )


def get_historical_data_from_db(symbol: str, days: int) -> List[Dict]:
    """Get historical data from database cache, fallback to API if needed."""
    try:
        from database.models import get_session, HistoricalPrice

        session = get_session()
        end_date = datetime.now()
        _ = end_date - timedelta(days=days)  # start_date not used

        # Always get maximum available data (up to 5 years), then let filtering handle the period
        max_start_date = end_date - timedelta(days=1825)  # 5 years max
        st.write(f"DEBUG: Querying all available {symbol} data (up to 5 years)")

        # Query database for all available historical prices
        historical_prices = (
            session.query(HistoricalPrice)
            .filter(
                HistoricalPrice.symbol == symbol,
                HistoricalPrice.interval == "1d",
                HistoricalPrice.date >= max_start_date,
            )
            .order_by(HistoricalPrice.date.asc())
            .all()
        )

        session.close()

        if historical_prices:
            st.write(
                f"DEBUG: Found {len(historical_prices)} database records for {symbol}"
            )
            st.write(
                f"DEBUG: Database date range: {historical_prices[0].date.date()} to {historical_prices[-1].date.date()}"
            )

            # Convert to expected format with validation
            converted_data = []
            for price in historical_prices:
                try:
                    converted_data.append(
                        {
                            "date": price.date.strftime("%Y-%m-%d"),
                            "close": float(price.price),
                            "timestamp": int(price.date.timestamp() * 1000),
                        }
                    )
                except Exception:
                    continue
            return converted_data if converted_data else []
        else:
            st.write(f"DEBUG: No historical data found for {symbol}")
            return []

    except Exception:
        return []  # Silent fail - let the calling function handle errors


def calculate_portfolio_history(
    assets: List, days: int, summary: Dict = None
) -> List[Dict]:
    """Calculate portfolio value history using database cache with API fallback."""
    try:
        # Try background service first for efficiency, then fallback to manual calculation
        if hasattr(background_service, "calculate_portfolio_history_from_db"):
            try:
                from database.models import get_session

                session = get_session()
                db_history = background_service.calculate_portfolio_history_from_db(
                    session, assets, days, max_days=1825
                )
                session.close()

                # Be more lenient with data requirements - just need at least 2 data points
                if db_history and len(db_history) >= 2:
                    # Convert background service data format to expected format
                    converted_history = []
                    for item in db_history:
                        try:
                            # Background service returns 'total_value', but chart expects 'value'
                            converted_item = {
                                "date": item["date"],
                                "value": item.get("total_value", item.get("value", 0)),
                            }
                            converted_history.append(converted_item)
                        except Exception as e:
                            continue

                    if len(converted_history) >= 2:
                        return converted_history
            except Exception as e:
                pass  # Fallback to individual asset calculation

        # Fallback: Calculate from individual asset data
        end_date = datetime.now()
        # IMPORTANT: Always fetch 5 years of data first, then subset by selected period
        _ = end_date - timedelta(days=1825)  # full_start_date not used
        selected_start_date = end_date - timedelta(
            days=days
        )  # The actual period we want

        # Collect all symbols and quantities
        symbols = [asset.symbol for asset in assets]
        _ = {
            asset.symbol: asset.quantity for asset in assets
        }  # asset_quantities not used

        # Get historical data for each symbol from database first (always get full year)
        historical_data = {}
        for symbol in symbols:
            if symbol in ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]:
                # For stablecoins, create dummy data at $1.00 for the selected period only
                dates = pd.date_range(selected_start_date, end_date, freq="D")
                historical_data[symbol] = [
                    {"date": date.strftime("%Y-%m-%d"), "close": 1.0} for date in dates
                ]
            else:
                # Get all available historical data (up to 5 years max)
                symbol_data = get_historical_data_from_db(
                    symbol, 1825
                )  # Always get maximum available
                st.write(
                    f"DEBUG: Retrieved {len(symbol_data) if symbol_data else 0} historical data points for {symbol}"
                )

                if symbol_data:
                    # Show first few data points
                    if len(symbol_data) > 0:
                        st.write(f"DEBUG: First 3 {symbol} prices:")
                        for _, item in enumerate(symbol_data[:3]):
                            st.write(f"  {item['date']}: ${item['close']:.2f}")
                        if len(symbol_data) > 3:
                            st.write(f"  ... and {len(symbol_data)-3} more data points")

                if symbol_data:
                    # First try to filter to the selected period
                    filtered_data = []
                    for item in symbol_data:
                        try:
                            # Handle both date string and datetime objects
                            if isinstance(item["date"], str):
                                item_date = datetime.strptime(
                                    item["date"], "%Y-%m-%d"
                                ).date()
                            else:
                                item_date = (
                                    item["date"].date()
                                    if hasattr(item["date"], "date")
                                    else item["date"]
                                )

                            # Filter to the selected period
                            if (
                                selected_start_date.date()
                                <= item_date
                                <= end_date.date()
                            ):
                                filtered_data.append(
                                    {
                                        "date": item_date.strftime("%Y-%m-%d"),
                                        "close": item["close"],
                                    }
                                )
                        except Exception:
                            continue  # Skip problematic dates

                    # If no data in selected period, use the most recent available data
                    if not filtered_data and symbol_data:
                        st.write(
                            f"DEBUG: No data in requested period, using most recent {days} days from available data"
                        )
                        # Take the most recent `days` worth of data from what's available
                        recent_data = (
                            symbol_data[-days:]
                            if len(symbol_data) >= days
                            else symbol_data
                        )
                        for item in recent_data:
                            try:
                                if isinstance(item["date"], str):
                                    item_date = datetime.strptime(
                                        item["date"], "%Y-%m-%d"
                                    ).date()
                                else:
                                    item_date = (
                                        item["date"].date()
                                        if hasattr(item["date"], "date")
                                        else item["date"]
                                    )

                                filtered_data.append(
                                    {
                                        "date": item_date.strftime("%Y-%m-%d"),
                                        "close": item["close"],
                                    }
                                )
                            except Exception:
                                continue

                    # Use whatever filtered data we have
                    if filtered_data:
                        historical_data[symbol] = filtered_data
                        st.write(
                            f"DEBUG: Using {len(filtered_data)} data points for {symbol}"
                        )
                        # Show price range
                        if len(filtered_data) > 1:
                            prices = [item["close"] for item in filtered_data]
                            st.write(
                                f"DEBUG: {symbol} price range: ${min(prices):.2f} - ${max(prices):.2f}"
                            )

                        # Show date range of data being used
                        if filtered_data:
                            st.write(
                                f"DEBUG: Using data from {filtered_data[0]['date']} to {filtered_data[-1]['date']}"
                            )
                else:
                    st.write(f"DEBUG: No historical data found for {symbol}")

        # Check if we have any historical data at all
        total_data_points = sum(len(data) for data in historical_data.values())

        if total_data_points == 0:
            # Ultimate fallback: create a simple 2-point chart using current value
            st.info(
                "Limited historical data available. Showing simplified chart based on current portfolio value."
            )
            st.info(
                "💡 **Tip**: The background service will automatically collect more historical data over time to improve charts."
            )

            # Trigger background data collection for this asset if possible
            try:
                # Try to collect some fresh data
                symbols_to_update = [
                    asset.symbol
                    for asset in assets
                    if asset.symbol
                    not in ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]
                ]
                if symbols_to_update and hasattr(background_service, "price_fetcher"):
                    # This is a non-blocking suggestion to the background service
                    # In a production app, you might want to queue this work
                    pass  # For now, just inform the user
            except Exception:
                pass  # Silent fail - don't disrupt user experience

            if summary:
                current_value = summary.get("total_value", 0)
                if current_value > 0:
                    today = datetime.now().date().strftime("%Y-%m-%d")
                    start_date_str = selected_start_date.date().strftime("%Y-%m-%d")

                    # Create a simple 2-point chart: start period = current value, today = current value
                    # This gives user a baseline chart while data is being collected
                    return [
                        {"date": start_date_str, "value": current_value},
                        {"date": today, "value": current_value},
                    ]

            return []

        # Calculate portfolio value for each day
        portfolio_history = []

        # Get all unique dates
        all_dates = set()
        for symbol_data in historical_data.values():
            for item in symbol_data:
                all_dates.add(item["date"])

        sorted_dates = sorted(all_dates)

        for date in sorted_dates:
            daily_portfolio_value = 0
            assets_with_data = 0

            # Calculate portfolio value using dot product: Σ(asset_quantity × close_price)
            for asset in assets:
                symbol = asset.symbol
                quantity = asset.quantity

                # Handle stablecoins - always $1.00
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
                        # Use close price from historical data
                        close_price = day_data["close"]
                        asset_value = quantity * close_price
                        daily_portfolio_value += asset_value
                        assets_with_data += 1
                    else:
                        # Use backward fill - find the most recent available price
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
                                asset_value = quantity * close_price
                                daily_portfolio_value += asset_value
                                assets_with_data += 1

            # Only add this day's data if we have price data for at least some assets
            if assets_with_data > 0:
                portfolio_history.append({"date": date, "value": daily_portfolio_value})

                # DEBUG: Show price data for first few days (remove this later)
                if len(portfolio_history) <= 3:
                    st.write(
                        f"DEBUG - Date: {date}, Portfolio Value: ${daily_portfolio_value:.2f}"
                    )
                    for asset in assets:
                        if asset.symbol in historical_data:
                            day_data = next(
                                (
                                    item
                                    for item in historical_data[asset.symbol]
                                    if item["date"] == date
                                ),
                                None,
                            )
                            if day_data:
                                st.write(
                                    f"  {asset.symbol}: {asset.quantity:.6f} × ${day_data['close']:.2f} = ${asset.quantity * day_data['close']:.2f}"
                                )
                            else:
                                st.write(
                                    f"  {asset.symbol}: No price data for {date} (using backward fill)"
                                )
                    st.write("---")

        # IMPORTANT: Always ensure today's value uses exact same calculation as main summary
        # This ensures perfect consistency between chart end value and displayed totals
        if portfolio_history and summary:
            today = datetime.now().date().strftime("%Y-%m-%d")

            # CRITICAL: Use EXACT SAME value from main summary for perfect alignment
            today_value = summary.get("total_value", 0)

            if today_value > 0:
                # Always replace/add today's value to ensure chart matches summary exactly
                today_updated = False
                for item in portfolio_history:
                    if item["date"] == today:
                        item["value"] = (
                            today_value  # Overwrite with exact summary value
                        )
                        today_updated = True
                        break

                # If today's entry doesn't exist, add it as the final point
                if not today_updated:
                    portfolio_history.append({"date": today, "value": today_value})

                # Sort by date to ensure proper chronological order
                portfolio_history.sort(key=lambda x: x["date"])

        return portfolio_history

    except Exception as e:
        st.error(f"Error calculating portfolio history: {e}")
        return []


def show_portfolio_assets_table(
    assets: List, summary: Dict, consistent_prices: Dict, use_aggregated: bool = False
):
    """Show the enhanced portfolio assets table with real-time price updates."""

    # Check for real-time updates within this function
    refresh_interval = st.session_state.settings.get("refresh_interval", 60)
    current_time = datetime.now()

    # More aggressive real-time updates every 10 seconds for the table itself
    aggressive_update_interval = min(refresh_interval, 30)  # Cap at 30 seconds max

    if "last_table_price_check" not in st.session_state:
        st.session_state.last_table_price_check = current_time

    time_since_table_check = (
        current_time - st.session_state.last_table_price_check
    ).total_seconds()

    # Quick database price refresh for table display
    if time_since_table_check >= aggressive_update_interval:
        try:
            # Quick price refresh from database without full page reload
            fresh_prices = st.session_state.portfolio_manager.get_cached_prices()
            if fresh_prices:
                updated_count = 0
                for symbol, price in fresh_prices.items():
                    if st.session_state.current_prices.get(symbol) != price:
                        st.session_state.current_prices[symbol] = price
                        updated_count += 1

                if updated_count > 0:
                    st.session_state.last_table_price_check = current_time
                    # Update main refresh timestamp
                    st.session_state.last_portfolio_price_update = current_time
                    # Show quick update notification
                    st.success(f"⚡ Quick update: {updated_count} prices refreshed")
        except Exception:
            # Silent fail for quick updates to avoid disrupting user experience
            pass

    # Use the same consistent prices for the table to ensure perfect alignment
    # No need to fetch again - we already have fresh prices from above

    assets_data = []

    for asset in assets:
        if use_aggregated:
            # Handle aggregated asset (dict format)
            symbol = asset["symbol"]
            quantity = asset["quantity"]
            average_buy_price = asset["average_buy_price"]
            total_spent = asset["total_spent"]
            portfolio_name = asset["portfolio_summary"]
        else:
            # Handle regular asset (object format)
            symbol = asset.symbol
            quantity = asset.quantity
            average_buy_price = asset.average_buy_price
            total_spent = asset.total_spent
            portfolio_name = (
                asset.portfolio.name
                if hasattr(asset, "portfolio") and asset.portfolio
                else "Unknown"
            )

        # Use consistent price source for perfect alignment with summary metrics
        current_price = (
            consistent_prices.get(symbol, average_buy_price)
            if consistent_prices
            else average_buy_price
        )
        current_value = quantity * current_price
        # FIXED: Use same P&L calculation as portfolio_manager for consistency
        # Use total_spent (actual money spent) instead of quantity * average_buy_price
        pnl = current_value - total_spent
        pnl_pct = (pnl / total_spent) * 100 if total_spent > 0 else 0

        # Calculate allocations
        summary_total_value = summary.get("total_value", 1)
        current_allocation = (
            (current_value / summary_total_value) * 100
            if summary_total_value > 0
            else 0
        )
        # FIXED: Use total_spent for consistent cost basis
        initial_value = total_spent
        summary_total_spent = summary.get("total_spent", 1)
        initial_allocation = (
            (initial_value / summary_total_spent) * 100
            if summary_total_spent > 0
            else 0
        )

        # Calculate 24h change using real historical data
        if symbol in ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]:
            # Stablecoins have no price change
            change_24h_pct = 0.0
            change_24h_dollar = 0.0
        else:
            # Get real 24h change from historical data
            try:
                from database.models import get_session, HistoricalPrice

                session = get_session()
                yesterday = datetime.now() - timedelta(days=1)

                yesterday_price_record = (
                    session.query(HistoricalPrice)
                    .filter(
                        HistoricalPrice.symbol == symbol,
                        HistoricalPrice.interval == "1d",
                        HistoricalPrice.date >= yesterday - timedelta(hours=12),
                        HistoricalPrice.date <= yesterday + timedelta(hours=12),
                    )
                    .first()
                )

                session.close()

                if yesterday_price_record:
                    yesterday_price = yesterday_price_record.price
                    change_24h_pct = (
                        (current_price - yesterday_price) / yesterday_price
                    ) * 100
                    change_24h_dollar = current_value * (change_24h_pct / 100)
                else:
                    # No historical data available
                    change_24h_pct = 0.0
                    change_24h_dollar = 0.0
            except Exception:
                # Error getting historical data
                change_24h_pct = 0.0
                change_24h_dollar = 0.0

        assets_data.append(
            {
                "Symbol": symbol,
                "Portfolio": portfolio_name,
                "Current Price": format_smart_currency(current_price),
                "Quantity": format_smart_quantity(quantity),
                "Avg Buy Price": format_smart_currency(average_buy_price),
                "Total Value": f"${current_value:,.2f}",  # NEW: Current Price * Quantity
                "Total Spent": f"${quantity * average_buy_price:,.2f}",  # NEW: Average Buy Price * Quantity
                "Current Allocation %": f"{current_allocation:.1f}%",
                "Initial Allocation %": f"{initial_allocation:.1f}%",
                "24h Change": f"${change_24h_dollar:+,.2f} ({change_24h_pct:+.2f}%)",
                "P&L": f"${pnl:,.2f}",
                "P&L %": format_percentage(pnl_pct),
            }
        )

    df = pd.DataFrame(assets_data)

    # Style the dataframe
    def style_pnl(val):
        if isinstance(val, str):
            if val.startswith("+") or (
                val.startswith("$") and not val.startswith("$-")
            ):
                return "color: #28a745"
            elif val.startswith("-") or val.startswith("$-"):
                return "color: #dc3545"
        return ""

    def style_24h_change(val):
        if isinstance(val, str) and "(" in val:
            if "+" in val:
                return "color: #28a745"
            elif "-" in val:
                return "color: #dc3545"
        return ""

    styled_df = df.style.map(style_pnl, subset=["P&L", "P&L %"]).map(
        style_24h_change, subset=["24h Change"]
    )
    st.dataframe(styled_df, width="stretch")


def show_allocation_charts(assets: List, _: Dict, use_aggregated: bool = False):
    """Show current and initial allocation pie charts in columns."""
    st.subheader("Portfolio Allocation")

    if not assets:
        show_empty_state(
            title="No Assets for Allocation",
            message="Add some assets to your portfolio to see allocation breakdowns by current and initial values.",
            icon="📊",
        )
        return

    # Add View StableCoins toggle button (ticked by default)
    view_stablecoins = st.checkbox(
        "View StableCoins",
        value=True,
        help="Include stablecoins (USDT, USDC, BUSD, DAI, USDD, TUSD) in allocation charts",
    )

    # Define stablecoins list
    stablecoins = ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]

    # Filter assets based on toggle
    filtered_assets = []
    for asset in assets:
        if use_aggregated:
            symbol = asset["symbol"]
        else:
            symbol = asset.symbol

        if view_stablecoins or symbol not in stablecoins:
            filtered_assets.append(asset)

    # If no assets remain after filtering, show message
    if not filtered_assets:
        st.info(
            "💡 All assets are stablecoins. Enable 'View StableCoins' to see allocation charts."
        )
        return

    # Create data for both charts using filtered assets
    current_data = []
    initial_data = []

    # Calculate totals for filtered assets only (for proper percentage calculation when stablecoins excluded)
    filtered_total_value = 0
    filtered_total_spent = 0

    for asset in filtered_assets:
        if use_aggregated:
            symbol = asset["symbol"]
            quantity = asset["quantity"]
            average_buy_price = asset["average_buy_price"]
            total_spent = asset["total_spent"]
        else:
            symbol = asset.symbol
            quantity = asset.quantity
            average_buy_price = asset.average_buy_price
            total_spent = asset.total_spent

        # Safely get current price with fallback
        current_price = average_buy_price  # Default fallback
        if (
            hasattr(st.session_state, "current_prices")
            and st.session_state.current_prices
        ):
            current_price = st.session_state.current_prices.get(
                symbol, average_buy_price
            )
        current_value = quantity * current_price
        initial_value = total_spent

        filtered_total_value += current_value
        filtered_total_spent += initial_value

    # Build chart data using filtered assets
    for asset in filtered_assets:
        if use_aggregated:
            # Handle aggregated asset (dict format)
            symbol = asset["symbol"]
            quantity = asset["quantity"]
            average_buy_price = asset["average_buy_price"]
            total_spent = asset["total_spent"]
        else:
            # Handle regular asset (object format)
            symbol = asset.symbol
            quantity = asset.quantity
            average_buy_price = asset.average_buy_price
            total_spent = asset.total_spent

        # Safely get current price with fallback
        current_price = average_buy_price  # Default fallback
        if (
            hasattr(st.session_state, "current_prices")
            and st.session_state.current_prices
        ):
            current_price = st.session_state.current_prices.get(
                symbol, average_buy_price
            )
        current_value = quantity * current_price
        # FIXED: Use total_spent for consistent cost basis
        initial_value = total_spent

        current_data.append({"Symbol": symbol, "Current Value": current_value})

        initial_data.append({"Symbol": symbol, "Initial Value": initial_value})

    # Display pie charts in two columns
    col1, col2 = st.columns(2)

    # Update chart titles based on stablecoin filtering
    current_title = (
        "Current Allocation"
        if view_stablecoins
        else "Current Allocation (excl. StableCoins)"
    )
    initial_title = (
        "Initial Allocation"
        if view_stablecoins
        else "Initial Allocation (excl. StableCoins)"
    )

    with col1:
        # Current allocation chart
        fig_current = create_portfolio_allocation_chart(
            [
                {
                    "Symbol": item["Symbol"],
                    "Current Value": f"${item['Current Value']:,.2f}",
                }
                for item in current_data
            ],
            current_title,
        )
        st.plotly_chart(fig_current, width="stretch")

    with col2:
        # Initial allocation chart
        fig_initial = create_portfolio_allocation_chart(
            [
                {
                    "Symbol": item["Symbol"],
                    "Current Value": f"${item['Initial Value']:,.2f}",
                }
                for item in initial_data
            ],
            initial_title,
        )
        st.plotly_chart(fig_initial, width="stretch")


def show_performance_bar_charts(assets: List, use_aggregated: bool = False):
    """Show asset performance bar charts only."""
    if not assets:
        return

    # Prepare data for both charts
    assets_pct_data = []  # For % Return
    assets_dollar_data = []  # For $ Change

    for asset in assets:
        if use_aggregated:
            # Handle aggregated asset (dict format)
            symbol = asset["symbol"]
            quantity = asset["quantity"]
            average_buy_price = asset["average_buy_price"]
            total_spent = asset["total_spent"]
        else:
            # Handle regular asset (object format)
            symbol = asset.symbol
            quantity = asset.quantity
            average_buy_price = asset.average_buy_price
            total_spent = asset.total_spent

        # Skip stablecoins for performance charts
        if symbol in ["USDT", "USDC", "BUSD"]:
            continue

        # Safely get current price with fallback
        current_price = average_buy_price  # Default fallback
        if (
            hasattr(st.session_state, "current_prices")
            and st.session_state.current_prices
        ):
            current_price = st.session_state.current_prices.get(
                symbol, average_buy_price
            )
        current_value = quantity * current_price
        # FIXED: Use total_spent for consistent cost basis
        initial_value = total_spent
        pnl_dollar = current_value - initial_value
        pnl_pct = (pnl_dollar / initial_value) * 100 if initial_value > 0 else 0

        assets_pct_data.append(
            {"Symbol": symbol, "P&L %": format_percentage(pnl_pct), "value": pnl_pct}
        )

        assets_dollar_data.append(
            {
                "Symbol": symbol,
                "P&L %": f"${pnl_dollar:+.2f}",  # Keep the expected key name but with $ format
                "value": pnl_dollar,
            }
        )

    # Create two separate bar charts
    if assets_pct_data:
        col1, col2 = st.columns(2)

        with col1:
            fig_pct = create_performance_bar_chart(
                assets_pct_data, title="% Change", exclude_stablecoins=False
            )
            # Update layout for increased height
            fig_pct.update_layout(height=500)
            st.plotly_chart(fig_pct, width="stretch")

        with col2:
            fig_dollar = create_performance_bar_chart(
                assets_dollar_data, title="$ Change", exclude_stablecoins=False
            )
            # Update layout for increased height
            fig_dollar.update_layout(height=500)
            st.plotly_chart(fig_dollar, width="stretch")


def show_value_scatter_plot(assets: List, use_aggregated: bool = False):
    """Show value scatter plot separately."""
    if not assets:
        return

    st.subheader("Value Scatter Plot")

    scatter_data = []
    for asset in assets:
        if use_aggregated:
            # Handle aggregated asset (dict format)
            symbol = asset["symbol"]
            quantity = asset["quantity"]
            average_buy_price = asset["average_buy_price"]
            total_spent = asset["total_spent"]
        else:
            # Handle regular asset (object format)
            symbol = asset.symbol
            quantity = asset.quantity
            average_buy_price = asset.average_buy_price
            total_spent = asset.total_spent

        if symbol in ["USDT", "USDC", "BUSD"]:  # Skip stablecoins
            continue

        # Safely get current price with fallback
        current_price = average_buy_price  # Default fallback
        if (
            hasattr(st.session_state, "current_prices")
            and st.session_state.current_prices
        ):
            current_price = st.session_state.current_prices.get(
                symbol, average_buy_price
            )
        current_value = quantity * current_price
        # FIXED: Use total_spent for consistent cost basis
        initial_value = total_spent
        return_pct = (
            ((current_value - initial_value) / initial_value * 100)
            if initial_value > 0
            else 0
        )

        scatter_data.append(
            {
                "symbol": symbol,
                "initial_value": initial_value,
                "current_value": current_value,
                "return_pct": abs(return_pct),
                "return_sign": 1 if return_pct >= 0 else -1,
                "return_display": f"{return_pct:+,.1f}%",
            }
        )

    if scatter_data:
        df_scatter = pd.DataFrame(scatter_data)

        fig_scatter = px.scatter(
            df_scatter,
            x="initial_value",
            y="current_value",
            size="return_pct",
            color="return_sign",
            hover_data=["symbol", "return_display"],
            title="Initial vs Current Value (Bubble size = |Return %|)",
            color_continuous_scale=["red", "green"],
            labels={
                "initial_value": "Initial Value ($)",
                "current_value": "Current Value ($)",
                "return_sign": "Return",
            },
        )

        # Add diagonal line for break-even
        max_val = max(
            df_scatter["initial_value"].max(), df_scatter["current_value"].max()
        )
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_val,
            y1=max_val,
            line=dict(color="gray", width=1, dash="dash"),
            name="Break-even",
        )

        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, width="stretch")


def show_asset_management(assets: List, selected_portfolio):
    """Show asset management section with enhanced functionality."""
    col1, col2 = st.columns(2)

    with col1:
        show_add_asset_form(selected_portfolio)
        show_portfolio_management()

    with col2:
        show_edit_asset_form(assets)
        show_delete_sell_asset_form(assets)
        show_exchange_functionality()


def show_add_asset_form(selected_portfolio):
    """Enhanced form to add new assets."""
    with st.expander("➕ Add New Asset", expanded=True):
        with st.form("add_asset"):
            col1, col2 = st.columns(2)

            with col1:
                symbol = st.text_input("Symbol (e.g., BTC)", key="add_symbol").upper()
                quantity = st.number_input(
                    "Quantity",
                    min_value=0.0,
                    step=0.000001,
                    format="%.6f",
                    key="add_quantity",
                )

            with col2:
                buy_price = st.number_input(
                    "Buy Price ($)",
                    min_value=0.0,
                    step=0.01,
                    format="%.4f",
                    key="add_price",
                )

                # Portfolio selection for new asset - always show selection
                portfolios = st.session_state.portfolio_manager.get_all_portfolios()
                if portfolios:
                    portfolio_options = {p.name: p.id for p in portfolios}

                    # Determine default selection
                    default_index = 0
                    if (
                        selected_portfolio != "all"
                        and selected_portfolio in portfolio_options.values()
                    ):
                        default_index = list(portfolio_options.values()).index(
                            selected_portfolio
                        )

                    selected_portfolio_name = st.selectbox(
                        "Add to Portfolio",
                        list(portfolio_options.keys()),
                        index=default_index,
                        key="add_asset_portfolio_selector",
                    )
                    target_portfolio = portfolio_options[selected_portfolio_name]
                else:
                    st.error("No portfolios found. Please create a portfolio first.")
                    target_portfolio = None

            # Operation type and stablecoin selection - BOTH shown upfront
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                operation_type = st.radio(
                    "Operation Type",
                    ["Add (Increase total wealth)", "Buy (Use USDT/USDC)"],
                    help="Add increases your total wealth, Buy uses existing stablecoin balance",
                )

            with col2:
                # ALWAYS show stablecoin selection - user chooses before confirming
                stable_coin = st.selectbox(
                    "Stablecoin (for Buy operations)",
                    ["USDT", "USDC"],
                    key="add_asset_stable_coin",
                    help="Choose which stablecoin to use for Buy operations",
                )

                # Show balance information when Buy is selected
                if operation_type.startswith("Buy") and target_portfolio is not None:
                    # FORCE fresh balance check - don't rely on cached data
                    stable_assets = (
                        st.session_state.portfolio_manager.get_portfolio_assets(
                            target_portfolio
                        )
                    )
                    stable_balance = 0
                    for asset in stable_assets:
                        if asset.symbol == stable_coin:
                            stable_balance = asset.quantity
                            break

                    total_cost = (
                        quantity * buy_price if quantity > 0 and buy_price > 0 else 0
                    )

                    if total_cost > 0:
                        if stable_balance >= total_cost:
                            st.success(
                                f"✅ {stable_coin}: ${stable_balance:.2f} (Need: ${total_cost:.2f})"
                            )
                        else:
                            st.error(
                                f"❌ {stable_coin}: ${stable_balance:.2f} (Need: ${total_cost:.2f})"
                            )
                    else:
                        st.info(f"💰 Available {stable_coin}: ${stable_balance:.2f}")

            submitted = st.form_submit_button("Add Asset", type="primary")

            if (
                submitted
                and symbol
                and quantity > 0
                and buy_price > 0
                and target_portfolio is not None
            ):
                # Prevent duplicate operations
                operation_key = f"add_asset_{symbol}_{target_portfolio}"
                if is_operation_in_progress(operation_key):
                    st.warning("⏳ Operation already in progress. Please wait...")
                    st.stop()

                set_operation_in_progress(operation_key, True)

                try:

                    if operation_type.startswith("Buy"):
                        # Re-check balance at submission time for accuracy
                        stable_assets = (
                            st.session_state.portfolio_manager.get_portfolio_assets(
                                target_portfolio
                            )
                        )
                        stable_balance = 0
                        stable_asset = None

                        for asset in stable_assets:
                            if asset.symbol == stable_coin:
                                stable_balance = asset.quantity
                                stable_asset = asset
                                break

                        total_cost = quantity * buy_price
                        # Use a more lenient balance check to avoid floating point precision issues
                        if stable_balance < (total_cost - 0.01):
                            st.error(
                                f"❌ Insufficient {stable_coin} balance. Available: ${stable_balance:.2f}, Required: ${total_cost:.2f}"
                            )
                            st.stop()  # Prevent further execution

                        # Execute the buy transaction atomically
                        # 1. Add the new asset first
                        st.session_state.portfolio_manager.add_asset(
                            portfolio_id=target_portfolio,
                            symbol=symbol,
                            quantity=quantity,
                            buy_price=buy_price,
                        )

                        # 2. Then reduce stablecoin balance
                        new_stable_quantity = stable_balance - total_cost
                        if new_stable_quantity > 0:
                            st.session_state.portfolio_manager.update_asset(
                                asset_id=stable_asset.id, quantity=new_stable_quantity
                            )
                        else:
                            st.session_state.portfolio_manager.delete_asset(
                                stable_asset.id
                            )

                        st.success(
                            f"✅ Successfully bought {quantity:.6f} {symbol} for ${total_cost:.2f} {stable_coin}"
                        )
                        st.info(
                            f"💰 Remaining {stable_coin} balance: ${new_stable_quantity:.2f}"
                        )

                    else:
                        # Simple add operation - increases total wealth
                        st.session_state.portfolio_manager.add_asset(
                            portfolio_id=target_portfolio,
                            symbol=symbol,
                            quantity=quantity,
                            buy_price=buy_price,
                        )

                        st.success(
                            f"✅ Added {quantity:.6f} {symbol} to portfolio (Increased total wealth by ${quantity * buy_price:.2f})"
                        )

                    # Properly refresh portfolio data and UI
                    refresh_portfolio_data_after_operation()

                    # Clear operation lock before rerun
                    set_operation_in_progress(operation_key, False)
                    st.rerun()

                except Exception as e:
                    # Clear operation lock on error
                    set_operation_in_progress(operation_key, False)
                    st.error(f"❌ Error adding asset: {str(e)}")
                    logger.error(f"Asset operation failed: {str(e)}")


def show_portfolio_management():
    """Portfolio management functionality."""
    with st.expander("📁 Manage Portfolios", expanded=False):
        st.subheader("🆕 Create New Portfolio")
        st.markdown(
            "**Create additional portfolios to organize your investments by strategy, purpose, or timeframe.**"
        )

        # Show some naming suggestions
        with st.expander("💡 Portfolio Naming Ideas", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                **By Strategy:**
                • Long-term Holdings
                • Day Trading Portfolio  
                • DeFi Investments
                • Blue Chip Cryptos
                • Altcoin Experiments
                """
                )
            with col2:
                st.markdown(
                    """
                **By Purpose:**
                • Retirement Savings
                • Emergency Fund
                • Growth Portfolio
                • Income Generation
                • Speculative Plays
                """
                )

        with st.form("create_portfolio"):
            st.markdown("**Enter your new portfolio name:**")
            new_portfolio_name = st.text_input(
                "Portfolio Name",
                value="",
                placeholder="e.g., DeFi Portfolio, Long-term Holdings, Trading Fund...",
                help="Choose a descriptive name that reflects the purpose or strategy of this portfolio",
            )

            col1, col2 = st.columns(2)
            with col1:
                if new_portfolio_name.strip():
                    create_clicked = st.form_submit_button(
                        f"✨ Create '{new_portfolio_name.strip()}'", type="primary"
                    )
                else:
                    create_clicked = st.form_submit_button(
                        "Create Portfolio", type="primary"
                    )

            if create_clicked:
                if new_portfolio_name.strip():
                    try:
                        portfolio = st.session_state.portfolio_manager.create_portfolio(
                            new_portfolio_name.strip()
                        )
                        st.success(
                            f"✅ Created portfolio '{portfolio.name}' successfully!"
                        )
                        st.info(
                            "🎉 Your new portfolio is ready! You can now add assets to it."
                        )

                        # Clear cached portfolio data to ensure fresh reload
                        for key in list(st.session_state.keys()):
                            if (
                                "cached_portfolios" in key
                                or "portfolio_cache" in key
                                or "portfolio_selector_cache" in key
                            ):
                                del st.session_state[key]

                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        if "UNIQUE constraint failed" in str(e):
                            st.error(
                                f"❌ Portfolio named '{new_portfolio_name.strip()}' already exists. Please choose a different name."
                            )
                        else:
                            st.error(f"❌ Error creating portfolio: {e}")
                else:
                    st.error("❌ Portfolio name cannot be empty")

        st.subheader("✏️ Edit Portfolio Name")
        portfolios = st.session_state.portfolio_manager.get_all_portfolios()

        if portfolios:
            with st.form("edit_portfolio_name"):
                portfolio_to_edit = st.selectbox(
                    "Select Portfolio to Rename",
                    options=[p.name for p in portfolios],
                    help="Choose which portfolio you want to rename",
                )

                # Get current portfolio
                current_portfolio = next(
                    p for p in portfolios if p.name == portfolio_to_edit
                )

                new_portfolio_name = st.text_input(
                    "New Portfolio Name",
                    value=current_portfolio.name,
                    help="Enter the new name for your portfolio",
                )

                # Always show an enabled button - validation happens after click
                rename_clicked = st.form_submit_button(
                    "✏️ Rename Portfolio", type="primary"
                )

                if rename_clicked:
                    new_name = new_portfolio_name.strip() if new_portfolio_name else ""
                    current_name = (
                        current_portfolio.name
                    )  # Store name before DB operation
                    current_id = current_portfolio.id  # Store ID before DB operation

                    if not new_name:
                        st.error("❌ Portfolio name cannot be empty")
                    elif new_name == current_name:
                        st.info("💡 Portfolio name is the same - no changes needed")
                    else:
                        try:
                            result = st.session_state.portfolio_manager.update_portfolio_name(
                                current_id, new_name
                            )

                            # Clear portfolio caches to ensure UI refresh
                            refresh_portfolio_data_after_operation()

                            st.success(
                                f"✅ Portfolio renamed from '{result['old_name']}' to '{result['name']}' !"
                            )
                            st.info(
                                "🎉 Your portfolio name has been updated successfully!"
                            )
                            time.sleep(1)
                            st.rerun()
                        except ValueError as e:
                            st.error(f"❌ {e}")
                        except Exception as e:
                            st.error(f"❌ Error renaming portfolio: {e}")
        else:
            st.info("No portfolios available to rename")

        st.subheader("🗑️ Delete Portfolio")
        portfolios = st.session_state.portfolio_manager.get_all_portfolios()
        deletable_portfolios = [p for p in portfolios if p.name != "Main Portfolio"]

        if deletable_portfolios:
            with st.form("delete_portfolio"):
                portfolio_to_delete = st.selectbox(
                    "Select Portfolio to Delete", [p.name for p in deletable_portfolios]
                )

                move_assets = st.checkbox(
                    "Move assets to Main Portfolio (otherwise assets will be deleted)"
                )

                if st.form_submit_button("Delete Portfolio"):
                    try:
                        portfolio = next(
                            p
                            for p in deletable_portfolios
                            if p.name == portfolio_to_delete
                        )

                        # Determine target portfolio for asset migration
                        target_portfolio_id = None
                        if move_assets:
                            main_portfolio = st.session_state.portfolio_manager.get_portfolio_by_name(
                                "Main Portfolio"
                            )
                            if not main_portfolio:
                                # Create Main Portfolio if it doesn't exist
                                main_portfolio = (
                                    st.session_state.portfolio_manager.create_portfolio(
                                        "Main Portfolio"
                                    )
                                )
                            target_portfolio_id = main_portfolio.id

                        # Use the new delete_portfolio method
                        result = st.session_state.portfolio_manager.delete_portfolio(
                            portfolio.id, target_portfolio_id
                        )

                        # Clear portfolio caches to ensure UI refresh
                        refresh_portfolio_data_after_operation()

                        # Show success message
                        if result["assets_moved"]:
                            st.success(
                                f"✅ Portfolio '{result['portfolio_name']}' deleted and {result['asset_count']} assets moved to Main Portfolio!"
                            )
                        else:
                            st.success(
                                f"✅ Portfolio '{result['portfolio_name']}' and {result['asset_count']} assets deleted!"
                            )

                        # Reset selected portfolio if we deleted the current one
                        if st.session_state.get("selected_portfolio") == portfolio.id:
                            st.session_state.selected_portfolio = "all"

                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting portfolio: {e}")


def show_edit_asset_form(assets: List):
    """Enhanced form to edit existing assets."""
    if not assets:
        return

    with st.expander("✏️ Edit Assets", expanded=True):
        asset_options = {
            f"{a.symbol} ({a.quantity:.6f}) - {a.portfolio.name if hasattr(a, 'portfolio') and a.portfolio else 'Unknown'}": a.id
            for a in assets
        }
        selected_asset = st.selectbox(
            "Select Asset to Edit", options=list(asset_options.keys())
        )

        if selected_asset:
            asset_id = asset_options[selected_asset]
            asset = next(a for a in assets if a.id == asset_id)

            with st.form("edit_asset"):
                col1, col2 = st.columns(2)

                with col1:
                    new_quantity = st.number_input(
                        "Quantity",
                        value=float(asset.quantity),
                        min_value=0.0,
                        step=0.000001,
                        format="%.6f",
                    )
                    new_price = st.number_input(
                        "Average Buy Price ($)",
                        value=float(asset.average_buy_price),
                        min_value=0.0,
                        step=0.01,
                        format="%.4f",
                    )

                with col2:
                    # Portfolio transfer
                    portfolios = st.session_state.portfolio_manager.get_all_portfolios()
                    portfolio_options = {p.name: p.id for p in portfolios}
                    current_portfolio_name = (
                        asset.portfolio.name
                        if hasattr(asset, "portfolio") and asset.portfolio
                        else "Unknown"
                    )

                    new_portfolio_name = st.selectbox(
                        "Move to Portfolio",
                        list(portfolio_options.keys()),
                        index=(
                            list(portfolio_options.keys()).index(current_portfolio_name)
                            if current_portfolio_name in portfolio_options.keys()
                            else 0
                        ),
                    )
                    new_portfolio_id = portfolio_options[new_portfolio_name]

                # Apply Changes always performs an update

                submitted = st.form_submit_button("Apply Changes", type="primary")

                if submitted:
                    # Prevent duplicate operations
                    operation_key = f"edit_asset_{asset.id}_update"
                    if is_operation_in_progress(operation_key):
                        st.warning("⏳ Operation already in progress. Please wait...")
                        st.stop()

                    set_operation_in_progress(operation_key, True)

                    try:
                        # Always update asset (no remove option)
                        if new_portfolio_id != asset.portfolio_id:
                            # Portfolio transfer - delete and recreate
                            st.session_state.portfolio_manager.delete_asset(asset_id)
                            st.session_state.portfolio_manager.add_asset(
                                portfolio_id=new_portfolio_id,
                                symbol=asset.symbol,
                                quantity=new_quantity,
                                buy_price=new_price,
                            )

                            st.success(
                                f"✅ Moved {asset.symbol} to {new_portfolio_name} and updated values!"
                            )
                        else:
                            # Simple update within same portfolio
                            st.session_state.portfolio_manager.update_asset(
                                asset_id=asset_id,
                                quantity=new_quantity,
                                buy_price=new_price,
                            )
                            st.success(
                                f"✅ Updated {asset.symbol}: {format_crypto_quantity(new_quantity)} @ {format_currency(new_price, 4)}"
                            )

                        # Force clear any cached portfolio/asset data
                        for key in list(st.session_state.keys()):
                            if "cached_" in key or "historical_data" in key:
                                del st.session_state[key]

                        # Refresh portfolio data and UI
                        refresh_portfolio_data_after_operation()

                        # Clear operation lock before rerun
                        set_operation_in_progress(operation_key, False)
                        st.rerun()

                    except Exception as e:
                        # Clear operation lock on error
                        set_operation_in_progress(operation_key, False)
                        st.error(f"❌ Error updating asset: {str(e)}")
                        logger.error(f"Asset operation failed: {str(e)}")


def show_delete_sell_asset_form(assets: List):
    """Dedicated form to delete or sell assets with custom sell price."""
    if not assets:
        return

    with st.expander("🗑️ Delete/Sell Assets", expanded=True):
        # Step 1: Asset Selection (outside of form)
        asset_options = {
            f"{a.symbol} ({a.quantity:.6f}) - {a.portfolio.name if hasattr(a, 'portfolio') and a.portfolio else 'Unknown'}": a.id
            for a in assets
        }
        selected_asset = st.selectbox(
            "Select Asset to Delete/Sell",
            options=list(asset_options.keys()),
            key="delete_sell_asset",
        )

        if selected_asset:
            asset_id = asset_options[selected_asset]
            asset = next(a for a in assets if a.id == asset_id)

            # Show asset details
            st.write(f"**Asset Details:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Symbol: {asset.symbol}")
                st.write(f"Current Quantity: {format_smart_quantity(asset.quantity)}")
            with col2:
                st.write(
                    f"Avg Buy Price: {format_smart_currency(asset.average_buy_price)}"
                )

                # Safely get current price with fallback
                current_price = asset.average_buy_price  # Default fallback
                if (
                    hasattr(st.session_state, "current_prices")
                    and st.session_state.current_prices
                ):
                    current_price = st.session_state.current_prices.get(
                        asset.symbol, asset.average_buy_price
                    )
                else:
                    st.session_state.current_prices = {}
                    current_price = asset.average_buy_price
                current_value = asset.quantity * current_price
                st.write(f"Current Market Price: ${current_price:.4f}")
                st.write(f"Total Current Value: ${current_value:.2f}")

            # Step 2: Action Selection (outside of form)
            st.markdown("---")
            action = st.radio(
                "Select Action",
                ["Delete Completely", "Sell (Convert to Stablecoin)"],
                key="delete_sell_action_choice",
            )

            # Step 3: Action-specific form
            if action == "Delete Completely":
                st.warning(
                    "⚠️ This will permanently remove the asset from your portfolio and decrease your total wealth."
                )

                with st.form("delete_asset_form"):
                    quantity_to_remove = st.number_input(
                        "Quantity to Delete",
                        min_value=0.0,
                        max_value=float(asset.quantity),
                        value=float(asset.quantity),
                        step=0.000001,
                        format="%.6f",
                        key="delete_quantity",
                    )

                    submitted = st.form_submit_button("🗑️ Delete Asset", type="primary")

                    if submitted:
                        # Prevent duplicate operations
                        operation_key = f"delete_asset_{asset.id}"
                        if is_operation_in_progress(operation_key):
                            st.warning(
                                "⏳ Operation already in progress. Please wait..."
                            )
                            st.stop()

                        set_operation_in_progress(operation_key, True)

                        try:
                            if quantity_to_remove >= asset.quantity:
                                # Delete entire asset
                                st.session_state.portfolio_manager.delete_asset(
                                    asset_id
                                )
                                st.success(
                                    f"✅ Completely removed {asset.symbol} from portfolio!"
                                )
                            else:
                                # Partial delete - reduce quantity
                                new_quantity = asset.quantity - quantity_to_remove
                                st.session_state.portfolio_manager.update_asset(
                                    asset_id, quantity=new_quantity
                                )
                                st.success(
                                    f"✅ Deleted {quantity_to_remove:.6f} {asset.symbol} from portfolio!"
                                )

                            # Force clear cached data and refresh UI
                            for key in list(st.session_state.keys()):
                                if "cached_" in key or "historical_data" in key:
                                    del st.session_state[key]

                            refresh_portfolio_data_after_operation()

                            # Clear operation lock before rerun
                            set_operation_in_progress(operation_key, False)
                            st.rerun()

                        except Exception as e:
                            # Clear operation lock on error
                            set_operation_in_progress(operation_key, False)
                            st.error(f"❌ Error deleting asset: {str(e)}")

            else:  # Sell action
                st.info(
                    "💰 This will convert your asset to a stablecoin at the specified price."
                )

                with st.form("sell_asset_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        quantity_to_sell = st.number_input(
                            "Quantity to Sell",
                            min_value=0.0,
                            max_value=float(asset.quantity),
                            value=float(asset.quantity),
                            step=0.000001,
                            format="%.6f",
                            key="sell_quantity",
                        )

                        sell_price = st.number_input(
                            "Sell Price ($)",
                            value=float(current_price),
                            min_value=0.01,
                            step=0.01,
                            format="%.4f",
                            key="sell_price",
                            help="Specify the price at which you want to sell",
                        )

                    with col2:
                        stable_coin = st.selectbox(
                            "Convert to",
                            ["USDT", "USDC"],
                            key="sell_stablecoin",
                            help="Choose which stablecoin to receive from the sale",
                        )
                        st.info(f"💰 Will receive {stable_coin} in same portfolio")

                        # Show potential proceeds
                        if quantity_to_sell > 0 and sell_price > 0:
                            proceeds = quantity_to_sell * sell_price
                            profit = proceeds - (
                                quantity_to_sell * asset.average_buy_price
                            )
                            profit_pct = (
                                (profit / (quantity_to_sell * asset.average_buy_price))
                                * 100
                                if asset.average_buy_price > 0
                                else 0
                            )

                            st.write(f"**Sale Summary:**")
                            st.write(f"Proceeds: ${proceeds:.2f}")
                            st.write(f"Profit: ${profit:.2f} ({profit_pct:+.2f}%)")

                    submitted = st.form_submit_button("💰 Execute Sale", type="primary")

                    if submitted and quantity_to_sell > 0:
                        # Prevent duplicate operations
                        operation_key = f"sell_asset_{asset.id}_{quantity_to_sell}"
                        if is_operation_in_progress(operation_key):
                            st.warning(
                                "⏳ Operation already in progress. Please wait..."
                            )
                            st.stop()

                        set_operation_in_progress(operation_key, True)

                        try:
                            # Calculate profit information
                            proceeds = quantity_to_sell * sell_price
                            cost_basis = quantity_to_sell * asset.average_buy_price
                            profit = proceeds - cost_basis
                            profit_pct = (
                                (profit / cost_basis) * 100 if cost_basis > 0 else 0
                            )

                            # Use the portfolio manager's sell_asset method which handles profit recording
                            try:
                                _, _ = st.session_state.portfolio_manager.sell_asset(
                                    asset_id=asset_id,
                                    quantity_to_sell=quantity_to_sell,
                                    sell_price=sell_price,
                                )

                                # Add stablecoin proceeds to the same portfolio
                                st.session_state.portfolio_manager.add_asset(
                                    portfolio_id=asset.portfolio_id,
                                    symbol=stable_coin,
                                    quantity=proceeds,
                                    buy_price=1.0,
                                )

                                st.success(
                                    f"✅ Sold {format_crypto_quantity(quantity_to_sell)} {asset.symbol} for {format_currency(proceeds)}"
                                )
                                st.success(
                                    f"💰 Will Receive: {proceeds:.2f} {stable_coin}"
                                )
                                st.success(
                                    f"💰 Realized P&L: ${profit:.2f} ({profit_pct:+.2f}%)"
                                )
                                st.info(
                                    f"📊 Profit record saved for portfolio tracking"
                                )

                            except Exception as sell_error:
                                # Fallback to manual implementation if sell_asset fails
                                st.warning(f"Using fallback sell method: {sell_error}")

                                # Manually reduce or remove the original asset
                                if quantity_to_sell >= asset.quantity:
                                    st.session_state.portfolio_manager.delete_asset(
                                        asset_id
                                    )
                                else:
                                    new_quantity = asset.quantity - quantity_to_sell
                                    st.session_state.portfolio_manager.update_asset(
                                        asset_id, quantity=new_quantity
                                    )

                                # Add stablecoin proceeds to the same portfolio
                                st.session_state.portfolio_manager.add_asset(
                                    portfolio_id=asset.portfolio_id,
                                    symbol=stable_coin,
                                    quantity=proceeds,
                                    buy_price=1.0,
                                )

                                st.success(
                                    f"✅ Sold {format_crypto_quantity(quantity_to_sell)} {asset.symbol} for {format_currency(proceeds)}"
                                )
                                st.success(
                                    f"💰 Will Receive: {proceeds:.2f} {stable_coin}"
                                )
                                st.success(
                                    f"💰 P&L: ${profit:.2f} ({profit_pct:+.2f}%)"
                                )

                            # Force clear any cached data for immediate UI refresh
                            for key in list(st.session_state.keys()):
                                if "cached_" in key or "historical_data" in key:
                                    del st.session_state[key]

                            # Refresh portfolio data and UI
                            refresh_portfolio_data_after_operation()

                            # Clear operation lock before rerun
                            set_operation_in_progress(operation_key, False)
                            st.rerun()

                        except Exception as e:
                            # Clear operation lock on error
                            set_operation_in_progress(operation_key, False)
                            st.error(f"❌ Error executing sale: {str(e)}")
                            logger.error(f"Asset operation failed: {str(e)}")


def show_exchange_functionality():
    """USDT/USDC exchange functionality with portfolio selection."""
    with st.expander("💱 Exchange USDT ⇄ USDC", expanded=True):
        # Get all portfolios for selection
        portfolios = st.session_state.portfolio_manager.get_all_portfolios()
        if not portfolios:
            st.error("No portfolios found.")
            return

        portfolio_options = {f"{p.name}": p.id for p in portfolios}

        with st.form("exchange_stablecoins"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Exchange Direction**")
                exchange_direction = st.radio(
                    "Direction",
                    ["USDT → USDC", "USDC → USDT"],
                    label_visibility="collapsed",
                )

                if exchange_direction == "USDT → USDC":
                    from_coin = "USDT"
                    to_coin = "USDC"
                else:
                    from_coin = "USDC"
                    to_coin = "USDT"

            with col2:
                st.markdown("**Source Portfolio**")
                source_portfolio_name = st.selectbox(
                    "From Portfolio",
                    list(portfolio_options.keys()),
                    key="exchange_source",
                )
                source_portfolio_id = portfolio_options[source_portfolio_name]

                # Get source balance
                source_assets = st.session_state.portfolio_manager.get_portfolio_assets(
                    source_portfolio_id
                )
                source_balance = 0
                source_asset = None

                for asset in source_assets:
                    if asset.symbol == from_coin:
                        source_balance = asset.quantity
                        source_asset = asset
                        break

                st.write(f"Available {from_coin}: {source_balance:.2f}")

            with col3:
                st.markdown("**Destination Portfolio**")
                dest_portfolio_name = st.selectbox(
                    "To Portfolio", list(portfolio_options.keys()), key="exchange_dest"
                )
                dest_portfolio_id = portfolio_options[dest_portfolio_name]

            # Amount input - use a more lenient maximum to avoid floating point precision issues
            amount = st.number_input(
                f"Amount to exchange ({from_coin})",
                min_value=0.0,
                max_value=source_balance * 1.001,
                step=0.01,
                format="%.2f",
            )

            if st.form_submit_button("Execute Exchange", type="primary"):
                # Use a more lenient balance check to avoid floating point precision issues
                if amount > 0 and amount <= (source_balance + 0.001):
                    # Prevent duplicate operations
                    operation_key = f"exchange_{from_coin}_{to_coin}_{amount}_{source_portfolio_id}_{dest_portfolio_id}"
                    if is_operation_in_progress(operation_key):
                        st.warning("⏳ Operation already in progress. Please wait...")
                        st.stop()

                    set_operation_in_progress(operation_key, True)

                    try:
                        # Show what's happening for transparency

                        # Atomically execute the exchange transaction
                        if not source_asset:
                            st.error(
                                f"❌ No {from_coin} balance found in source portfolio"
                            )
                            st.stop()

                        # 1. Reduce source balance first
                        new_source_balance = source_balance - amount
                        if new_source_balance > 0:
                            st.session_state.portfolio_manager.update_asset(
                                source_asset.id, quantity=new_source_balance
                            )
                        else:
                            st.session_state.portfolio_manager.delete_asset(
                                source_asset.id
                            )

                        # 2. Add to destination portfolio
                        st.session_state.portfolio_manager.add_asset(
                            portfolio_id=dest_portfolio_id,
                            symbol=to_coin,
                            quantity=amount,
                            buy_price=1.0,
                        )

                        st.success(
                            f"✅ Successfully exchanged {amount:.2f} {from_coin} → {to_coin}"
                        )
                        st.info(
                            f"📋 From: {source_portfolio_name} → To: {dest_portfolio_name}"
                        )

                        # Force clear any cached data for immediate UI refresh
                        for key in list(st.session_state.keys()):
                            if "cached_" in key or "historical_data" in key:
                                del st.session_state[key]

                        # Multiple refresh operations to ensure UI synchronization
                        refresh_portfolio_data_after_operation()

                        # Clear operation lock before rerun
                        set_operation_in_progress(operation_key, False)

                        # Force session state to update immediately
                        st.session_state._rerun_requested = True
                        st.rerun()

                    except Exception as e:
                        # Clear operation lock on error
                        set_operation_in_progress(operation_key, False)
                        st.error(f"❌ Error executing exchange: {str(e)}")
                        logger.error(f"Asset operation failed: {str(e)}")
                else:
                    if amount <= 0:
                        st.error(
                            "❌ Invalid exchange amount. Amount must be greater than 0."
                        )
                    elif amount > source_balance:
                        st.error(
                            f"❌ Insufficient balance. Available: ${source_balance:.2f}, Requested: ${amount:.2f}"
                        )
                    else:
                        st.error("❌ Unknown validation error.")


def show_portfolios_summary_table(consistent_prices: Dict):
    """Show portfolios summary table when 'All Portfolios' is selected."""
    st.subheader("Portfolios")

    # Get all portfolios
    portfolios = st.session_state.portfolio_manager.get_all_portfolios()
    if not portfolios:
        st.info("No portfolios found.")
        return

    portfolios_data = []

    # Get total summary for allocation calculations
    total_summary = (
        st.session_state.portfolio_manager.get_all_portfolios_summary_aggregated(
            consistent_prices
        )
    )
    total_value_all = total_summary.get("total_value", 1)
    total_spent_all = total_summary.get("total_spent", 1)

    for portfolio in portfolios:
        # Get individual portfolio summary
        portfolio_summary = st.session_state.portfolio_manager.get_portfolio_summary(
            portfolio.id, consistent_prices
        )

        total_value = portfolio_summary.get("total_value", 0)
        total_spent = portfolio_summary.get("total_spent", 0)
        pnl = portfolio_summary.get("total_return", 0)
        pnl_pct = portfolio_summary.get("total_return_percentage", 0)

        # Calculate allocations
        current_allocation = (
            (total_value / total_value_all) * 100 if total_value_all > 0 else 0
        )
        initial_allocation = (
            (total_spent / total_spent_all) * 100 if total_spent_all > 0 else 0
        )

        # Calculate 24h change by summing individual asset 24h changes
        try:
            from database.models import get_session, HistoricalPrice

            session = get_session()
            yesterday = datetime.now() - timedelta(days=1)

            # Get all assets in this portfolio
            portfolio_assets = st.session_state.portfolio_manager.get_portfolio_assets(
                portfolio.id
            )
            change_24h_dollar = 0.0

            for asset in portfolio_assets:
                if asset.symbol in ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]:
                    # Stablecoins have no price change
                    continue

                # Get current and yesterday's price
                current_price = consistent_prices.get(
                    asset.symbol, asset.average_buy_price
                )

                yesterday_price_record = (
                    session.query(HistoricalPrice)
                    .filter(
                        HistoricalPrice.symbol == asset.symbol,
                        HistoricalPrice.interval == "1d",
                        HistoricalPrice.date >= yesterday - timedelta(hours=12),
                        HistoricalPrice.date <= yesterday + timedelta(hours=12),
                    )
                    .first()
                )

                if yesterday_price_record:
                    yesterday_price = yesterday_price_record.price
                    price_change = current_price - yesterday_price
                    asset_change_dollar = asset.quantity * price_change
                    change_24h_dollar += asset_change_dollar

            # Calculate percentage change
            yesterday_portfolio_value = total_value - change_24h_dollar
            change_24h_pct = (
                (change_24h_dollar / yesterday_portfolio_value) * 100
                if yesterday_portfolio_value > 0
                else 0
            )

            session.close()
        except Exception:
            change_24h_dollar = 0.0
            change_24h_pct = 0.0

        portfolios_data.append(
            {
                "Portfolio": portfolio.name,
                "Total Value": f"${total_value:,.2f}",
                "Total Spent": f"${total_spent:,.2f}",
                "Current Allocation %": f"{current_allocation:.1f}%",
                "Initial Allocation %": f"{initial_allocation:.1f}%",
                "24h Change": f"${change_24h_dollar:+,.2f} ({change_24h_pct:+.2f}%)",
                "P&L": f"${pnl:,.2f}",
                "P&L %": format_percentage(pnl_pct),
            }
        )

    df = pd.DataFrame(portfolios_data)

    # Style the dataframe similar to portfolio assets table
    def style_pnl(val):
        if isinstance(val, str):
            if val.startswith("+") or (
                val.startswith("$") and not val.startswith("$-")
            ):
                return "color: #28a745"
            elif val.startswith("-") or val.startswith("$-"):
                return "color: #dc3545"
        return ""

    def style_24h_change(val):
        if isinstance(val, str) and "(" in val:
            if "+" in val:
                return "color: #28a745"
            elif "-" in val:
                return "color: #dc3545"
        return ""

    styled_df = df.style.map(style_pnl, subset=["P&L", "P&L %"]).map(
        style_24h_change, subset=["24h Change"]
    )
    st.dataframe(styled_df, width="stretch")


def show_portfolio_charts(consistent_prices: Dict):
    """Show portfolio-level charts when 'All Portfolios' is selected."""
    st.subheader("Portfolio Analysis")

    # Get all portfolios
    portfolios = st.session_state.portfolio_manager.get_all_portfolios()
    if not portfolios:
        st.info("No portfolios found.")
        return

    # Prepare data for charts
    portfolio_current_allocation_data = []
    portfolio_initial_allocation_data = []
    portfolio_performance_pct_data = []
    portfolio_performance_dollar_data = []

    for portfolio in portfolios:
        portfolio_summary = st.session_state.portfolio_manager.get_portfolio_summary(
            portfolio.id, consistent_prices
        )

        total_value = portfolio_summary.get("total_value", 0)
        total_spent = portfolio_summary.get("total_spent", 0)
        pnl = portfolio_summary.get("total_return", 0)
        pnl_pct = portfolio_summary.get("total_return_percentage", 0)

        if total_value > 0:  # Only include portfolios with value
            portfolio_current_allocation_data.append(
                {"Symbol": portfolio.name, "Current Value": f"${total_value:,.2f}"}
            )

            portfolio_initial_allocation_data.append(
                {"Symbol": portfolio.name, "Current Value": f"${total_spent:,.2f}"}
            )

            portfolio_performance_pct_data.append(
                {
                    "Symbol": portfolio.name,
                    "P&L %": f"{pnl_pct:+.1f}%",
                    "value": pnl_pct,
                }
            )

            portfolio_performance_dollar_data.append(
                {"Symbol": portfolio.name, "P&L %": f"${pnl:+,.2f}", "value": pnl}
            )

    if not portfolio_current_allocation_data:
        st.info("No portfolios with value found for charts.")
        return

    # First row: Portfolio allocation pie charts
    st.subheader("Portfolio Allocation")
    col1, col2 = st.columns(2)

    with col1:
        fig_current = create_portfolio_allocation_chart(
            portfolio_current_allocation_data, "Current Allocation"
        )
        st.plotly_chart(fig_current, width="stretch")

    with col2:
        fig_initial = create_portfolio_allocation_chart(
            portfolio_initial_allocation_data, "Initial Allocation"
        )
        st.plotly_chart(fig_initial, width="stretch")

    # Second row: Portfolio performance bar charts
    st.subheader("Portfolio Performance")
    col1, col2 = st.columns(2)

    with col1:
        if portfolio_performance_pct_data:
            fig_pct = create_performance_bar_chart(
                portfolio_performance_pct_data,
                title="% Change",
                exclude_stablecoins=False,
            )
            fig_pct.update_layout(height=500)
            st.plotly_chart(fig_pct, width="stretch")

    with col2:
        if portfolio_performance_dollar_data:
            fig_dollar = create_performance_bar_chart(
                portfolio_performance_dollar_data,
                title="$ Change",
                exclude_stablecoins=False,
            )
            fig_dollar.update_layout(height=500)
            st.plotly_chart(fig_dollar, width="stretch")


def show_asset_management_all_portfolios():
    """Show asset management for All Portfolios view with portfolio selection.

    Fetches individual assets directly for proper asset management functionality.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("➕ Add Asset")

        # Get all portfolios for selection
        portfolios = st.session_state.portfolio_manager.get_all_portfolios()
        if not portfolios:
            st.info("No portfolios available. Create a portfolio first.")
            return

        # Portfolio selection for adding assets
        portfolio_options = {p.name: p.id for p in portfolios}
        selected_portfolio_name = st.selectbox(
            "Select Portfolio to Add Asset To",
            options=list(portfolio_options.keys()),
            help="Choose which portfolio to add the new asset to",
        )
        selected_portfolio_id = portfolio_options[selected_portfolio_name]

        # Use the existing add asset form logic but with portfolio selection
        show_add_asset_form(selected_portfolio_id)

        # Portfolio management (rename/delete)
        st.markdown("---")
        show_portfolio_management()

    with col2:
        st.subheader("✏️ Edit/Delete Assets")

        # Get individual assets (not aggregated) for editing/deleting
        all_individual_assets = st.session_state.portfolio_manager.get_all_assets()

        if not all_individual_assets:
            st.info("No assets to manage.")
            return

        # Asset selection for editing/deleting - show individual assets with portfolio info
        asset_options = {}
        for asset in all_individual_assets:
            # Find the portfolio for this asset
            portfolio = next(
                (p for p in portfolios if p.id == asset.portfolio_id), None
            )
            portfolio_name = portfolio.name if portfolio else "Unknown"
            display_name = f"{asset.symbol} ({portfolio_name}) - {asset.quantity:.6f}"
            asset_options[display_name] = asset

        selected_asset_name = st.selectbox(
            "Select Asset to Edit/Delete",
            options=list(asset_options.keys()),
            help="Choose an asset to edit or delete",
        )

        if selected_asset_name:
            selected_asset = asset_options[selected_asset_name]

            # Show edit form for selected asset
            show_edit_asset_form([selected_asset])

            st.markdown("---")

            # Show delete/sell form for selected asset
            show_delete_sell_asset_form([selected_asset])
