"""
Data Providers Management Page - Manage and configure data providers for each tracked asset.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List
from database.models import get_session, TrackedAsset
from services.portfolio_manager import PortfolioManager
from components.shared import show_empty_state


def show_data_providers_page():
    """Display data providers management interface."""
    st.header("📡 Data Providers Management")
    st.markdown(
        "Manage which exchange provides data for each tracked asset. "
        "The system automatically learns the best provider, but you can override these settings manually."
    )

    # Initialize portfolio manager
    pm = PortfolioManager()

    # Get all tracked assets
    tracked_assets = pm.get_tracked_assets(active_only=True)

    if not tracked_assets:
        show_empty_state(
            title="No Tracked Assets",
            message="Start by adding assets to your portfolio or watchlist. Once you have assets, you can manage their data providers here.",
            icon="📡",
        )
        return

    # Available providers
    available_providers = [
        "BinanceProvider",
        "BybitProvider",
        "MEXCProvider",
        "KrakenProvider",
    ]

    # Provider display names
    provider_names = {
        "BinanceProvider": "Binance",
        "BybitProvider": "Bybit",
        "MEXCProvider": "MEXC",
        "KrakenProvider": "Kraken",
    }

    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Provider Settings", "📊 Provider Statistics"])

    with tab1:
        st.subheader("Asset Provider Configuration")
        st.markdown(
            "Set the preferred data provider for each asset. Changes apply immediately."
        )

        # Create columns for the table
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            st.markdown("**Asset**")
        with col2:
            st.markdown("**Data Provider**")
        with col3:
            st.markdown("**Success Rate**")
        with col4:
            st.markdown("**Actions**")

        st.divider()

        # Display each asset with provider selection
        changes_made = False

        for asset in tracked_assets:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

            with col1:
                st.write(f"**{asset.symbol}**")

            with col2:
                # Current provider
                current_provider = asset.preferred_data_provider or "Auto"

                # Provider selection dropdown
                provider_options = ["Auto"] + [
                    provider_names[p] for p in available_providers
                ]
                current_index = 0

                if current_provider in provider_names:
                    current_display = provider_names[current_provider]
                    if current_display in provider_options:
                        current_index = provider_options.index(current_display)
                elif current_provider != "Auto":
                    current_index = 0  # Fallback to Auto

                selected_provider = st.selectbox(
                    "Provider",
                    provider_options,
                    index=current_index,
                    key=f"provider_{asset.symbol}",
                    label_visibility="collapsed",
                )

                # Convert back to internal name
                if selected_provider == "Auto":
                    new_provider = None
                else:
                    # Find the internal name
                    new_provider = None
                    for internal_name, display_name in provider_names.items():
                        if display_name == selected_provider:
                            new_provider = internal_name
                            break

                # Update if changed
                if new_provider != asset.preferred_data_provider:
                    pm.update_asset_provider(asset.symbol, new_provider)
                    changes_made = True

            with col3:
                # Calculate success rate
                total_attempts = (asset.provider_success_count or 0) + (
                    asset.provider_fail_count or 0
                )
                if total_attempts > 0:
                    success_rate = (
                        (asset.provider_success_count or 0) / total_attempts * 100
                    )
                    color = (
                        "green"
                        if success_rate >= 80
                        else "orange" if success_rate >= 60 else "red"
                    )
                    st.markdown(
                        f"<span style='color: {color}'>{success_rate:.1f}%</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("N/A")

            with col4:
                # Test button
                if st.button(
                    "🧪 Test", key=f"test_{asset.symbol}", help="Test data availability"
                ):
                    test_provider_for_asset(asset.symbol, new_provider or "Auto")

        if changes_made:
            st.success("✅ Provider settings updated!")
            st.rerun()

    with tab2:
        st.subheader("Provider Performance Statistics")

        # Create summary statistics
        provider_stats = {}
        total_success = 0
        total_failures = 0

        for asset in tracked_assets:
            provider = asset.preferred_data_provider or "Auto"
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "assets": 0,
                    "success": 0,
                    "failures": 0,
                    "success_rate": 0,
                }

            provider_stats[provider]["assets"] += 1
            provider_stats[provider]["success"] += asset.provider_success_count or 0
            provider_stats[provider]["failures"] += asset.provider_fail_count or 0

            total_success += asset.provider_success_count or 0
            total_failures += asset.provider_fail_count or 0

        # Calculate success rates
        for provider in provider_stats:
            total = (
                provider_stats[provider]["success"]
                + provider_stats[provider]["failures"]
            )
            if total > 0:
                provider_stats[provider]["success_rate"] = (
                    provider_stats[provider]["success"] / total * 100
                )

        # Create DataFrame for display
        stats_data = []
        for provider, stats in provider_stats.items():
            display_name = provider_names.get(provider, provider)
            stats_data.append(
                {
                    "Provider": display_name,
                    "Assets": stats["assets"],
                    "Successes": stats["success"],
                    "Failures": stats["failures"],
                    "Success Rate": (
                        f"{stats['success_rate']:.1f}%"
                        if stats["success_rate"] > 0
                        else "N/A"
                    ),
                }
            )

        if stats_data:
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)

            # Overall statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Assets", len(tracked_assets))
            with col2:
                st.metric("Total API Calls", total_success + total_failures)
            with col3:
                overall_rate = (
                    total_success / (total_success + total_failures) * 100
                    if total_success + total_failures > 0
                    else 0
                )
                st.metric("Overall Success Rate", f"{overall_rate:.1f}%")
        else:
            st.info("No provider statistics available yet.")


def test_provider_for_asset(symbol: str, provider: str):
    """Test data availability for an asset with a specific provider."""
    import asyncio
    from data_providers.data_fetcher import CryptoPriceFetcher
    from datetime import datetime, timedelta

    with st.spinner(f"Testing {symbol} with {provider}..."):
        try:
            fetcher = CryptoPriceFetcher()

            # Test current price
            current_price = asyncio.run(fetcher.get_real_time_price(symbol))

            # Test historical data (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            historical_data = asyncio.run(
                fetcher.get_historical_data(symbol, start_date, end_date, "1d")
            )

            if current_price and historical_data:
                st.success(
                    f"✅ {symbol}: Current price ${current_price:.4f}, {len(historical_data)} historical points available"
                )
            elif current_price:
                st.warning(
                    f"⚠️ {symbol}: Current price ${current_price:.4f} available, but no historical data"
                )
            else:
                st.error(f"❌ {symbol}: No data available")

        except Exception as e:
            st.error(f"❌ {symbol}: Test failed - {str(e)}")


if __name__ == "__main__":
    show_data_providers_page()
