#!/usr/bin/env python3
"""
Crypto Portfolio Tracker - Refactored Streamlit Application
A comprehensive crypto portfolio monitoring application with real-time tracking,
technical analysis, alerts, and multi-exchange data integration.
"""

import streamlit as st
import logging
from components.shared import (
    init_session_state,
    load_custom_css,
    smart_price_refresh,
    poll_database_continuously,
)

# Import page modules
from page_modules import (
    portfolio_overview,
    asset_charts,
    monitoring,
    take_profit,
    take_profit_levels,
    watchlist,
    settings,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Crypto Portfolio Tracker",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def sidebar_navigation():
    """
    Create sidebar navigation and controls.

    Returns:
        str: The selected page identifier
    """
    st.sidebar.title("₿ Crypto Portfolio")

    # Navigation menu
    pages = {
        "📊 Portfolio Overview": "portfolio_overview",
        "📈 Asset Charts": "asset_charts",
        "🔔 Monitoring": "monitoring",
        "💰 Take Profit": "take_profit",
        "🎯 Take Profit Levels": "take_profit_levels",
        "👀 Watchlist": "watchlist",
        "⚙️ Settings": "settings",
    }

    selected_page = st.sidebar.selectbox("Navigation", list(pages.keys()))

    # Add portfolio selector under Navigation for Portfolio Overview page
    if selected_page == "📊 Portfolio Overview":
        from components.shared import portfolio_selector
        
        # Always call portfolio selector for Portfolio Overview page
        # It will handle the empty state (no portfolios) properly
        portfolio_selector(use_sidebar=True)

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "🔄 Refresh Data", type="primary", help="Refresh all data from database"
    ):
        with st.spinner("Refreshing all data..."):
            if hasattr(st.session_state, "historical_data"):
                st.session_state.historical_data.clear()

            success = smart_price_refresh(force_refresh=True)
            if success and st.session_state.current_prices:
                st.sidebar.success(
                    f"✅ Refreshed {len(st.session_state.current_prices)} assets"
                )
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to refresh data")

    try:
        from database.models import get_session, CachedPrice

        session = get_session()
        latest_price = (
            session.query(CachedPrice).order_by(CachedPrice.last_updated.desc()).first()
        )
        if latest_price:
            last_update_utc = latest_price.last_updated.strftime("%m/%d/%Y %H:%M")
            st.sidebar.markdown(
                f"**Last Update**<br><small>{last_update_utc} UTC</small>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                "**Last Update**<br><small>No data</small>", unsafe_allow_html=True
            )
        session.close()
    except Exception as e:
        st.sidebar.markdown("**Last update**: Error")

    poll_database_continuously()

    try:
        portfolios = st.session_state.portfolio_manager.get_all_portfolios()
        if portfolios:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Quick Stats**")

            consistent_sidebar_prices = st.session_state.current_prices

            if st.session_state.selected_portfolio == "all":
                summary = st.session_state.portfolio_manager.get_all_portfolios_summary(
                    consistent_sidebar_prices
                )
            else:
                summary = st.session_state.portfolio_manager.get_portfolio_summary(
                    st.session_state.selected_portfolio, consistent_sidebar_prices
                )

            total_value = summary.get("total_value", 0)
            total_return = summary.get("total_return", 0)
            return_pct = summary.get("total_return_percentage", 0)

            st.sidebar.metric("Portfolio Value", f"${total_value:,.2f}")

            if total_return >= 0:
                st.sidebar.markdown(
                    f"<span class='profit-positive'>↗️ +${total_return:,.2f} ({return_pct:+.1f}%)</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.sidebar.markdown(
                    f"<span class='profit-negative'>↘️ ${total_return:,.2f} ({return_pct:+.1f}%)</span>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.sidebar.error(f"Error loading stats: {e}")

    active_alerts = [a for a in st.session_state.alerts if a.get("active", True)]
    if active_alerts:
        st.sidebar.markdown(f"🚨 **{len(active_alerts)} Active Alerts**")

    return pages[selected_page]


def main():
    """
    Main application entry point.

    Initializes session state, loads CSS, handles navigation,
    and routes to appropriate page modules.
    """
    init_session_state()
    load_custom_css()

    try:
        cached_prices = st.session_state.portfolio_manager.get_cached_prices()
        if cached_prices and not st.session_state.current_prices:
            st.session_state.current_prices.update(cached_prices)
            logger.info(f"Loaded {len(cached_prices)} cached prices on app startup")
    except Exception as e:
        logger.error(f"Error loading cached prices on startup: {e}")

    selected_page = sidebar_navigation()

    if selected_page == "portfolio_overview":
        portfolio_overview.show()
    elif selected_page == "asset_charts":
        asset_charts.show()
    elif selected_page == "monitoring":
        monitoring.show()
    elif selected_page == "take_profit":
        take_profit.show()
    elif selected_page == "take_profit_levels":
        take_profit_levels.show()
    elif selected_page == "watchlist":
        watchlist.show()
    elif selected_page == "settings":
        settings.show()


if __name__ == "__main__":
    main()
