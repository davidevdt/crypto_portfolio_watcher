"""
Take Profit Page - Enhanced profit tracking with timeline chart
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional

from components.shared import (
    format_currency,
    format_percentage,
    format_number,
    format_crypto_quantity,
    send_desktop_notification,
    show_empty_state,
    show_empty_profit_history,
)


def show():
    """Main take profit tracking page."""
    st.title("💰 Take Profit History")

    # Profit history summary
    show_profit_summary()

    # Profit timeline chart
    show_profit_timeline()

    # Profit history table
    show_profit_history_table()

    # Quick sell interface
    show_quick_sell_interface()


def show_profit_summary():
    """Show profit statistics summary."""
    try:
        profit_stats = st.session_state.portfolio_manager.get_total_realized_profit()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_profit = profit_stats.get("total_profit", 0)
            st.metric("Total Realized Profit", format_currency(total_profit))

        with col2:
            trade_count = profit_stats.get("trade_count", 0)
            st.metric("Total Trades", trade_count)

        with col3:
            avg_profit = total_profit / max(trade_count, 1) if trade_count > 0 else 0
            st.metric("Average Profit/Trade", format_currency(avg_profit))

        with col4:
            # Calculate success rate (profitable trades)
            try:
                profit_history = st.session_state.portfolio_manager.get_profit_history()
                profitable_trades = len(
                    [p for p in profit_history if p.realized_profit > 0]
                )
                success_rate = (
                    (profitable_trades / max(trade_count, 1) * 100)
                    if trade_count > 0
                    else 0
                )
                st.metric("Success Rate", f"{success_rate:.1f}%")
            except:
                st.metric("Success Rate", "N/A")

    except Exception as e:
        st.error(f"Error loading profit statistics: {e}")


def show_profit_timeline():
    """Show profit timeline chart with blue line."""
    st.subheader("📈 Profit Timeline")

    try:
        profit_history = st.session_state.portfolio_manager.get_profit_history()

        if profit_history:
            # Prepare data for timeline
            timeline_data = []
            cumulative_profit = 0

            for record in reversed(
                profit_history
            ):  # Reverse to get chronological order
                cumulative_profit += record.realized_profit
                timeline_data.append(
                    {
                        "date": record.sold_at,
                        "profit": record.realized_profit,
                        "cumulative_profit": cumulative_profit,
                        "symbol": record.symbol,
                    }
                )

            df_timeline = pd.DataFrame(timeline_data)

            # Create simple line plot
            fig = go.Figure()

            # Add cumulative profit line (simple line only)
            fig.add_trace(
                go.Scatter(
                    x=df_timeline["date"],
                    y=df_timeline["cumulative_profit"],
                    mode="lines",
                    name="Cumulative Profit",
                    line=dict(color="#007bff", width=2),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                    + "Cumulative Profit: $%{y:,.2f}<br>"
                    + "<extra></extra>",
                )
            )

            fig.update_layout(
                title="Profit Timeline - Cumulative Realized Profits",
                xaxis_title="Date",
                yaxis_title="Cumulative Profit ($)",
                hovermode="closest",
                height=500,
                showlegend=True,
            )

            # Add horizontal line at break-even
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            st.plotly_chart(fig, width="stretch")

            # Show some timeline stats
            col1, col2, col3 = st.columns(3)

            with col1:
                best_trade = (
                    df_timeline.loc[df_timeline["profit"].idxmax()]
                    if len(df_timeline) > 0
                    else None
                )
                if best_trade is not None:
                    st.metric(
                        "Best Trade",
                        format_currency(best_trade["profit"]),
                        delta=f"{best_trade['symbol']}",
                    )

            with col2:
                worst_trade = (
                    df_timeline.loc[df_timeline["profit"].idxmin()]
                    if len(df_timeline) > 0
                    else None
                )
                if worst_trade is not None:
                    st.metric(
                        "Worst Trade",
                        format_currency(worst_trade["profit"]),
                        delta=f"{worst_trade['symbol']}",
                    )

            with col3:
                if len(df_timeline) >= 2:
                    recent_trend = (
                        df_timeline.iloc[-3:]["profit"].mean()
                        if len(df_timeline) >= 3
                        else df_timeline.iloc[-2:]["profit"].mean()
                    )
                    st.metric(
                        "Recent Trend",
                        format_currency(recent_trend),
                        delta="Last few trades avg",
                    )

        else:
            st.info(
                "No realized profits yet. Profits will appear here when you sell assets."
            )

    except Exception as e:
        st.error(f"Error creating profit timeline: {e}")


def show_profit_history_table():
    """Show detailed profit history table."""
    st.subheader("📊 Realized Profits History")

    try:
        profit_history = st.session_state.portfolio_manager.get_profit_history()

        if profit_history:
            history_data = []

            for record in profit_history:
                history_data.append(
                    {
                        "Date": record.sold_at.strftime("%Y-%m-%d %H:%M"),
                        "Symbol": record.symbol,
                        "Quantity Sold": f"{record.quantity_sold:,.6f}",
                        "Sell Price": format_currency(record.sell_price, 4),
                        "Buy Price": format_currency(record.average_buy_price, 4),
                        "Sale Value": format_currency(
                            record.quantity_sold * record.sell_price
                        ),
                        "Cost Basis": format_currency(
                            record.quantity_sold * record.average_buy_price
                        ),
                        "Realized Profit": format_currency(record.realized_profit),
                        "Profit %": format_percentage(record.profit_percentage),
                        "Portfolio": record.portfolio_name,
                    }
                )

            df = pd.DataFrame(history_data)

            # Add filters
            col1, col2, col3 = st.columns(3)

            with col1:
                # Portfolio filter
                portfolios = df["Portfolio"].unique().tolist()
                selected_portfolio = st.selectbox(
                    "Filter by Portfolio",
                    ["All"] + portfolios,
                    key="profit_portfolio_filter",
                )

            with col2:
                # Symbol filter
                symbols = df["Symbol"].unique().tolist()
                selected_symbol = st.selectbox(
                    "Filter by Symbol", ["All"] + symbols, key="profit_symbol_filter"
                )

            with col3:
                # Profit type filter
                profit_type = st.selectbox(
                    "Filter by Type",
                    ["All", "Profitable Only", "Losses Only"],
                    key="profit_type_filter",
                )

            # Apply filters
            filtered_df = df.copy()

            if selected_portfolio != "All":
                filtered_df = filtered_df[
                    filtered_df["Portfolio"] == selected_portfolio
                ]

            if selected_symbol != "All":
                filtered_df = filtered_df[filtered_df["Symbol"] == selected_symbol]

            if profit_type == "Profitable Only":
                filtered_df = filtered_df[
                    filtered_df["Realized Profit"]
                    .str.replace("$", "")
                    .str.replace(",", "")
                    .astype(float)
                    > 0
                ]
            elif profit_type == "Losses Only":
                filtered_df = filtered_df[
                    filtered_df["Realized Profit"]
                    .str.replace("$", "")
                    .str.replace(",", "")
                    .astype(float)
                    < 0
                ]

            # Style profit columns
            def style_profit(val):
                if val.startswith("$-") or val.startswith("-"):
                    return "color: #dc3545"
                elif val.startswith("$") and not val.startswith("$-"):
                    profit_val = float(val.replace("$", "").replace(",", ""))
                    return "color: #28a745" if profit_val > 0 else ""
                elif val.startswith("+"):
                    return "color: #28a745"
                return ""

            if len(filtered_df) > 0:
                styled_df = filtered_df.style.map(
                    style_profit, subset=["Realized Profit", "Profit %"]
                )
                st.dataframe(styled_df, width="stretch")

                # Show filtered summary
                col1, col2, col3 = st.columns(3)

                profits = (
                    filtered_df["Realized Profit"]
                    .str.replace("$", "")
                    .str.replace(",", "")
                    .astype(float)
                )

                with col1:
                    st.metric("Filtered Total", format_currency(profits.sum()))
                with col2:
                    st.metric("Filtered Trades", len(filtered_df))
                with col3:
                    avg_filtered = profits.mean() if len(profits) > 0 else 0
                    st.metric("Filtered Average", format_currency(avg_filtered))
            else:
                st.info("No records match the selected filters.")

            # Export options
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Export to CSV"):
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"profit_history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )

            with col2:
                if st.button("🗑️ Clear History"):
                    with st.expander("⚠️ Confirm Deletion"):
                        st.warning("This action cannot be undone!")
                        if st.checkbox(
                            "I understand this will delete all profit history"
                        ):
                            if st.button("🗑️ Delete All Records", type="secondary"):
                                deleted_count = (
                                    st.session_state.portfolio_manager.clear_profit_history()
                                )
                                st.success(f"Cleared {deleted_count} profit records")
                                st.rerun()

        else:
            st.info(
                "No realized profits yet. Profits will appear here when you sell assets."
            )

    except Exception as e:
        st.error(f"Error loading profit history: {e}")


def show_quick_sell_interface():
    """Enhanced quick sell interface with asset table updates."""
    st.subheader("🔄 Quick Sell")

    # Portfolio selection for quick sell
    portfolios = st.session_state.portfolio_manager.get_all_portfolios()
    if portfolios:
        portfolio_options = {p.name: p.id for p in portfolios}
        selected_portfolio_name = st.selectbox(
            "Select Portfolio for Quick Sell",
            list(portfolio_options.keys()),
            key="quick_sell_portfolio",
        )
        selected_portfolio_id = portfolio_options[selected_portfolio_name]
    else:
        show_empty_state(
            title="No Portfolios Available",
            message="Create a portfolio and add some assets to start using the take profit features.",
            icon="💰",
        )
        return

    assets = st.session_state.portfolio_manager.get_portfolio_assets(
        selected_portfolio_id
    )

    if assets:
        # Exclude stablecoins from selling
        sellable_assets = [
            a for a in assets if a.symbol not in ["USDT", "USDC", "BUSD"]
        ]

        if sellable_assets:
            with st.form("quick_sell"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    asset_options = {
                        f"{a.symbol} ({a.quantity:.6f})": a for a in sellable_assets
                    }
                    selected_asset_key = st.selectbox(
                        "Asset to Sell", list(asset_options.keys())
                    )
                    asset = asset_options[selected_asset_key]

                    # Show asset info
                    current_price = st.session_state.current_prices.get(
                        asset.symbol, asset.average_buy_price
                    )
                    st.write(f"**Current Price:** {format_currency(current_price, 4)}")
                    st.write(
                        f"**Avg Buy Price:** {format_currency(asset.average_buy_price, 4)}"
                    )

                with col2:
                    max_quantity = float(asset.quantity)
                    sell_quantity = st.number_input(
                        "Quantity to Sell",
                        min_value=0.000001,
                        max_value=max_quantity,
                        value=max_quantity,  # Default to selling all
                        step=0.000001,
                        format="%.6f",
                    )

                    # Quick quantity buttons
                    col_25, col_50, col_75, col_100 = st.columns(4)
                    with col_25:
                        if st.form_submit_button("25%"):
                            sell_quantity = max_quantity * 0.25
                    with col_50:
                        if st.form_submit_button("50%"):
                            sell_quantity = max_quantity * 0.50
                    with col_75:
                        if st.form_submit_button("75%"):
                            sell_quantity = max_quantity * 0.75
                    with col_100:
                        if st.form_submit_button("100%"):
                            sell_quantity = max_quantity

                with col3:
                    sell_price = st.number_input(
                        "Sell Price ($)",
                        value=float(current_price),
                        min_value=0.01,
                        step=0.01,
                        format="%.4f",
                    )

                    # Choose which stablecoin to receive
                    stable_coin = st.selectbox(
                        "Receive proceeds in",
                        ["USDT", "USDC"],
                        help="Choose which stablecoin to receive the sale proceeds in",
                    )

                # Profit calculation preview
                if sell_quantity > 0 and sell_price > 0:
                    profit_preview = (
                        sell_price - asset.average_buy_price
                    ) * sell_quantity
                    profit_pct = (
                        (sell_price - asset.average_buy_price) / asset.average_buy_price
                    ) * 100
                    sale_value = sell_quantity * sell_price

                    st.markdown("---")
                    st.subheader("📊 Sale Preview")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sale Value", format_currency(sale_value))
                    with col2:
                        st.metric(
                            "Cost Basis",
                            format_currency(sell_quantity * asset.average_buy_price),
                        )
                    with col3:
                        st.metric("Estimated Profit", format_currency(profit_preview))
                    with col4:
                        st.metric("Profit %", format_percentage(profit_pct))

                submitted = st.form_submit_button("💰 Execute Sale", type="primary")

                if submitted and sell_quantity > 0 and sell_price > 0:
                    try:
                        # Execute the sale
                        remaining_asset, profit_record = (
                            st.session_state.portfolio_manager.sell_asset(
                                asset.id, sell_quantity, sell_price
                            )
                        )

                        # Add proceeds to stablecoin balance
                        proceeds = sell_quantity * sell_price
                        st.session_state.portfolio_manager.add_asset(
                            portfolio_id=selected_portfolio_id,
                            symbol=stable_coin,
                            quantity=proceeds,
                            buy_price=1.0,
                        )

                        # Success message
                        st.success(
                            f"""
                        ✅ Sale executed successfully!
                        
                        **Sold:** {sell_quantity} {asset.symbol}
                        **Price:** {format_currency(sell_price, 4)}
                        **Total Proceeds:** {format_currency(proceeds)} {stable_coin}
                        **Profit:** {format_currency(profit_record.realized_profit)} ({format_percentage(profit_record.profit_percentage)})
                        """
                        )

                        # Send notification if enabled
                        if st.session_state.settings.get("desktop_notifications"):
                            send_desktop_notification(
                                "Sale Executed",
                                f"Sold {format_crypto_quantity(sell_quantity)} {asset.symbol} for {format_currency(profit_record.realized_profit)} profit",
                            )

                        # Update current prices to reflect the sale
                        st.session_state.current_prices[stable_coin] = 1.0

                        # Show updated portfolio info
                        st.info(
                            "Portfolio has been updated. The asset table will reflect the changes."
                        )

                        # Force page refresh after a short delay
                        import time

                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error executing sale: {e}")
        else:
            st.info(
                "No sellable assets available (stablecoins are excluded from quick sell)"
            )
    else:
        st.info("No assets to sell in the selected portfolio")
