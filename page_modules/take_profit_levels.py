"""
Take Profit Levels Page - Advanced TP strategies with real-time profit preview
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
    send_whatsapp_notification,
)


def show():
    """Main take profit levels and DCA strategy page."""
    st.title("🎯 Take Profit Levels")

    # Asset selection
    assets = st.session_state.portfolio_manager.get_all_assets()

    if not assets:
        st.info("No assets available for take profit level management.")
        return

    # Filter out stablecoins
    tradeable_assets = [
        asset for asset in assets if asset.symbol not in ["USDT", "USDC", "BUSD"]
    ]

    if not tradeable_assets:
        st.info("No tradeable assets available (stablecoins excluded).")
        return

    # Asset selection for TP levels
    st.subheader("📊 Take Profit Level Management")

    asset_options = {
        f"{a.symbol} - {a.portfolio.name if hasattr(a, 'portfolio') and a.portfolio else 'Unknown'} (Qty: {format_crypto_quantity(a.quantity)}, Avg: {format_currency(a.average_buy_price, 4)})": a
        for a in tradeable_assets
    }

    selected_asset_key = st.selectbox(
        "Select Asset for Take Profit Management", list(asset_options.keys())
    )
    selected_asset = asset_options[selected_asset_key]

    # Show current asset info
    current_price = st.session_state.current_prices.get(
        selected_asset.symbol, selected_asset.average_buy_price
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", format_currency(current_price, 4))
    with col2:
        st.metric("Avg Buy Price", format_currency(selected_asset.average_buy_price, 4))
    with col3:
        current_value = selected_asset.quantity * current_price
        st.metric("Current Value", format_currency(current_value))
    with col4:
        unrealized_pnl = current_value - (
            selected_asset.quantity * selected_asset.average_buy_price
        )
        unrealized_pct = (
            (
                unrealized_pnl
                / (selected_asset.quantity * selected_asset.average_buy_price)
            )
            * 100
            if selected_asset.average_buy_price > 0
            else 0
        )
        st.metric(
            "Unrealized P&L",
            format_currency(unrealized_pnl),
            delta=format_percentage(unrealized_pct),
        )

    # Show existing TP levels
    show_existing_tp_levels(selected_asset, current_price)

    # Create new TP levels
    show_tp_level_creation(selected_asset, current_price)


def show_existing_tp_levels(asset, current_price: float):
    """Show existing take profit levels for the selected asset."""
    st.subheader(f"🎯 Active Take Profit Levels for {asset.symbol}")

    try:
        tp_levels = st.session_state.portfolio_manager.get_take_profit_levels(asset.id)

        if tp_levels:
            levels_data = []
            total_expected_profit = 0

            for level in tp_levels:
                profit_pct = (
                    (level.target_price - asset.average_buy_price)
                    / asset.average_buy_price
                ) * 100
                quantity_to_sell = asset.quantity * (level.percentage_to_sell / 100)
                expected_profit = (
                    level.target_price - asset.average_buy_price
                ) * quantity_to_sell

                # Determine status
                if not level.is_active:
                    status = "🟢 Triggered" if level.triggered_at else "❌ Inactive"
                elif current_price >= level.target_price:
                    status = "🎯 Target Reached"
                else:
                    status = "⏳ Waiting"

                total_expected_profit += expected_profit if level.is_active else 0

                levels_data.append(
                    {
                        "ID": level.id,
                        "Target Price": format_currency(level.target_price, 4),
                        "Profit %": f"{profit_pct:+,.1f}%",
                        "Sell %": f"{level.percentage_to_sell:.1f}%",
                        "Quantity": f"{quantity_to_sell:.6f}",
                        "Expected Profit": format_currency(expected_profit),
                        "Strategy": level.strategy_type.replace("_", " ").title(),
                        "Status": status,
                        "Notes": level.notes or "",
                        "Created": level.created_at.strftime("%Y-%m-%d"),
                    }
                )

            df = pd.DataFrame(levels_data)

            # Color code the status
            def style_status(val):
                if "Triggered" in val:
                    return "color: #28a745"
                elif "Target Reached" in val:
                    return "color: #ffc107; font-weight: bold"
                elif "Waiting" in val:
                    return "color: #007bff"
                else:
                    return "color: #6c757d"

            styled_df = df.style.map(style_status, subset=["Status"])
            st.dataframe(styled_df, width="stretch")

            # Show total expected profit
            if total_expected_profit > 0:
                st.success(
                    f"💰 **Total Expected Profit from Active Levels:** {format_currency(total_expected_profit)}"
                )

            # TP Level management controls
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🗑️ Clear All TP Levels"):
                    with st.expander("⚠️ Confirm Deletion"):
                        if st.checkbox(
                            "I understand this will delete all take profit levels"
                        ):
                            if st.button("Delete All Levels"):
                                for level in tp_levels:
                                    st.session_state.portfolio_manager.delete_take_profit_level(
                                        level.id
                                    )
                                st.success("All take profit levels cleared")
                                st.rerun()

            with col2:
                if st.button("🔄 Check TP Triggers"):
                    check_tp_triggers(asset, tp_levels, current_price)

            with col3:
                if st.button("📊 Show TP Chart"):
                    show_tp_visualization(asset, tp_levels, current_price)

        else:
            st.info(f"No take profit levels set for {asset.symbol}")

    except Exception as e:
        st.error(f"Error loading take profit levels: {e}")


def show_tp_level_creation(asset, current_price: float):
    """Show take profit level creation interface."""
    st.subheader("➕ Create Take Profit Levels")

    tab1, tab2, tab3 = st.tabs(["📈 Advanced Strategy", "📊 DCA Out", "📐 Fibonacci"])

    with tab1:
        show_advanced_strategy_creator(asset, current_price)

    with tab2:
        show_dca_out_strategy(asset, current_price)

    with tab3:
        show_fibonacci_strategy(asset, current_price)


def show_advanced_strategy_creator(asset, current_price: float):
    """Advanced strategy with multiple rounds and aggressiveness options."""
    st.subheader("🎯 Advanced Multi-Round Strategy")
    st.info("Create multiple take profit levels with configurable aggressiveness")

    # Real-time input controls (outside form for immediate updates)
    col1, col2, col3 = st.columns(3)

    with col1:
        # Choice between percentage or dollar amounts
        profit_type = st.radio(
            "Profit Type", ["Percentage (%)", "Dollar Amount ($)"], horizontal=True
        )

        if profit_type == "Percentage (%)":
            desired_profit_pct = st.number_input(
                "Desired Total Profit %",
                min_value=5.0,
                max_value=500.0,
                value=50.0,
                step=5.0,
                help="Total profit percentage you want to achieve",
            )
            # Convert to dollar amount for calculations
            current_asset_value = asset.quantity * current_price
            desired_profit_dollars = (desired_profit_pct / 100) * current_asset_value
        else:
            # Calculate reasonable defaults based on asset value
            current_asset_value = asset.quantity * current_price
            suggested_profit = (
                current_asset_value * 0.2
            )  # 20% of asset value as default
            max_reasonable_profit = current_asset_value * 5.0  # 5x asset value as max

            desired_profit_dollars = st.number_input(
                "Desired Total Profit ($)",
                min_value=1.0,
                max_value=max_reasonable_profit,
                value=max(10.0, min(suggested_profit, 1000.0)),  # Smart default
                step=max(1.0, current_asset_value * 0.01),  # Dynamic step size
                help=f"Total profit in dollar amount you want to achieve. Asset value: {format_currency(current_asset_value)}",
            )
            # Calculate equivalent percentage
            desired_profit_pct = (desired_profit_dollars / current_asset_value) * 100

        num_rounds = st.slider(
            "Number of Rounds",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of take profit levels to create",
        )

    with col2:
        aggressiveness = st.selectbox(
            "Strategy Aggressiveness",
            ["Conservative", "Neutral", "Aggressive"],
            index=1,
            help="Conservative: Sell more at lower prices, Aggressive: Sell more at higher prices",
        )

        min_profit_step = st.number_input(
            "Minimum Profit Step %",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="Minimum profit percentage between levels",
        )

    with col3:
        # Show real-time calculations
        st.metric("Current Asset Value", format_currency(current_asset_value))
        if profit_type == "Percentage (%)":
            st.metric("Target Profit ($)", format_currency(desired_profit_dollars))
        else:
            st.metric("Target Profit (%)", f"{desired_profit_pct:.1f}%")

    # Real-time strategy preview (outside form for immediate updates)
    st.subheader("📊 Real-time Strategy Preview")
    strategy_levels = calculate_advanced_strategy(
        asset, desired_profit_pct, num_rounds, aggressiveness, min_profit_step
    )

    if strategy_levels:
        preview_data = []
        total_quantity_sold = 0
        total_expected_profit = 0

        for i, level in enumerate(strategy_levels):
            quantity_to_sell = asset.quantity * (level["sell_percentage"] / 100)
            expected_profit = (
                level["target_price"] - asset.average_buy_price
            ) * quantity_to_sell
            total_quantity_sold += quantity_to_sell
            total_expected_profit += expected_profit

            preview_data.append(
                {
                    "Round": i + 1,
                    "Target Price": format_currency(level["target_price"], 4),
                    "Profit %": f"+{level['profit_pct']:.1f}%",
                    "Sell %": f"{level['sell_percentage']:.1f}%",
                    "Quantity": f"{quantity_to_sell:.6f}",
                    "Expected Profit": f"${expected_profit:.2f}",
                }
            )

        # Display preview table
        df_preview = pd.DataFrame(preview_data)
        st.dataframe(df_preview, width="stretch")

        # Show strategy summary with color coding
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Total Expected Profit", format_currency(total_expected_profit)
            )
        with col2:
            avg_profit_pct = (
                total_expected_profit / (asset.quantity * asset.average_buy_price) * 100
            )
            st.metric("📊 Total Profit %", f"{avg_profit_pct:.1f}%")
        with col3:
            sell_pct = (total_quantity_sold / asset.quantity) * 100
            st.metric("📦 Total Sell %", f"{sell_pct:.1f}%")
        with col4:
            remaining_quantity = asset.quantity - total_quantity_sold
            st.metric("🔄 Remaining Quantity", f"{remaining_quantity:.6f}")

    # Form for creating the strategy (only submit button inside form)
    with st.form("create_advanced_strategy"):
        st.info(
            "⬆️ Adjust parameters above to see real-time preview, then click create to save the strategy."
        )

        submitted = st.form_submit_button("🚀 Create Advanced Strategy", type="primary")

        if submitted:
            try:
                levels_created = create_advanced_strategy_levels(
                    asset, desired_profit, num_rounds, aggressiveness, min_profit_step
                )
                st.success(
                    f"Created {len(levels_created)} advanced take profit levels!"
                )
                st.rerun()

            except Exception as e:
                st.error(f"Error creating advanced strategy: {e}")


def show_dca_out_strategy(asset, current_price: float):
    """DCA out strategy creation."""
    st.subheader("📊 DCA Out Strategy")
    st.info("Create multiple take profit levels with increasing profit targets")

    with st.form("dca_out"):
        col1, col2, col3 = st.columns(3)

        with col1:
            num_levels = st.slider(
                "Number of Levels", min_value=2, max_value=10, value=5
            )
            start_profit = st.number_input(
                "Starting Profit %",
                min_value=5.0,
                max_value=100.0,
                value=15.0,
                step=5.0,
            )

        with col2:
            profit_increment = st.number_input(
                "Profit Increment %",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=5.0,
            )
            sell_percentage = st.slider(
                "Percentage to Sell per Level", min_value=5, max_value=50, value=20
            )

        with col3:
            # Preview toggle
            show_preview = st.checkbox("Show Preview", value=True, key="dca_preview")

        # Preview
        if show_preview:
            st.write("**DCA Out Levels Preview:**")
            preview_data = []
            total_expected_profit = 0

            for i in range(num_levels):
                profit_pct = start_profit + (i * profit_increment)
                target_price = asset.average_buy_price * (1 + profit_pct / 100)
                quantity = asset.quantity * (sell_percentage / 100)
                expected_profit = (target_price - asset.average_buy_price) * quantity
                total_expected_profit += expected_profit

                preview_data.append(
                    {
                        "Level": i + 1,
                        "Profit %": f"+{profit_pct:.1f}%",
                        "Target Price": format_currency(target_price, 4),
                        "Sell %": f"{sell_percentage:.1f}%",
                        "Quantity": f"{quantity:.6f}",
                        "Expected Profit": f"${expected_profit:.2f}",
                    }
                )

            df_preview = pd.DataFrame(preview_data)
            st.dataframe(df_preview, width="stretch")

            # Show total expected profit with color
            if total_expected_profit > 0:
                st.success(
                    f"💰 **Total Expected Profit:** {format_currency(total_expected_profit)}"
                )

        submitted = st.form_submit_button("📊 Create DCA Out Levels")

        if submitted:
            try:
                levels = st.session_state.portfolio_manager.generate_dca_out_levels(
                    asset_id=asset.id,
                    num_levels=num_levels,
                    profit_start=start_profit,
                    profit_increment=profit_increment,
                    sell_percentage=sell_percentage,
                )
                st.success(f"Created {len(levels)} DCA out levels!")
                st.rerun()

            except Exception as e:
                st.error(f"Error creating DCA out levels: {e}")


def show_fibonacci_strategy(asset, current_price: float):
    """Fibonacci retracement strategy."""
    st.subheader("📐 Fibonacci Strategy")
    st.info("Create take profit levels based on Fibonacci retracement levels")

    with st.form("fibonacci_tp"):
        col1, col2 = st.columns(2)

        with col1:
            max_target_price = st.number_input(
                "Maximum Target Price ($)",
                min_value=asset.average_buy_price,
                value=asset.average_buy_price * 2.5,
                step=0.01,
                format="%.4f",
                help="The highest price target for Fibonacci calculations",
            )

        with col2:
            base_sell_percentage = st.slider(
                "Base Sell Percentage",
                min_value=5,
                max_value=30,
                value=15,
                help="Base percentage to sell (will increase with higher Fib levels)",
            )

        # Preview Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        price_range = max_target_price - asset.average_buy_price

        st.write("**Fibonacci Levels Preview:**")
        preview_data = []
        total_expected_profit = 0

        for i, fib_ratio in enumerate(fib_levels):
            target_price = asset.average_buy_price + (price_range * fib_ratio)
            sell_percentage = base_sell_percentage + (
                i * 3.0
            )  # Increasing sell percentage
            quantity = asset.quantity * (sell_percentage / 100)
            profit_pct = (
                (target_price - asset.average_buy_price) / asset.average_buy_price
            ) * 100
            expected_profit = (target_price - asset.average_buy_price) * quantity
            total_expected_profit += expected_profit

            preview_data.append(
                {
                    "Fib Level": f"{fib_ratio:.3f}",
                    "Target Price": f"${target_price:.4f}",
                    "Profit %": f"+{profit_pct:.1f}%",
                    "Sell %": f"{sell_percentage:.1f}%",
                    "Quantity": f"{quantity:.6f}",
                    "Expected Profit": f"${expected_profit:.2f}",
                }
            )

        df_preview = pd.DataFrame(preview_data)
        st.dataframe(df_preview, width="stretch")

        if total_expected_profit > 0:
            st.success(f"💰 **Total Expected Profit:** ${total_expected_profit:.2f}")

        submitted = st.form_submit_button("📐 Create Fibonacci Levels")

        if submitted and max_target_price > asset.average_buy_price:
            try:
                levels = st.session_state.portfolio_manager.generate_fibonacci_levels(
                    asset_id=asset.id, max_price=max_target_price
                )
                st.success(f"Created {len(levels)} Fibonacci take profit levels!")
                st.rerun()

            except Exception as e:
                st.error(f"Error creating Fibonacci levels: {e}")


def calculate_advanced_strategy(
    asset, desired_profit: float, num_rounds: int, aggressiveness: str, min_step: float
) -> List[Dict]:
    """Calculate advanced strategy levels with aggressiveness adjustment."""
    try:
        levels = []

        # Calculate price targets
        max_profit = desired_profit
        profit_steps = []

        # Create profit steps based on aggressiveness
        if aggressiveness == "Conservative":
            # More levels at lower profit percentages
            base_steps = np.linspace(min_step, max_profit, num_rounds)
            profit_steps = [
                step**0.7 / (max_profit**0.7) * max_profit for step in base_steps
            ]
        elif aggressiveness == "Aggressive":
            # More levels at higher profit percentages
            base_steps = np.linspace(min_step, max_profit, num_rounds)
            profit_steps = [
                step**1.5 / (max_profit**1.5) * max_profit for step in base_steps
            ]
        else:  # Neutral
            profit_steps = np.linspace(min_step, max_profit, num_rounds)

        # Calculate sell percentages based on aggressiveness
        total_sell_pct = 80  # Don't sell everything
        sell_percentages = []

        if aggressiveness == "Conservative":
            # Sell more at lower prices
            weights = [(num_rounds - i) ** 1.2 for i in range(num_rounds)]
        elif aggressiveness == "Aggressive":
            # Sell more at higher prices
            weights = [(i + 1) ** 1.2 for i in range(num_rounds)]
        else:  # Neutral
            weights = [1] * num_rounds

        # Normalize weights
        weight_sum = sum(weights)
        sell_percentages = [(w / weight_sum) * total_sell_pct for w in weights]

        for i in range(num_rounds):
            profit_pct = profit_steps[i]
            target_price = asset.average_buy_price * (1 + profit_pct / 100)
            sell_pct = sell_percentages[i]
            quantity_to_sell = asset.quantity * (sell_pct / 100)
            expected_profit = (
                target_price - asset.average_buy_price
            ) * quantity_to_sell

            levels.append(
                {
                    "profit_pct": profit_pct,
                    "target_price": target_price,
                    "sell_percentage": sell_pct,
                    "quantity": quantity_to_sell,
                    "expected_profit": expected_profit,
                }
            )

        return levels

    except Exception as e:
        st.error(f"Error calculating strategy: {e}")
        return []


def create_advanced_strategy_levels(
    asset, desired_profit: float, num_rounds: int, aggressiveness: str, min_step: float
) -> List:
    """Create the calculated advanced strategy levels in the database."""
    strategy_levels = calculate_advanced_strategy(
        asset, desired_profit, num_rounds, aggressiveness, min_step
    )

    created_levels = []
    for level_data in strategy_levels:
        try:
            level = st.session_state.portfolio_manager.add_take_profit_level(
                asset_id=asset.id,
                target_price=level_data["target_price"],
                percentage_to_sell=level_data["sell_percentage"],
                strategy_type="advanced_multi",
                notes=f"Advanced {aggressiveness.lower()} strategy - Target: {level_data['profit_pct']:.1f}% profit",
            )
            created_levels.append(level)
        except Exception as e:
            st.warning(
                f"Could not create level at {format_currency(level_data['target_price'], 4)}: {e}"
            )

    return created_levels


def check_tp_triggers(asset, tp_levels: List, current_price: float):
    """Check and trigger take profit levels."""
    triggered_count = 0

    for level in tp_levels:
        if level.is_active and current_price >= level.target_price:
            try:
                # Execute the take profit level
                updated_asset, profit_record = (
                    st.session_state.portfolio_manager.trigger_take_profit_level(
                        level.id, current_price
                    )
                )
                triggered_count += 1

                # Create alert
                alert = {
                    "timestamp": datetime.now(),
                    "symbol": asset.symbol,
                    "type": "TP_TRIGGERED",
                    "message": f"Take Profit triggered at {format_currency(current_price, 4)} - Sold {level.percentage_to_sell:,.1f}% for {format_currency(profit_record.realized_profit)} profit",
                    "active": True,
                    "severity": "HIGH",
                    "profit": profit_record.realized_profit,
                }
                st.session_state.alerts.insert(0, alert)

                # Send notifications
                notification_title = f"Take Profit Triggered: {asset.symbol}"
                notification_message = f"Sold {level.percentage_to_sell:,.1f}% at {format_currency(current_price, 4)} for {format_currency(profit_record.realized_profit)} profit"

                if st.session_state.settings.get("desktop_notifications"):
                    send_desktop_notification(notification_title, notification_message)

                if st.session_state.settings.get("whatsapp_notifications"):
                    phone = st.session_state.settings.get("whatsapp_number")
                    if phone:
                        from components.shared import send_whatsapp_notification

                        send_whatsapp_notification(
                            f"{notification_title}: {notification_message}", phone
                        )

            except Exception as e:
                st.error(f"Error triggering TP level {level.id}: {e}")

    if triggered_count > 0:
        st.success(f"✅ Triggered {triggered_count} take profit levels!")
        st.rerun()
    else:
        st.info("No take profit levels triggered at current price")


def show_tp_visualization(asset, tp_levels: List, current_price: float):
    """Show visualization of take profit levels."""
    if not tp_levels:
        return

    st.subheader(f"📊 Take Profit Levels Visualization - {asset.symbol}")

    # Create price chart with TP levels
    fig = go.Figure()

    # Current price line
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color="blue",
        line_width=3,
        annotation_text=f"Current: {format_currency(current_price, 4)}",
        annotation_position="right",
    )

    # Average buy price line
    fig.add_hline(
        y=asset.average_buy_price,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Avg Buy: {format_currency(asset.average_buy_price, 4)}",
        annotation_position="right",
    )

    # TP level lines
    colors = ["green", "orange", "red", "purple", "brown", "pink", "olive", "cyan"]

    for i, level in enumerate(tp_levels):
        color = colors[i % len(colors)]
        line_style = "solid" if level.is_active else "dot"

        status = "Active" if level.is_active else "Triggered/Inactive"
        if level.is_active and current_price >= level.target_price:
            status = "Target Reached!"

        fig.add_hline(
            y=level.target_price,
            line_dash=line_style,
            line_color=color,
            annotation_text=f"TP {i+1}: {format_currency(level.target_price, 4)} ({level.percentage_to_sell:,.1f}%) - {status}",
            annotation_position="left",
        )

    fig.update_layout(
        title=f"Take Profit Levels for {asset.symbol}",
        yaxis_title="Price ($)",
        height=600,
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch")
