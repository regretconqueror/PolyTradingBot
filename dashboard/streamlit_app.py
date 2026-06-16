"""
Streamlit Dashboard for PolyTradingBot
Provides real-time visualization of trading activities, market data, portfolio performance, and risk metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.trading_bot import PolymarketTradingBot
from core import PortfolioConstraints, Market
from strategies import EnsembleModel, WeightedMovingAverageModel, VolatilityAdjustedModel, SimpleEdgeModel
from config import load_settings
import time

# Page configuration
st.set_page_config(
    page_title="PolyTradingBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .positive { color: #00C851; }
    .negative { color: #ff4444; }
    .neutral { color: #ffbb33; }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_bot():
    """Initialize the trading bot with current settings"""
    try:
        settings = load_settings()

        constraints = PortfolioConstraints(
            max_total_exposure=settings.max_exposure,
            max_single_position=settings.max_position,
            max_drawdown=settings.max_drawdown,
            min_bet_size=settings.min_bet_size
        )

        # Use EnsembleModel as default (can be made configurable)
        model = EnsembleModel()

        bot = PolymarketTradingBot(
            capital=settings.capital,
            constraints=constraints,
            model=model,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            passphrase=settings.api_passphrase,
            private_key=settings.private_key,
            funder_address=settings.funder_address,
            signature_type=settings.signature_type,
            live_trading_enabled=settings.live_trading_enabled,
            live_dry_run=settings.live_dry_run,
            max_live_order_size=settings.max_live_order_size,
            max_live_orders_per_cycle=settings.max_live_orders_per_cycle,
            paper_mode=settings.paper_mode,
            enable_yes_no_arb=True
        )
        return bot, settings
    except Exception as e:
        st.error(f"Failed to initialize bot: {str(e)}")
        return None, None

def fetch_markets_data(bot):
    """Fetch and format market data for display"""
    try:
        markets = bot.fetch_markets(min_edge=0.01)  # Lower threshold to see more markets
        if not markets:
            return pd.DataFrame()

        # Convert to DataFrame for easy manipulation
        markets_data = []
        for market in markets:
            markets_data.append({
                'Market': market.question[:50] + ('...' if len(market.question) > 50 else ''),
                'Full Question': market.question,
                'Category': market.category,
                'Current Price': f"{market.price:.1%}",
                'Your Probability': f"{market.probability:.1%}",
                'Edge': f"{market.edge:.1%}",
                'Liquidity': f"${market.liquidity:,.0f}",
                'Volume 24h': f"${market.volume_24h:,.0f}",
                'Resolution Date': market.resolution_date[:10] if market.resolution_date else 'N/A',
                'Token ID': market.token_id[:8] + '...',
                'Condition ID': market.condition_id[:8] + '...',
                'raw_market': market  # Keep reference for debugging
            })

        return pd.DataFrame(markets_data)
    except Exception as e:
        st.error(f"Error fetching markets: {str(e)}")
        return pd.DataFrame()

def create_allocations_chart(allocations_df):
    """Create a pie chart showing portfolio allocations"""
    if allocations_df.empty or 'Allocation (%)' not in allocations_df.columns:
        return None

    # Filter out zero allocations
    df_nonzero = allocations_df[allocations_df['Allocation (%)'] > 0.1]
    if df_nonzero.empty:
        return None

    fig = px.pie(
        df_nonzero,
        values='Allocation (%)',
        names='Market',
        title='Portfolio Allocation by Market',
        hover_data=['Size ($)', 'Edge'],
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    return fig

def create_performance_chart(performance_history):
    """Create a line chart showing performance over time"""
    if not performance_history or len(performance_history) < 2:
        return None

    df = pd.DataFrame(performance_history)
    if 'timestamp' not in df.columns or 'total_pnl' not in df.columns:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_pnl'],
        mode='lines+markers',
        name='Total P&L ($)',
        line=dict(color='#00C851', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title='Portfolio Performance Over Time',
        xaxis_title='Time',
        yaxis_title='P&L ($)',
        hovermode='x unified',
        height=300
    )
    return fig

def create_risk_gauge(risk_metrics):
    """Create a gauge chart for risk metrics"""
    fig = go.Figure()

    # Exposure gauge
    exposure = risk_metrics.get('exposure_ratio', 0) * 100  # Convert to percentage
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = exposure,
        domain = {'x': [0, 0.5], 'y': [0.2, 0.8]},
        title = {'text': "Total Exposure (%)"},
        delta = {'reference': 75},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    # VaR gauge
    var_95 = risk_metrics.get('var_95', 0)
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = var_95,
        domain = {'x': [0.5, 1], 'y': [0.2, 0.8]},
        title = {'text': "VaR 95% ($)"},
        gauge = {
            'axis': {'range': [None, max(var_95*2, 100)]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, var_95*0.5], 'color': "lightgray"},
                {'range': [var_95*0.5, var_95], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': var_95*1.5
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def display_market_table(markets_df):
    """Display the markets data in an interactive table"""
    if markets_df.empty:
        st.warning("No market data available")
        return

    # Select columns to display
    display_columns = ['Market', 'Category', 'Current Price', 'Your Probability', 'Edge', 'Liquidity', 'Volume 24h']
    display_df = markets_df[display_columns].copy()

    # Style the dataframe
    def highlight_edge(val):
        if isinstance(val, str) and '%' in val:
            try:
                numeric_val = float(val.replace('%', ''))
                if numeric_val > 0.1:
                    return 'background-color: #d4edda'
                elif numeric_val < -0.1:
                    return 'background-color: #f8d7da'
                else:
                    return ''
            except:
                return ''
        return ''

    styled_df = display_df.style.map(highlight_edge, subset=['Edge'])
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

    # Show detailed view when a row is selected
    if st.checkbox("Show Detailed Market Info"):
        selected_index = st.selectbox(
            "Select a market to view details:",
            range(len(markets_df)),
            format_func=lambda x: markets_df.iloc[x]['Market']
        )

        if selected_index is not None:
            market = markets_df.iloc[selected_index]
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Market Details")
                st.write(f"**Question:** {market['Full Question']}")
                st.write(f"**Category:** {market['Category']}")
                st.write(f"**Resolution Date:** {market['Resolution Date']}")
                st.write(f"**Condition ID:** {market['Condition ID']}")

            with col2:
                st.subheader("Trading Information")
                st.write(f"**Current Price:** {market['Current Price']}")
                st.write(f"**Your Probability:** {market['Your Probability']}")
                st.write(f"**Edge:** {market['Edge']}")
                st.write(f"**Liquidity:** {market['Liquidity']}")
                st.write(f"**24h Volume:** {market['Volume 24h']}")

def display_allocations_table(allocations_df):
    """Display portfolio allocations in a table"""
    if allocations_df.empty:
        st.info("No active positions")
        return

    display_columns = ['Market', 'Direction', 'Size ($)', 'Allocation (%)', 'Market Price', 'Your Probability', 'Edge']
    display_df = allocations_df[display_columns].copy() if all(col in allocations_df.columns for col in display_columns) else allocations_df

    st.dataframe(
        display_df.style.format({
            'Size ($)': '${:,.2f}',
            'Allocation (%)': '{:.1f}%',
            'Market Price': '{:.1%}',
            'Your Probability': '{:.1%}',
            'Edge': '{:.1%}'
        }),
        use_container_width=True
    )

def main():
    """Main dashboard application"""
    st.title("🤖 PolyTradingBot Dashboard")
    st.markdown("--")

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 30)

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data") or auto_refresh:
        st.rerun()

    # Trading mode selection
    st.sidebar.subheader("Trading Mode")
    paper_mode = st.sidebar.checkbox("Paper Trading Mode", value=True, help="Enable for simulated trading")

    # Capital adjustment
    capital = st.sidebar.number_input(
        "Trading Capital ($)",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Available capital for trading"
    )

    # Risk parameters
    st.sidebar.subheader("Risk Parameters")
    max_exposure = st.sidebar.slider("Max Total Exposure (%)", 10, 100, 75, help="Maximum percentage of capital to risk")
    max_position = st.sidebar.slider("Max Single Position (%)", 1, 50, 20, help="Maximum percentage per single position")
    max_drawdown = st.sidebar.slider("Max Drawdown (%)", 5, 50, 15, help="Maximum allowed drawdown before stopping")

    # Model selection
    st.sidebar.subheader("Probability Model")
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["Ensemble Model (Recommended)", "Weighted Moving Average", "Volatility Adjusted", "Simple Edge (Baseline)"],
        help="Choose the probability estimation model"
    )

    # Initialize bot
    with st.spinner("Initializing trading bot..."):
        bot, settings = initialize_bot()

    if bot is None:
        st.error("Failed to initialize trading bot. Please check your configuration.")
        return

    # Update bot settings based on sidebar selections
    bot.paper_mode = paper_mode
    bot.capital = float(capital)
    bot.constraints.max_total_exposure = max_exposure / 100.0
    bot.constraints.max_single_position = max_position / 100.0
    bot.constraints.max_drawdown = max_drawdown / 100.0

    # Set model based on selection
    if model_option == "Weighted Moving Average":
        bot.model = WeightedMovingAverageModel()
    elif model_option == "Volatility Adjusted":
        bot.model = VolatilityAdjustedModel()
    elif model_option == "Simple Edge (Baseline)":
        bot.model = SimpleEdgeModel()
    # Else keep EnsembleModel (default)

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Overview",
        "💼 Portfolio",
        "📈 Performance",
        "⚠️ Risk Management",
        "🚨 Alerts & Monitoring",
        "📋 Orders & Trades"
    ])

    # Fetch data
    with st.spinner("Fetching market data..."):
        markets_df = fetch_markets_data(bot)

    # Tab 1: Market Overview
    with tab1:
        st.header("Market Opportunities")

        if markets_df.empty:
            st.warning("No market data available. Check your API connection or try again later.")
        else:
            # Market summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Markets", len(markets_df))
            with col2:
                avg_edge = markets_df['Edge'].str.replace('%', '').astype(float).mean() if len(markets_df) > 0 else 0
                st.metric("Average Edge", f"{avg_edge:.1f}%")
            with col3:
                high_edge_markets = len(markets_df[markets_df['Edge'].str.replace('%', '').astype(float) > 0.1])
                st.metric("High Edge (>10%)", high_edge_markets)
            with col4:
                total_liquidity = markets_df['Liquidity'].str.replace('$', '').str.replace(',', '').astype(float).sum()
                st.metric("Total Liquidity", f"${total_liquidity:,.0f}")

            st.markdown("---")

            # Market table
            display_market_table(markets_df)

            # Market analysis charts
            if len(markets_df) > 0:
                st.subheader("Market Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    # Edge distribution
                    edge_numeric = markets_df['Edge'].str.replace('%', '').astype(float)
                    fig_edge = px.histogram(
                        x=edge_numeric,
                        nbins=20,
                        title="Edge Distribution",
                        labels={'x': 'Edge (%)', 'y': 'Number of Markets'},
                        color_discrete_sequence=['#00C851']
                    )
                    fig_edge.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_edge, use_container_width=True)

                with col2:
                    # Price vs Probability scatter
                    price_numeric = markets_df['Current Price'].str.replace('%', '').astype(float) / 100
                    prob_numeric = markets_df['Your Probability'].str.replace('%', '').astype(float) / 100

                    # Create a DataFrame for plotly to work with properly
                    scatter_df = pd.DataFrame({
                        'price': price_numeric,
                        'probability': prob_numeric,
                        'size': markets_df['Liquidity'].str.replace('$', '').str.replace(',', '').astype(float),
                        'edge': edge_numeric,
                        'market': markets_df['Market']
                    })

                    fig_scatter = px.scatter(
                        scatter_df,
                        x='price',
                        y='probability',
                        size='size',
                        color='edge',
                        hover_data=['market'],
                        title="Market Price vs Your Probability",
                        labels={'price': 'Market Price', 'probability': 'Your Probability', 'size': 'Liquidity ($)', 'edge': 'Edge (%)', 'market': 'Market'},
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0
                    )
                    # Add diagonal line (perfect prediction)
                    fig_scatter.add_shape(
                        type="line", line=dict(dash="dash"),
                        x0=0, y0=0, x1=1, y1=1
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

    # Tab 2: Portfolio
    with tab2:
        st.header("Portfolio Allocation")

        # Generate current allocations based on bot's optimization
        if not markets_df.empty:
            try:
                markets_for_optimization = markets_df['raw_market'].tolist()

                # Run optimization with current bot settings
                allocations, status, info = bot.optimizer.optimize(
                    markets_for_optimization,
                    bot.constraints
                )

                logger.info(f"Optimization {status.value} in {info.get('iterations', 0)} iterations")

                # Convert allocations to display format
                allocations_data = []
                for i, (market, alloc) in enumerate(zip(markets_for_optimization, allocations)):
                    if alloc < 0.001:  # Skip negligible allocations
                        continue

                    size_usd = alloc * bot.capital
                    direction = "YES" if market.edge > 0 else "NO"

                    allocations_data.append({
                        'Market': market.question[:50] + ('...' if len(market.question) > 50 else ''),
                        'Full Question': market.question,
                        'Direction': direction,
                        'Size ($)': size_usd,
                        'Allocation (%)': alloc * 100,
                        'Market Price': market.price,
                        'Your Probability': market.probability,
                        'Edge': market.edge,
                        'Category': market.category,
                        'Token ID': market.token_id[:8] + '...',
                        'Condition ID': market.condition_id[:8] + '...'
                    })

                allocations_df = pd.DataFrame(allocations_data)

                if not allocations_df.empty:
                    # Allocation summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Active Positions", len(allocations_df))
                    with col2:
                        total_allocation = allocations_df['Allocation (%)'].sum()
                        st.metric("Total Allocation", f"{total_allocation:.1f}%")
                    with col3:
                        avg_position_size = allocations_df['Size ($)'].mean()
                        st.metric("Avg Position Size", f"${avg_position_size:,.0f}")

                    st.markdown("---")

                    # Allocation charts and table
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        allocations_fig = create_allocations_chart(allocations_df)
                        if allocations_fig:
                            st.plotly_chart(allocations_fig, use_container_width=True)
                        else:
                            st.info("No significant allocations to display")

                    with col2:
                        st.subheader("Position Details")
                        display_allocations_table(allocations_df)

                    # Show optimization info
                    with st.expander("Optimization Details"):
                        st.write(f"**Status:** {status.value}")
                        st.write(f"**Iterations:** {info.get('iterations', 'N/A')}")
                        st.write(f"**Expected Log Utility:** {info.get('final_objective', 'N/A'):.6f}")
                        st.write(f"**Frank-Wolfe Gap:** {info.get('fw_gap', 'N/A'):.6e}")

                else:
                    st.info("No significant allocations generated by optimizer")

            except Exception as e:
                st.error(f"Error generating portfolio data: {str(e)}")
                st.info("This is expected during initialization or when no clear opportunities exist")
                # Fallback to placeholder
                st.info("💡 Portfolio optimization runs during each trading cycle. Check the 'Performance' tab for latest allocation data.")

                # Placeholder for allocation data
                allocations_data = []
                if len(markets_df) > 0:
                    # Show top 5 markets by edge as example allocation
                    # Convert Edge to numeric for sorting
                    markets_df['Edge_Numeric'] = markets_df['Edge'].str.replace('%', '').astype(float)
                    top_markets = markets_df.nlargest(5, 'Edge_Numeric') if len(markets_df) >= 5 else markets_df
                    total_edge = top_markets['Edge_Numeric'].sum()

                    for idx, (_, market) in enumerate(top_markets.iterrows()):
                        edge_val = float(market['Edge'].replace('%', ''))
                        allocation_pct = (edge_val / total_edge) * 100 if total_edge > 0 else 0
                        size_usd = (allocation_pct / 100) * float(capital)

                        allocations_data.append({
                            'Market': market['Market'],
                            'Direction': 'YES' if float(market['Your Probability'].replace('%', '')) > float(market['Current Price'].replace('%', '')) else 'NO',
                            'Size ($)': size_usd,
                            'Allocation (%)': allocation_pct,
                            'Market Price': float(market['Current Price'].replace('%', '')) / 100,
                            'Your Probability': float(market['Your Probability'].replace('%', '')) / 100,
                            'Edge': float(market['Edge'].replace('%', '')) / 100
                        })

                allocations_df = pd.DataFrame(allocations_data)

                if not allocations_df.empty:
                    # Allocation summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Active Positions", len(allocations_df))
                    with col2:
                        total_allocation = allocations_df['Allocation (%)'].sum()
                        st.metric("Total Allocation", f"{total_allocation:.1f}%")
                    with col3:
                        avg_position_size = allocations_df['Size ($)'].mean()
                        st.metric("Avg Position Size", f"${avg_position_size:,.0f}")

                    st.markdown("---")

                    # Allocation charts and table
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        allocations_fig = create_allocations_chart(allocations_df)
                        if allocations_fig:
                            st.plotly_chart(allocations_fig, use_container_width=True)
                        else:
                            st.info("No significant allocations to display")

                    with col2:
                        st.subheader("Position Details")
                        display_allocations_table(allocations_df)
                else:
                    st.info("No current allocations to display")
        else:
            st.warning("No market data available for portfolio allocation")

    # Tab 3: Performance
    with tab3:
        st.header("Performance Analytics")

        # Get performance metrics from bot
        try:
            perf = bot.get_performance_summary()

            # Display key metrics (direct dict access, not string parsing)
            col1, col2, col3, col4 = st.columns(4)

            total_trades = perf.get('total_trades', 0)
            win_rate = perf.get('win_rate', 0.0) * 100
            total_pnl = perf.get('total_pnl', 0.0)
            profit_factor = perf.get('profit_factor', 0.0)

            with col1:
                st.metric("Total Trades", f"{total_trades}")
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            with col4:
                st.metric("Profit Factor", f"{profit_factor:.2f}")

            st.markdown("---")

            # Performance details
            st.subheader("Performance Details")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Total Trades:** {total_trades}")
                st.write(f"**Winning Trades:** {perf.get('winning_trades', 0)}")
                st.write(f"**Losing Trades:** {perf.get('losing_trades', 0)}")
                st.write(f"**Win Rate:** {win_rate:.1f}%")
            with col_b:
                st.write(f"**Avg Win:** ${perf.get('avg_win', 0):.2f}")
                st.write(f"**Avg Loss:** ${perf.get('avg_loss', 0):.2f}")
                st.write(f"**Profit Factor:** {profit_factor:.2f}")
                st.write(f"**Max Drawdown:** {perf.get('max_drawdown', 0):.1%}")
            st.write(f"**Last Updated:** {perf.get('last_updated', 'N/A')}")

            # Placeholder for performance history chart
            st.info("💡 Performance history chart will appear after multiple trading cycles are recorded")

            # Show recent trades if available
            if hasattr(bot, 'trade_history') and bot.trade_history:
                st.subheader("Recent Trades")
                trades_df = pd.DataFrame(bot.trade_history[-10:])  # Last 10 trades
                if not trades_df.empty:
                    # Select relevant columns for display
                    display_cols = ['market', 'direction', 'size', 'allocation', 'edge', 'timestamp']
                    available_cols = [col for col in display_cols if col in trades_df.columns]
                    if available_cols:
                        display_trades = trades_df[available_cols].copy()
                        st.dataframe(display_trades, use_container_width=True)
                    else:
                        st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trade history available yet")
            else:
                st.info("No trade history available yet")

        except Exception as e:
            st.error(f"Error loading performance data: {str(e)}")

    # Tab 4: Risk Management
    with tab4:
        st.header("Risk Management")

        try:
            # Get risk summary from bot
            # We need current prices for risk calculation - use latest market data
            current_prices = {}
            if not markets_df.empty:
                for _, market in markets_df.iterrows():
                    raw_market = market.get('raw_market')
                    token_id = raw_market.token_id if raw_market is not None else market['Token ID']
                    price_str = market['Current Price'].replace('%', '')
                    try:
                        price = float(price_str) / 100.0
                        current_prices[token_id] = price
                    except:
                        pass

            # Get risk summary (this will use the bot's risk manager)
            risk_summary = bot.risk_manager.get_risk_summary(current_prices)

            # Display risk metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Active Positions", risk_summary.get('total_positions', 0))
            with col2:
                exposure_pct = risk_summary.get('exposure_ratio', 0) * 100
                st.metric("Total Exposure", f"{exposure_pct:.1f}%")
            with col3:
                var_amount = risk_summary.get('var_95', 0)
                st.metric("VaR 95%", f"${var_amount:,.2f}")
            with col4:
                max_corr = risk_summary.get('max_correlation', 0)
                st.metric("Max Correlation", f"{max_corr:.2f}")

            st.markdown("---")

            # Risk gauges
            if current_prices:  # Only show if we have price data
                risk_fig = create_risk_gauge(risk_summary)
                st.plotly_chart(risk_fig, use_container_width=True)
            else:
                st.info("Risk gauges will appear when market data is available")

            # Risk details
            st.subheader("Risk Details")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Position Limits**")
                st.write(f"- Max Single Position: {max_position}%")
                st.write(f"- Max Total Exposure: {max_exposure}%")
                st.write(f"- Max Drawdown: {max_drawdown}%")
                st.write(f"- Min Bet Size: {settings.min_bet_size*100}%")

            with col2:
                st.write("**Current Risk Status**")
                violations = risk_summary.get('risk_violations', [])
                if violations:
                    st.error("⚠️ Risk Violations Detected:")
                    for violation in violations:
                        st.write(f"- {violation.get('message', 'Unknown violation')}")
                else:
                    st.success("✅ All risk limits within bounds")

                # Correlation warnings
                high_corr_pairs = risk_summary.get('high_correlation_pairs', [])
                if high_corr_pairs:
                    st.warning("⚠️ High Correlation Positions:")
                    for token1, token2, corr in high_corr_pairs[:3]:  # Show top 3
                        st.write(f"- {token1[:6]}... & {token2[:6]}...: {corr:.2f}")
                else:
                    st.info("No high correlation position pairs detected")

        except Exception as e:
            st.error(f"Error loading risk data: {str(e)}")
            st.info("Risk management data will be available after market data is loaded")

    # Tab 5: Orders & Trades
    with tab5:
        st.header("Order Management & Trade History")

        # Sub-tabs for orders and trades
        order_tab1, order_tab2 = st.tabs(["📋 Active Orders", "📜 Trade History"])

        with order_tab1:
            st.subheader("Active Orders")
            st.info("💡 Active orders section will show real-time order status when connected to live trading")

            # Placeholder for active orders
            if hasattr(bot, 'open_orders') and bot.open_orders:
                orders_df = pd.DataFrame(list(bot.open_orders.values()))
                st.dataframe(orders_df, use_container_width=True)
            else:
                st.info("No active orders at the moment")

                # Show example of what orders would look like
                st.subheader("Order Format Example")
                example_order = {
                    'Order ID': 'ord_123abc',
                    'Market': 'Will Germany win the 2026 FIFA World Cup?',
                    'Direction': 'BUY YES',
                    'Size': '$661.01',
                    'Price': '0.051',
                    'Status': 'OPEN',
                    'Timestamp': '2026-05-14 00:27:16'
                }
                st.json(example_order)

        with order_tab2:
            st.subheader("Trade History")

            if hasattr(bot, 'trade_history') and bot.trade_history:
                trades_df = pd.DataFrame(bot.trade_history)

                # Display options
                col1, col2 = st.columns([3, 1])
                with col1:
                    show_all = st.checkbox("Show All Trades", value=False)
                with col2:
                    if not show_all:
                        limit = st.selectbox("Show Last N Trades", [5, 10, 20, 50], index=1)

                # Filter trades
                if show_all:
                    display_trades = trades_df
                else:
                    display_trades = trades_df.tail(limit) if len(trades_df) > limit else trades_df

                if not display_trades.empty:
                    # Format for display
                    display_cols = ['market', 'direction', 'size', 'allocation', 'edge', 'timestamp', 'status']
                    available_cols = [col for col in display_cols if col in display_trades.columns]

                    if available_cols:
                        formatted_trades = display_trades[available_cols].copy()

                        # Format numeric columns
                        if 'size' in formatted_trades.columns:
                            formatted_trades['size'] = formatted_trades['size'].apply(lambda x: f"${float(x):,.2f}" if isinstance(x, (int, float)) else x)
                        if 'allocation' in formatted_trades.columns:
                            formatted_trades['allocation'] = formatted_trades['allocation'].apply(lambda x: f"{float(x):.1f}%" if isinstance(x, (int, float)) else x)
                        if 'edge' in formatted_trades.columns:
                            formatted_trades['edge'] = formatted_trades['edge'].apply(lambda x: f"{float(x):.1f}%" if isinstance(x, (int, float)) else x)

                        st.dataframe(formatted_trades, use_container_width=True)
                    else:
                        st.dataframe(display_trades, use_container_width=True)

                    # Trade statistics
                    st.subheader("Trade Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", len(trades_df))
                    with col2:
                        winning_trades = len(trades_df[trades_df.get('status', '') == 'filled']) if 'status' in trades_df.columns else 0
                        st.metric("Winning Trades", winning_trades)
                    with col3:
                        losing_trades = len(trades_df[trades_df.get('status', '') == 'failed']) if 'status' in trades_df.columns else 0
                        st.metric("Losing Trades", losing_trades)
                    with col4:
                        win_rate = (winning_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                else:
                    st.info("No trade history available yet")
            else:
                st.info("No trade history available yet")
                st.subheader("Expected Trade Format")
                example_trade = {
                    'market': 'Will Germany win the 2026 FIFA World Cup?',
                    'direction': 'BUY YES',
                    'size': 661.01,
                    'allocation': 0.066,
                    'edge': 0.409,
                    'timestamp': '2026-05-14 00:27:16.241077',
                    'status': 'filled',
                    'execution_price': 0.051,
                    'order_id': 'ord_123abc'
                }
                st.json(example_trade)

    # Tab 6: Alerts & Monitoring
    with tab6:
        st.header("🚨 Alerts & Monitoring")

        # Get alerts from bot
        try:
            if hasattr(bot, 'alert_manager'):
                alerts = bot.alert_manager.get_recent_alerts(limit=50)

                if alerts:
                    # Alert summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    # Count alerts by level
                    alert_levels = {}
                    for alert in alerts:
                        level = alert.get('level', 'info')
                        alert_levels[level] = alert_levels.get(level, 0) + 1

                    with col1:
                        st.metric("Total Alerts", len(alerts))
                    with col2:
                        st.metric("Critical", alert_levels.get('critical', 0))
                    with col3:
                        st.metric("Errors", alert_levels.get('error', 0))
                    with col4:
                        st.metric("Warnings", alert_levels.get('warning', 0))

                    st.markdown("---")

                    # Alert filter
                    filter_col1, filter_col2 = st.columns([3, 1])
                    with filter_col1:
                        alert_filter = st.multiselect(
                            "Filter by Level",
                            options=['critical', 'error', 'warning', 'info'],
                            default=['critical', 'error', 'warning', 'info']
                        )
                    with filter_col2:
                        if st.button("Clear Alerts"):
                            # Clear alerts file
                            with open(bot.alert_manager.alert_file, 'w') as f:
                                json.dump([], f)
                            st.rerun()

                    # Filter alerts
                    filtered_alerts = [alert for alert in alerts if alert.get('level', 'info') in alert_filter]

                    if filtered_alerts:
                        # Display alerts in reverse chronological order (newest first)
                        for alert in reversed(filtered_alerts):
                            level = alert.get('level', 'info')
                            source = alert.get('source', 'unknown')
                            message = alert.get('message', 'No message')
                            timestamp = alert.get('timestamp', 'Unknown time')

                            # Style based on level
                            if level == 'critical':
                                st.error(f"🚨 **{source}** [{timestamp}]: {message}")
                            elif level == 'error':
                                st.error(f"❌ **{source}** [{timestamp}]: {message}")
                            elif level == 'warning':
                                st.warning(f"⚠️ **{source}** [{timestamp}]: {message}")
                            else:
                                st.info(f"ℹ️ **{source}** [{timestamp}]: {message}")

                            # Show metadata if available
                            metadata = alert.get('metadata', {})
                            if metadata:
                                with st.expander(f"Details for {source} alert"):
                                    st.json(metadata)
                    else:
                        st.info("No alerts match the current filter")
                else:
                    st.info("No alerts have been generated yet")

                    # Show bot health metrics
                    st.subheader("Bot Health Metrics")
                    health_col1, health_col2, health_col3, health_col4 = st.columns(4)

                    with health_col1:
                        st.metric("Cycles Completed", getattr(bot, 'cycles_completed', 0))
                    with health_col2:
                        st.metric("Successful Cycles", getattr(bot, 'successful_cycles', 0))
                    with health_col3:
                        st.metric("Failed Cycles", getattr(bot, 'failed_cycles', 0))
                    with health_col4:
                        success_rate = (getattr(bot, 'successful_cycles', 0) / max(getattr(bot, 'cycles_completed', 1), 1)) * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")

                    # API health
                    st.subheader("API Health")
                    api_col1, api_col2, api_col3 = st.columns(3)
                    with api_col1:
                        st.metric("API Failures", getattr(bot, 'api_failures', 0))
                    with api_col2:
                        st.metric("Max Failures Threshold", getattr(bot, 'max_api_failures', 5))
                    with api_col3:
                        last_fetch = getattr(bot, 'last_successful_fetch', None)
                        if last_fetch:
                            st.metric("Last Successful Fetch", last_fetch.strftime("%H:%M:%S"))
                        else:
                            st.metric("Last Successful Fetch", "Never")
            else:
                st.warning("Alert manager not available in bot instance")

        except Exception as e:
            st.error(f"Error loading alerts: {str(e)}")
            st.info("Alerts & monitoring will be fully available after bot initialization")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>PolyTradingBot Dashboard | Built with Streamlit |
            <a href='https://github.com/yourusername/polymarket-trading-bot' target='_blank'>GitHub</a></p>
            <p><em>Delay: Data refreshed on demand. For live trading, ensure proper API configuration.</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
