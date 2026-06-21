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
import requests

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.trading_bot import PolymarketTradingBot
from core import PortfolioConstraints, Market
from strategies import EnsembleModel, WeightedMovingAverageModel, VolatilityAdjustedModel, SimpleEdgeModel
from config import load_settings
from backtest.historical_data import (
    HistoricalDataManager,
    BacktestEngine,
    ensemble_edge_strategy,
    buy_cheap_strategy,
)
import time

def update_config_toml(mode: str):
    """Dynamically update config.toml for Glide Data Grid alignment without crashing."""
    try:
        config_dir = Path(__file__).parent.parent / ".streamlit"
        config_path = config_dir / "config.toml"
        if mode == "Light Mode":
            config_content = """[theme]
primaryColor = "#8b5cf6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f1f5f9"
textColor = "#0f172a"
font = "sans serif"
"""
        else:
            config_content = """[theme]
primaryColor = "#a78bfa"
backgroundColor = "#12131c"
secondaryBackgroundColor = "#191b27"
textColor = "#d4d6e0"
font = "sans serif"
"""
        current_content = ""
        if config_path.exists():
            current_content = config_path.read_text(encoding="utf-8")
        if current_content.strip() != config_content.strip():
            config_dir.mkdir(exist_ok=True)
            config_path.write_text(config_content, encoding="utf-8")
            st.rerun()
    except Exception as e:
        logger.error(f"Failed to update config.toml: {e}")

# Resolve theme configuration
config_path = Path(__file__).parent.parent / ".streamlit" / "config.toml"
initial_theme_index = 0
if config_path.exists():
    try:
        if "#ffffff" in config_path.read_text(encoding="utf-8"):
            initial_theme_index = 1
    except Exception:
        pass
theme_mode = "Light Mode" if initial_theme_index == 1 else "Dark Mode"

# Plotly theme layouts
DARK_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(8,9,13,0)',
    plot_bgcolor='rgba(18,19,28,0.5)',
    font=dict(family='Inter, sans-serif', color='#d4d6e0', size=12),
    title_font=dict(color='#a78bfa', size=15, family='Space Grotesk, sans-serif'),
    xaxis=dict(gridcolor='#2a2d42', zerolinecolor='#2a2d42', title_font=dict(color='#8b8fa8')),
    yaxis=dict(gridcolor='#2a2d42', zerolinecolor='#2a2d42', title_font=dict(color='#8b8fa8')),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(font=dict(color='#8b8fa8', size=11)),
)

LIGHT_LAYOUT = dict(
    template='plotly_white',
    paper_bgcolor='rgba(255,255,255,0)',
    plot_bgcolor='rgba(241,245,249,0.5)',
    font=dict(family='Inter, sans-serif', color='#334155', size=12),
    title_font=dict(color='#7c3aed', size=15, family='Space Grotesk, sans-serif'),
    xaxis=dict(gridcolor='#cbd5e1', zerolinecolor='#cbd5e1', title_font=dict(color='#64748b')),
    yaxis=dict(gridcolor='#cbd5e1', zerolinecolor='#cbd5e1', title_font=dict(color='#64748b')),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(font=dict(color='#64748b', size=11)),
)

plotly_layout = LIGHT_LAYOUT if theme_mode == "Light Mode" else DARK_LAYOUT

# Page configuration
st.set_page_config(
    page_title="PolyTradingBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# Dark "vibe" theme. Mirrors the colors configured in .streamlit/config.toml
# so the injected CSS reinforces rather than fights the base theme.
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-surface: #191b27;
        --bg-card: #1f2230;
        --text-bright: #f5f6fa;
        --text-secondary: #b4b8cc;
        --text-muted: #8b8fa8;
        --accent-purple: #a78bfa;
        --accent-cyan: #22d3ee;
        --accent-green: #10b981;
        --accent-rose: #fb7185;
        --border: #2a2d42;
        --radius-lg: 14px;
        --gradient-hero: linear-gradient(90deg, #a78bfa, #22d3ee);
    }

    /* Force the dark palette onto Streamlit primitives */
    .stApp { background: #12131c; color: var(--text-secondary); }
    section[data-testid="stSidebar"] { background: var(--bg-surface); }

    .positive { color: var(--accent-green); }
    .negative { color: var(--accent-rose); }
    .neutral { color: #ffbb33; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1rem 1.1rem !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] div {
        color: var(--text-bright) !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* Tab bar */
    [data-testid="stTabs"] {
        background: var(--bg-surface) !important;
        border-radius: var(--radius-lg) !important;
        padding: 5px !important;
        border: 1px solid var(--border) !important;
        gap: 2px !important;
    }

    /* Section divider */
    .section-line {
        height: 2px;
        background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan), transparent);
        border: none;
        margin: 0.5rem 0 1.25rem;
        border-radius: 1px;
    }

    /* Status dots (header + live indicators) */
    .status-dot {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 1.8s infinite ease-in-out;
    }
    .status-dot.live { background: var(--accent-green); box-shadow: 0 0 8px rgba(16,185,129,0.5); }
    .status-dot.off  { background: var(--text-muted); animation: none; }
    .status-dot.err  { background: var(--accent-rose); box-shadow: 0 0 8px rgba(251,113,133,0.4); }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.3); }
    }

    /* Vibe card (used by the Crypto Scalper widgets) */
    .vibe-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
    }

    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(251,113,133,0.08);
        border: 1px solid rgba(251,113,133,0.3);
        color: var(--accent-rose);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .gradient-text {
        background: var(--gradient-hero);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--accent-purple);
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Popover (wallet details menu) */
    [data-testid="stPopoverBody"] {
        max-width: 180px !important;
        min-width: 0px !important;
        padding: 14px !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }

    /* Footer */
    .dashboard-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: var(--text-muted);
        font-size: 0.8rem;
        font-family: 'Space Grotesk', sans-serif;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
    .dashboard-footer a { color: var(--accent-purple); }

    /* Subtle canvas noise texture overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        z-index: 0;
        opacity: 0.015;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
        background-size: 128px 128px;
    }
</style>
""", unsafe_allow_html=True)

if theme_mode == "Light Mode":
    st.markdown("""
    <style>
        :root {
            /* Override official Streamlit theme variables to force light mode for widgets and Glide Grid */
            --background-color:           #ffffff !important;
            --secondary-background-color: #f1f5f9 !important;
            --text-color:                 #0f172a !important;
            --primary-color:              #8b5cf6 !important;

            --bg-void:       #f8fafc;
            --bg-base:       #f1f5f9;
            --bg-surface:    #ffffff;
            --bg-elevated:   #e2e8f0;
            --bg-card:       #ffffff;
            --bg-hover:      #f1f5f9;
            --border:        #cbd5e1;
            --border-light:  #94a3b8;
            --border-glow:   rgba(139, 92, 246, 0.08);
            --text-bright:   #0f172a;
            --text-primary:  #334155;
            --text-secondary:#64748b;
            --text-muted:    #94a3b8;
            --accent-purple: #8b5cf6;
            --accent-violet: #7c3aed;
            --accent-indigo: #4f46e5;
            --accent-cyan:   #06b6d4;
            --accent-teal:   #0d9488;
            --accent-green:  #10b981;
            --accent-emerald:#059669;
            --accent-amber:  #d97706;
            --accent-orange: #ea580c;
            --accent-rose:   #e11d48;
            --accent-red:    #dc2626;
            --accent-pink:   #db2777;
            --gradient-hero:  linear-gradient(135deg, #7c3aed 0%, #4f46e5 40%, #06b6d4 100%);
            --gradient-card:  linear-gradient(135deg, rgba(124,58,237,0.03) 0%, rgba(6,182,212,0.01) 100%);
            --gradient-green: linear-gradient(135deg, #059669 0%, #10b981 100%);
            --gradient-danger:linear-gradient(135deg, #dc2626 0%, #e11d48 100%);
            --glow-purple:   0 0 20px rgba(124,58,237,0.05), 0 0 60px rgba(124,58,237,0.02);
            --glow-green:    0 0 20px rgba(5,150,105,0.05), 0 0 60px rgba(5,150,105,0.02);
            --glow-cyan:     0 0 20px rgba(6,182,212,0.05), 0 0 60px rgba(6,182,212,0.02);
        }

        /* ── Light Mode: Dropdown Menu overrides ────────────────────────── */
        div[data-baseweb="menu"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }
        div[data-baseweb="menu"] li, div[data-baseweb="menu"] div[role="option"] {
            background-color: #ffffff !important;
            color: #334155 !important;
        }
        div[data-baseweb="menu"] li:hover, div[data-baseweb="menu"] div[role="option"]:hover {
            background-color: #f1f5f9 !important;
            color: #7c3aed !important;
        }

        /* ── Light Mode: Global Backgrounds ─────────────────────────── */
        .stApp, [data-testid="stHeader"] {
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] {
            background-color: var(--bg-surface) !important;
            border-right: 1px solid var(--border) !important;
        }

        /* ── Light Mode: Typography ─────────────────────────────────── */
        h1, h2, h3, h4, h5, h6, label, p, span, div {
            color: var(--text-primary) !important;
        }
        h1 {
            background: var(--gradient-hero) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        /* ── Light Mode: Cards ──────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text-bright) !important;
        }
        .vibe-card {
            background: linear-gradient(135deg, rgba(124,58,237,0.03) 0%, rgba(6,182,212,0.01) 100%), #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8) !important;
        }
        .vibe-card:hover {
            border-color: rgba(124,58,237,0.25) !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 0 20px rgba(124,58,237,0.04), inset 0 1px 0 rgba(255,255,255,0.8) !important;
        }

        /* ── Light Mode: Dividers ───────────────────────────────────── */
        hr {
            background: linear-gradient(90deg, transparent 0%, #cbd5e1 20%, #cbd5e1 80%, transparent 100%) !important;
        }
        .section-line {
            background: linear-gradient(90deg, #7c3aed, #06b6d4, transparent) !important;
        }

        /* ── Light Mode: Scrollbar ──────────────────────────────────── */
        ::-webkit-scrollbar-track { background: #f8fafc !important; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1 !important; }
        ::-webkit-scrollbar-thumb:hover { background: #7c3aed !important; }

        /* ── Light Mode: Noise Overlay Off ──────────────────────────── */
        .stApp::before {
            opacity: 0 !important;
        }

        /* ── Light Mode: Header Badge ───────────────────────────────── */
        .header-badge {
            background: rgba(124,58,237,0.06) !important;
            border: 1px solid rgba(124,58,237,0.15) !important;
            color: #7c3aed !important;
        }

        /* ── Light Mode: Status Badges ──────────────────────────────── */
        .status-dot.live { background: #10b981 !important; box-shadow: 0 0 8px rgba(16,185,129,0.3) !important; }
        .status-dot.warn { background: #d97706 !important; box-shadow: 0 0 8px rgba(217,119,6,0.3) !important; }
        .status-dot.err  { background: #e11d48 !important; box-shadow: 0 0 8px rgba(225,29,72,0.3) !important; }

        /* ── Light Mode: Footer ─────────────────────────────────────── */
        .dashboard-footer {
            color: #94a3b8 !important;
            border-top: 1px solid #cbd5e1 !important;
        }

        /* ── Light Mode: Gradient Text Helper ───────────────────────── */
        .gradient-text {
            background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 40%, #06b6d4 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        /* ── Light Mode: Metric Bar ─────────────────────────────────── */
        .metric-bar {
            background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 40%, #06b6d4 100%) !important;
        }
    </style>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header with a gradient divider line."""
    html = f"""
    <div style="margin-top:0.25rem;">
        <h2 style="margin:0;font-family:'Space Grotesk',sans-serif;font-size:1.35rem;
                   font-weight:700;color:var(--text-bright);">{title}</h2>
        {f'<p style="margin:0.15rem 0 0;color:var(--text-muted);font-size:0.82rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    <hr class="section-line">
    """
    st.markdown(html, unsafe_allow_html=True)



@st.fragment(run_every=1.0)
def render_header_time():
    """Render dynamic live ticking system time in IST timezone"""
    from datetime import timezone
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    ist_time_str = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown(f"""
    <div style="font-family: monospace; font-size: 0.78rem; color: #888; display: inline-flex; align-items: center; gap: 6px; margin-top: 2px;">
        <span style="color: #a78bfa; font-weight: 600;">🕒 Live Time:</span> {ist_time_str}
    </div>
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

def create_allocations_chart(allocations_df, layout=DARK_LAYOUT, theme_mode="Dark Mode"):
    """Donut chart — portfolio allocations"""
    if allocations_df.empty or 'Allocation (%)' not in allocations_df.columns:
        return None

    # Filter out zero allocations
    df_nonzero = allocations_df[allocations_df['Allocation (%)'] > 0.1]
    if df_nonzero.empty:
        return None

    is_light = theme_mode == "Light Mode"
    text_color = '#334155' if is_light else '#f0f1f5'
    line_color = '#ffffff' if is_light else '#0c0d14'

    fig = px.pie(
        df_nonzero,
        values='Allocation (%)',
        names='Market',
        title='Portfolio Allocation by Market',
        hover_data=['Size ($)', 'Edge'],
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(color=text_color, size=11, family='Inter'),
        marker=dict(line=dict(color=line_color, width=2.5)),
        hoverinfo='label+percent+value',
    )
    fig.update_layout(**layout, height=400, showlegend=True)
    return fig

def create_performance_chart(performance_history, layout=DARK_LAYOUT, theme_mode="Dark Mode"):
    """Create a line chart showing performance over time"""
    if not performance_history or len(performance_history) < 2:
        return None

    df = pd.DataFrame(performance_history)
    if 'timestamp' not in df.columns or 'total_pnl' not in df.columns:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    fig = go.Figure()
    
    is_light = theme_mode == "Light Mode"
    line_color = '#7c3aed' if is_light else '#a78bfa'
    marker_line_color = '#ffffff' if is_light else '#0c0d14'
    fill_color = 'rgba(124,58,237,0.06)' if is_light else 'rgba(139,92,246,0.06)'

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_pnl'],
        mode='lines+markers',
        name='Total P&L ($)',
        line=dict(color=line_color, width=3, shape='spline'),
        marker=dict(size=6, line=dict(width=1, color=marker_line_color)),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>',
    ))

    fig.update_layout(
        **layout,
        title='',
        xaxis_title='',
        yaxis_title='P&L ($)',
        hovermode='x unified',
        height=300
    )
    return fig

def create_risk_gauge(risk_metrics, layout=DARK_LAYOUT, theme_mode="Dark Mode"):
    """Dual gauge chart — Exposure + VaR"""
    fig = go.Figure()

    is_light = theme_mode == "Light Mode"
    title_color = '#64748b' if is_light else '#8b8fa8'
    num_cyan = '#0891b2' if is_light else '#22d3ee'
    num_purple = '#7c3aed' if is_light else '#a78bfa'
    tick_color = '#94a3b8' if is_light else '#5c6080'
    gauge_bg = '#f1f5f9' if is_light else '#191b27'
    gauge_border = '#cbd5e1' if is_light else '#2a2d42'
    step_base = '#e2e8f0' if is_light else '#12131c'
    step_purple = 'rgba(124,58,237,0.08)' if is_light else 'rgba(139,92,246,0.08)'
    step_danger = 'rgba(225,29,72,0.08)' if is_light else 'rgba(251,113,133,0.1)'
    step_cyan = 'rgba(6,182,212,0.08)' if is_light else 'rgba(34,211,238,0.08)'

    # Exposure gauge
    exposure = risk_metrics.get('exposure_ratio', 0) * 100
    fig.add_trace(go.Indicator(
        mode='gauge+number+delta',
        value=exposure,
        domain={'x': [0, 0.5], 'y': [0.2, 0.8]},
        title={'text': 'EXPOSURE %', 'font': {'color': title_color, 'size': 12, 'family': 'Space Grotesk'}},
        number={'font': {'color': num_purple, 'size': 26, 'family': 'JetBrains Mono'}},
        delta={'reference': 75, 'increasing': {'color': '#fb7185'}, 'decreasing': {'color': '#34d399'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': tick_color, 'dtick': 25},
            'bar': {'color': num_purple, 'thickness': 0.65},
            'bgcolor': gauge_bg,
            'bordercolor': gauge_border,
            'steps': [
                {'range': [0, 50], 'color': step_base},
                {'range': [50, 75], 'color': step_purple},
                {'range': [75, 100], 'color': step_danger},
            ],
            'threshold': {
                'line': {'color': '#fb7185', 'width': 3},
                'thickness': 0.8,
                'value': 90,
            },
        },
    ))

    # VaR gauge
    var_95 = risk_metrics.get('var_95', 0)
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=var_95,
        domain={'x': [0.5, 1], 'y': [0.2, 0.8]},
        title={'text': 'VAR 95% ($)', 'font': {'color': title_color, 'size': 12, 'family': 'Space Grotesk'}},
        number={'font': {'color': num_cyan, 'size': 26, 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [None, max(var_95 * 2, 100)], 'tickcolor': tick_color},
            'bar': {'color': num_cyan, 'thickness': 0.65},
            'bgcolor': gauge_bg,
            'bordercolor': gauge_border,
            'steps': [
                {'range': [0, var_95 * 0.5], 'color': step_base},
                {'range': [var_95 * 0.5, var_95], 'color': step_cyan},
            ],
            'threshold': {
                'line': {'color': '#fb7185', 'width': 3},
                'thickness': 0.8,
                'value': var_95 * 1.5,
            },
        },
    ))

    fig.update_layout(**layout, height=280)
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

def fetch_5min_crypto_events():
    """Fetch active 5-minute prediction contracts from Polymarket API (tag_id=102892)"""
    try:
        url = "https://gamma-api.polymarket.com/events?active=true&tag_id=102892&limit=100"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            events = response.json()
            return [e for e in events if e.get("active") and e.get("markets")]
    except Exception as e:
        logger.error(f"Failed to fetch 5min crypto events: {e}")
    return []


def render_crypto_markets_tab(bot):
    """Render the Crypto Markets tab showing 5-minute contracts for major coins."""
    import random
    
    section_header("Crypto markets", "Active 5-minute prediction contracts across major assets")
    
    # 1. Fetch live events
    live_events = fetch_5min_crypto_events()
    
    # 2. Group and find the latest event for each coin
    COINS = {
        "Bitcoin": {"emoji": "🪙", "color": "#e28743", "slug_pattern": "btc-up", "label": "BTC"},
        "Ethereum": {"emoji": "🔷", "color": "#2563eb", "slug_pattern": "eth-up", "label": "ETH"},
        "Solana": {"emoji": "☀️", "color": "#8b5cf6", "slug_pattern": "sol-up", "label": "SOL"},
        "XRP": {"emoji": "💧", "color": "#06b6d4", "slug_pattern": "xrp-up", "label": "XRP"}
    }
    
    coin_data = {}
    
    # Walk through the configured coins and match them to fetched events
    for coin_name, info in COINS.items():
        matched_event = None
        # Try to find an event that matches the slug pattern
        for event in live_events:
            slug = event.get("seriesSlug") or event.get("slug") or ""
            if info["slug_pattern"] in slug.lower():
                # Take the latest matching event by start date or end date
                if matched_event is None or event.get("startDate", "") > matched_event.get("startDate", ""):
                    matched_event = event
        
        # If we matched an event, extract market details
        if matched_event and matched_event.get("markets"):
            market = matched_event["markets"][0]
            prices_raw = market.get("outcomePrices")
            prices = [0.5, 0.5]
            if isinstance(prices_raw, str):
                try:
                    prices = json.loads(prices_raw)
                except:
                    pass
            elif isinstance(prices_raw, list):
                prices = prices_raw
            
            try:
                yes_price = float(prices[0]) * 100 if len(prices) > 0 else 50.0
                no_price = float(prices[1]) * 100 if len(prices) > 1 else 50.0
            except:
                yes_price, no_price = 50.0, 50.0
                
            coin_data[coin_name] = {
                "live": True,
                "question": market.get("question", f"{coin_name} Price Contract"),
                "yes_price": yes_price,
                "no_price": no_price,
                "accepting_orders": market.get("acceptingOrders", False),
                "token_id": market.get("clobTokenIds", ["", ""])[0] if isinstance(market.get("clobTokenIds"), list) else json.loads(market.get("clobTokenIds", "[\"\", \"\"]"))[0]
            }
        else:
            # Fallback to simulated pricing
            # To make it dynamic, we walk it in session state
            sim_state_key = f"sim_price_{coin_name}"
            if sim_state_key not in st.session_state:
                st.session_state[sim_state_key] = random.uniform(40.0, 60.0)
            
            st.session_state[sim_state_key] = max(1.0, min(99.0, st.session_state[sim_state_key] + random.uniform(-2.0, 2.0)))
            yes_p = st.session_state[sim_state_key]
            no_p = 100.0 - yes_p
            
            coin_data[coin_name] = {
                "live": False,
                "question": f"Will {coin_name} be UP or DOWN at the next 5-minute candle close?",
                "yes_price": yes_p,
                "no_price": no_p,
                "accepting_orders": True,
                "token_id": f"sim_token_{coin_name}"
            }
            
    # 3. Render 4 columns of tiles
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    # Focused coin selection state
    if "focused_coin" not in st.session_state:
        st.session_state["focused_coin"] = "Bitcoin"
        
    for idx, (coin_name, data) in enumerate(coin_data.items()):
        info = COINS[coin_name]
        is_accepting = data["accepting_orders"]
        
        status_text = "Accepting Orders" if is_accepting else "Locked/Settling"
        status_bg = "rgba(16, 185, 129, 0.12)" if is_accepting else "rgba(148, 163, 184, 0.12)"
        status_color = "#10b981" if is_accepting else "#94a3b8"
        
        # Highlight border if focused
        is_focused = st.session_state["focused_coin"] == coin_name
        border_style = f"2px solid {info['color']}" if is_focused else "1px solid var(--border)"
        box_shadow = f"0 0 12px {info['color']}33" if is_focused else "0 4px 6px -1px rgba(0,0,0,0.1)"
        
        html_card = f"""
        <div class="vibe-card" style="padding: 16px; border-radius: var(--radius-lg); border: {border_style}; 
                    display: flex; flex-direction: column; gap: 8px; 
                    box-shadow: {box_shadow}; transition: all 0.2s ease-in-out;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; color: {info['color']};">
                    {info['emoji']} {coin_name}
                </span>
                <span style="font-size: 0.65rem; font-weight: 700; padding: 2px 8px; border-radius: 12px; 
                             background: {status_bg}; color: {status_color}; text-transform: uppercase;">
                    {status_text}
                </span>
            </div>
            <div style="font-size: 0.76rem; color: var(--text-secondary); line-height: 1.35; min-height: 2.8rem; font-family: 'Inter', sans-serif;">
                {data['question']}
            </div>
            <div style="display: flex; gap: 8px; margin-top: 4px;">
                <div style="flex: 1; text-align: center; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 8px; padding: 6px 2px;">
                    <div style="font-size: 0.6rem; color: var(--accent-green); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">YES Price</div>
                    <div style="font-size: 1.25rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--accent-green);">{data['yes_price']:.1f}¢</div>
                </div>
                <div style="flex: 1; text-align: center; background: rgba(251, 113, 133, 0.08); border: 1px solid rgba(251, 113, 133, 0.15); border-radius: 8px; padding: 6px 2px;">
                    <div style="font-size: 0.6rem; color: var(--accent-rose); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">NO Price</div>
                    <div style="font-size: 1.25rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--accent-rose);">{data['no_price']:.1f}¢</div>
                </div>
            </div>
        </div>
        """
        
        with cols[idx]:
            st.markdown(html_card, unsafe_allow_html=True)
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button(f"🔍 Select {coin_name}", key=f"focus_{coin_name}", use_container_width=True):
                st.session_state["focused_coin"] = coin_name
                st.rerun()

    # 4. Detailed Chart View section
    st.markdown("---")
    focused_coin = st.session_state["focused_coin"]
    focused_data = coin_data[focused_coin]
    focused_info = COINS[focused_coin]
    
    # Store price history data in session state for charting
    history_key = f"price_history_{focused_coin}"
    if history_key not in st.session_state:
        # Prepopulate with 150 points
        prices_list = []
        current_price = focused_data["yes_price"]
        temp_price = current_price
        for _ in range(150):
            temp_price = max(1.0, min(99.0, temp_price + random.uniform(-1.5, 1.5)))
            prices_list.append(temp_price)
        prices_list.reverse()
        
        timestamps = [datetime.now() - timedelta(seconds=i * 2) for i in range(150)]
        timestamps.reverse()
        
        st.session_state[history_key] = pd.DataFrame({
            'Time': timestamps,
            'YES Price (¢)': prices_list
        })
    else:
        # Tick the price and append it
        df = st.session_state[history_key]
        now = datetime.now()
        current_price = focused_data["yes_price"]
        current_price_ticked = max(1.0, min(99.0, current_price + random.uniform(-0.5, 0.5)))
        
        new_row = pd.DataFrame({'Time': [now], 'YES Price (¢)': [current_price_ticked]})
        df = pd.concat([df, new_row]).iloc[-150:]
        st.session_state[history_key] = df
        
    df = st.session_state[history_key]
    price_change = df['YES Price (¢)'].iloc[-1] - df['YES Price (¢)'].iloc[0]
    
    c_chart, c_details = st.columns([2.2, 1])
    
    with c_chart:
        st.markdown(
            f"<h4 style='margin:0 0 0.5rem;font-family:Space Grotesk;font-size:0.95rem;"
            f"color:var(--text-bright);'>Contract: {focused_data['question']}</h4>",
            unsafe_allow_html=True
        )
        
        fig = go.Figure()
        line_color = '#10b981' if price_change >= 0 else '#ef4444'
        fill_color = 'rgba(16, 185, 129, 0.06)' if price_change >= 0 else 'rgba(239, 68, 68, 0.06)'
        
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df['YES Price (¢)'],
            mode='lines',
            line=dict(color=line_color, width=3, shape='spline'),
            fill='tozeroy',
            fillcolor=fill_color,
            hovertemplate='Price: %{y:.1f}¢<br>Time: %{x|%H:%M:%S}<extra></extra>'
        ))
        
        chart_layout = dict(plotly_layout)
        chart_layout.update(
            xaxis=dict(
                showgrid=False, zeroline=False,
                tickformat='%H:%M:%S',
                tickfont=dict(color='#8b8fa8', size=10)
            ),
            yaxis=dict(
                showgrid=True, gridcolor='#2a2d42' if theme_mode == "Dark Mode" else '#cbd5e1', 
                zeroline=False,
                range=[0, 100],
                tickfont=dict(color='#8b8fa8', size=10),
                ticksuffix='¢'
            ),
            margin=dict(l=40, r=20, t=10, b=30),
            height=320,
            showlegend=False
        )
        fig.update_layout(chart_layout)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
    with c_details:
        section_header("Contract Details", "")
        is_live_str = "🟢 Live Polymarket API" if focused_data["live"] else "ℹ️ Simulated Feed"
        
        st.markdown(f"""
        <div class="vibe-card" style="padding:14px; border-radius:10px; display:flex; flex-direction:column; gap:10px;">
            <div>
                <div style="font-size:0.65rem;color:var(--text-muted);text-transform:uppercase;font-weight:600;">Selected Asset</div>
                <div style="font-size:1.15rem;font-weight:700;color:{focused_info['color']};font-family:Space Grotesk;">
                    {focused_info['emoji']} {focused_coin} ({focused_info['label']})
                </div>
            </div>
            <div>
                <div style="font-size:0.65rem;color:var(--text-muted);text-transform:uppercase;font-weight:600;">Data Feed Source</div>
                <div style="font-size:0.82rem;font-weight:600;color:var(--text-bright);font-family:sans-serif;margin-top:2px;">
                    {is_live_str}
                </div>
            </div>
            <div>
                <div style="font-size:0.65rem;color:var(--text-muted);text-transform:uppercase;font-weight:600;">Conditional Token ID</div>
                <code style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:var(--accent-purple);
                             background:rgba(167,139,250,0.06);border:1px solid rgba(167,139,250,0.15);
                             border-radius:6px;padding:3px 6px;display:block;margin-top:4px;word-break:break-all;white-space:normal;line-height:1.2;">
                    {focused_data['token_id']}
                </code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Price contracts represent the binary outcome of the asset's 5-minute candle closing higher or lower than its opening level.")


def main():
    """Main dashboard application"""
    # Initialize bot
    with st.spinner("Initializing trading bot..."):
        bot, settings = initialize_bot()

    if bot is None:
        st.error("Failed to initialize trading bot. Please check your configuration.")
        return

    # Fetch connection status
    conn = bot.get_api_connection_details()

    # ── HERO HEADER ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("""
        <div>
            <h1 style="margin:0;font-size:2.2rem;padding-top:4px;">⚡ PolyTradingBot</h1>
            <p style="margin:0.25rem 0 0;color:#888;font-size:0.85rem;font-family:sans-serif;display:inline-block;">
                Polymarket Prediction Market Engine
            </p>
        </div>
        """, unsafe_allow_html=True)
        render_header_time()

    with col_right:
        status_col, menu_col = st.columns([4, 1])
        with status_col:
            if conn.get("connected"):
                proxy_short = f"{conn['proxy_address'][:6]}...{conn['proxy_address'][-4:]}" if conn.get('proxy_address') else "Unknown Proxy"
                badge_html = f'''
                <div style="display:flex; flex-direction:column; align-items:flex-end; gap:2px; padding-top:4px;">
                    <span style="display:inline-flex; align-items:center; gap:5px; background:rgba(16,185,129,0.1); border:1px solid #10b981; color:#10b981; border-radius:16px; padding:2px 10px; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                        <span class="status-dot live"></span>
                        {conn.get("status_text", "Connected")}
                    </span>
                    <span style="font-family:monospace; font-size:0.72rem; color:#888;">
                        Proxy: {proxy_short}
                    </span>
                    <span style="font-family:monospace; font-size:0.82rem; color:#00e676 !important; font-weight:700; margin-top:2px; display:inline-flex; align-items:center; gap:4px;">
                        💰 {conn.get('proxy_balance', 0.0):,.2f} USDC
                    </span>
                </div>
                '''
                st.markdown(badge_html, unsafe_allow_html=True)
            else:
                badge_html = f'''
                <div style="display:flex; flex-direction:column; align-items:flex-end; gap:2px; padding-top:4px;">
                    <span style="display:inline-flex; align-items:center; gap:5px; background:rgba(92,96,128,0.1); border:1px solid #ff4444; color:#ff4444; border-radius:16px; padding:2px 10px; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                        <span class="status-dot off"></span>
                        {conn.get("status_text", "Disconnected")}
                    </span>
                    <span style="font-family:sans-serif; font-size:0.72rem; color:#ff4444;">
                        API Offline
                    </span>
                    <span style="font-family:monospace; font-size:0.82rem; color:#00e676 !important; font-weight:700; margin-top:2px; display:inline-flex; align-items:center; gap:4px;">
                        💰 {conn.get('proxy_balance', 0.0):,.2f} USDC
                    </span>
                </div>
                '''
                st.markdown(badge_html, unsafe_allow_html=True)
        with menu_col:
            # Display 3-dots popover dropdown
            with st.popover("⋮", help="Wallet & Proxy Details"):
                eoa = conn.get("eoa_address") or "Not Available"
                proxy = conn.get("proxy_address") or "Not Available"
                
                # Shorten addresses
                eoa_short = f"{eoa[:6]}...{eoa[-4:]}" if len(eoa) > 10 else eoa
                proxy_short = f"{proxy[:6]}...{proxy[-4:]}" if len(proxy) > 10 else proxy
                
                popover_html = f"""
                <div style="font-family: Space Grotesk, sans-serif; display: flex; flex-direction: column; gap: 6px; padding-bottom: 6px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-bright); border-bottom: 1px solid var(--border); padding-bottom: 2px; margin-bottom: 2px;">
                        💼 Wallet Details
                    </div>
                    <div>
                        <div style="font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">EOA Wallet</div>
                        <code style="font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: var(--accent-purple); background: rgba(167, 139, 250, 0.06); border: 1px solid rgba(167, 139, 250, 0.15); border-radius: 6px; padding: 4px 8px; display: inline-block; margin-top: 2px;">{eoa_short}</code>
                    </div>
                    <div>
                        <div style="font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Proxy Wallet</div>
                        <code style="font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: var(--accent-cyan); background: rgba(34, 211, 238, 0.06); border: 1px solid rgba(34, 211, 238, 0.15); border-radius: 6px; padding: 4px 8px; display: inline-block; margin-top: 2px;">{proxy_short}</code>
                    </div>
                </div>
                """
                st.markdown(popover_html, unsafe_allow_html=True)
                if conn.get("error"):
                    st.warning(f"**Notice:** {conn['error']}")

    st.markdown("""<hr class="section-line">""", unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 30)

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data") or auto_refresh:
        st.rerun()

    # Theme selection
    st.sidebar.subheader("Dashboard Theme")
    selected_theme = st.sidebar.selectbox(
        "Theme Mode",
        ["Dark Mode", "Light Mode"],
        index=initial_theme_index,
        help="Switch between Dark and Light theme layout"
    )
    if selected_theme != theme_mode:
        update_config_toml(selected_theme)

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

    # Execution parameters
    st.sidebar.subheader("Execution Mode")
    use_limit_orders = st.sidebar.checkbox(
        "Use Limit Orders (Post-Only)",
        value=getattr(bot, 'use_limit_orders', False),
        help="Rest limit orders on CLOB spread instead of taking market prices"
    )
    quote_aggressiveness = st.sidebar.slider(
        "Quote Aggressiveness",
        min_value=0.0,
        max_value=1.0,
        value=float(getattr(bot, 'quote_aggressiveness', 0.3)),
        step=0.1,
        disabled=not use_limit_orders,
        help="0.0 = passive (at bid/ask), 0.5 = midpoint, 1.0 = active (1 tick inside spread)"
    )

    # Update bot settings based on sidebar selections
    bot.paper_mode = paper_mode
    bot.capital = float(capital)
    bot.constraints.max_total_exposure = max_exposure / 100.0
    bot.constraints.max_single_position = max_position / 100.0
    bot.constraints.max_drawdown = max_drawdown / 100.0
    bot.use_limit_orders = use_limit_orders
    bot.quote_aggressiveness = quote_aggressiveness
    if hasattr(bot, 'limit_quoter') and bot.limit_quoter is not None:
        bot.limit_quoter.aggressiveness = quote_aggressiveness

    # Set model based on selection
    if model_option == "Weighted Moving Average":
        bot.model = WeightedMovingAverageModel()
    elif model_option == "Volatility Adjusted":
        bot.model = VolatilityAdjustedModel()
    elif model_option == "Simple Edge (Baseline)":
        bot.model = SimpleEdgeModel()
    # Else keep EnsembleModel (default)

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊  Markets",
        "🪙  Crypto markets",
        "💼  Portfolio",
        "📈  Performance",
        "⚠️  Risk",
        "📋  Orders",
        "🚨  Alerts",
        "🧪  Backtest"
    ])

    # Fetch data
    with st.spinner("Fetching market data..."):
        markets_df = fetch_markets_data(bot)

    # Tab 1: Markets
    with tab1:
        section_header("Market Opportunities", "Live prediction market scan with edge detection")

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
                    fig_edge.update_layout(**plotly_layout)
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
                    fig_scatter.update_layout(**plotly_layout)
                    st.plotly_chart(fig_scatter, use_container_width=True)

    # Tab 2: Crypto markets
    with tab2:
        render_crypto_markets_tab(bot)

    # Tab 3: Portfolio
    with tab3:
        section_header("Portfolio Allocation", "Frank-Wolfe Kelly Criterion optimizer positions")

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
                        allocations_fig = create_allocations_chart(allocations_df, layout=plotly_layout, theme_mode=theme_mode)
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
                        allocations_fig = create_allocations_chart(allocations_df, layout=plotly_layout, theme_mode=theme_mode)
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

    # Tab 4: Performance
    with tab4:
        section_header("Performance Analytics", "Trading history, win rate, and P&L tracking")

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

            # Performance history chart
            perf_fig = create_performance_chart(bot.performance_log, layout=plotly_layout, theme_mode=theme_mode)
            if perf_fig:
                st.plotly_chart(perf_fig, use_container_width=True)
            else:
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

    # Tab 5: Risk
    with tab5:
        section_header("Risk Management", "Exposure, VaR, correlation analysis, and limit monitoring")

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
                risk_fig = create_risk_gauge(risk_summary, layout=plotly_layout, theme_mode=theme_mode)
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

    # Tab 6: Orders
    with tab6:
        section_header("Order Management", "Active quotes, open orders, and full trade history")

        # Sub-tabs for orders and trades
        order_tab1, order_tab2 = st.tabs(["📋 Active Orders", "📜 Trade History"])

        with order_tab1:
            st.subheader("Active Quotes")

            # Live limit-order state lives in limit_quoter.open_quotes
            open_quotes = None
            if hasattr(bot, 'limit_quoter') and hasattr(bot.limit_quoter, 'open_quotes'):
                open_quotes = bot.limit_quoter.open_quotes

            if open_quotes:
                quotes_data = []
                for token_id, q in open_quotes.items():
                    quotes_data.append({
                        'Token ID': f"{token_id[:12]}...",
                        'Order ID': q.get('order_id', 'N/A'),
                        'Side': q.get('side', '?').upper(),
                        'Price': f"{q.get('price', 0):.4f}",
                        'Size': f"${q.get('size', 0):,.2f}",
                        'Timestamp': q.get('timestamp', 'N/A'),
                    })
                quotes_df = pd.DataFrame(quotes_data)
                st.dataframe(quotes_df, use_container_width=True)
            else:
                st.info("No active limit quotes at the moment")

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

    # Tab 7: Alerts
    with tab7:
        section_header("Alerts & Monitoring", "Bot health metrics, API status, and alert log")

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

    # ═══════════════════════════════════════════════════════════════════
    # TAB 8 — BACKTEST
    # ═══════════════════════════════════════════════════════════════════
    with tab8:
        section_header("Interactive Backtesting", "Replay strategies on historical Polymarket price data")

        c1, c2 = st.columns(2)
        with c1:
            backtest_capital = st.number_input("Backtest Initial Capital ($)", min_value=10, max_value=1000000, value=1000, step=50)
            slippage_bps = st.slider("Backtest Slippage (BPS)", min_value=0, max_value=500, value=50, step=10)
        with c2:
            strategy_option = st.selectbox("Select Backtest Strategy", ["Ensemble Edge Strategy", "Buy Cheap Strategy"])
            max_pos_pct = st.slider("Max Single Position Limit (%)", min_value=5, max_value=100, value=20, step=5) / 100.0

        section_header("Strategy Parameters", "")
        if strategy_option == "Ensemble Edge Strategy":
            min_edge = st.slider("Minimum Edge (for BUY/SELL)", min_value=0.01, max_value=0.20, value=0.03, step=0.01)
            bet_fraction = st.slider("Bet Fraction (% of Capital per trade)", min_value=1, max_value=50, value=10, step=1) / 100.0

            def run_strat(markets, capital=1000.0):
                return ensemble_edge_strategy(markets, min_edge=min_edge, bet_fraction=bet_fraction, capital=capital)
        else:
            threshold = st.slider("Price Threshold (buy below)", min_value=0.02, max_value=0.50, value=0.15, step=0.01)
            bet_size = st.slider("Bet Size ($)", min_value=10, max_value=500, value=50, step=10)

            def run_strat(markets, capital=1000.0):
                return buy_cheap_strategy(markets, threshold=threshold, bet_size=bet_size)

        data_source = st.radio(
            "Choose price history source",
            ["Simulate from current live markets (Brownian bridge)", "Use saved CSV files in backtest/data"]
        )

        historical_df = pd.DataFrame()

        if data_source == "Simulate from current live markets (Brownian bridge)":
            sim_steps = st.slider("Simulated Days", min_value=5, max_value=90, value=30, step=5)
            if st.button("🎲 Generate Simulated History"):
                with st.spinner("Fetching live snapshot and generating history..."):
                    try:
                        manager = HistoricalDataManager()
                        snapshot = manager.fetch_current_snapshot()
                        if not snapshot.empty:
                            historical_df = manager.generate_simulated_history(snapshot, num_steps=sim_steps)
                            st.session_state['backtest_history'] = historical_df
                            st.success(f"Generated {len(historical_df)} historical records across {sim_steps} days.")
                        else:
                            st.error("Failed to fetch live markets snapshot.")
                    except Exception as e:
                        st.error(f"Error generating history: {e}")
        else:
            import os
            data_dir = "backtest/data"
            csv_files = (
                [f for f in os.listdir(data_dir) if f.startswith("snapshot_") and f.endswith(".csv")]
                if os.path.exists(data_dir)
                else []
            )
            if not csv_files:
                st.warning("No saved snapshots found in backtest/data/. Use the other option to simulate from live markets.")
            else:
                selected_csv = st.selectbox("Select snapshot file", csv_files)
                if st.button("📂 Load Selected Snapshot"):
                    try:
                        manager = HistoricalDataManager(data_dir=data_dir)
                        historical_df = manager.load_all_snapshots()
                        if not historical_df.empty:
                            st.session_state['backtest_history'] = historical_df
                            st.success(f"Loaded {len(historical_df)} records from {len(csv_files)} snapshot(s).")
                        else:
                            st.error("Failed to load snapshots.")
                    except Exception as e:
                        st.error(f"Error loading snapshots: {e}")

        # Allow using cached history from session_state
        if 'backtest_history' in st.session_state and historical_df.empty:
            historical_df = st.session_state['backtest_history']

        if not historical_df.empty:
            st.markdown("---")
            if st.button("▶️ Run Backtest", type="primary"):
                engine = BacktestEngine(
                    initial_capital=backtest_capital,
                    max_position_pct=max_pos_pct,
                    slippage_bps=slippage_bps,
                )
                with st.spinner("Running backtest..."):
                    try:
                        results = engine.run_backtest(historical_df, run_strat)
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
                        results = {}

                if results:
                    # Results summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Value", f"${results.get('final_value', 0):,.2f}")
                    with col2:
                        st.metric("Total Return", f"{results.get('total_return_pct', 0):.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.3f}")
                    with col4:
                        st.metric("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%")

                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("Total Trades", results.get('total_trades', 0))
                    with col6:
                        st.metric("Closed Trades", results.get('closed_trades', 0))
                    with col7:
                        st.metric("Win Rate", f"{results.get('win_rate_pct', 0):.1f}%")
                    with col8:
                        st.metric("Open Positions", results.get('open_positions', 0))

                    st.markdown("---")

                    # Equity curve
                    pv = results.get('portfolio_values', [])
                    if pv:
                        st.subheader("Equity Curve (last 30 points)")
                        st.line_chart(pv)

                    # Trade log
                    section_header("Trades Log", "")
                    trade_log = results.get('trade_log', [])
                    if trade_log:
                        trade_df = pd.DataFrame(trade_log)
                        st.dataframe(trade_df, use_container_width=True)
                    else:
                        st.info("No trades executed during this backtest.")
                else:
                    st.error("Backtest engine returned empty results.")

    # ── FOOTER ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="dashboard-footer">
        <p style="margin:0 0 0.25rem;">
            <span style="font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--text-bright);">PolyTradingBot</span>
            <span style="color:var(--text-muted);margin:0 0.5rem;">·</span>
            Polymarket Prediction Engine
            <span style="color:var(--text-muted);margin:0 0.5rem;">·</span>
            <a href="https://github.com/ThanveerShaik-git/PolyTradingBot" target="_blank">GitHub</a>
        </p>
        <p style="margin:0;color:var(--text-muted);font-size:0.72rem;">
            Data refreshed on demand · Built with Streamlit + Plotly
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
