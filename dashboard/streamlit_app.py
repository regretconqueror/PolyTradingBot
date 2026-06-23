"""
Streamlit Dashboard for PolyTradingBot
Provides real-time visualization of trading activities, market data, portfolio performance, and risk metrics.
"""
import streamlit as st
import streamlit.components.v1 as components
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

import subprocess
import os
import signal

def clean_html(html_str: str) -> str:
    """Minimize and clean HTML string to prevent Markdown code-block rendering."""
    import textwrap
    import re
    if not html_str:
        return ""
    dedented = textwrap.dedent(html_str).strip()
    return re.sub(r'\s+', ' ', dedented)

def get_performance_history(bot):
    """Retrieve or reconstruct performance history from bot state"""
    if hasattr(bot, 'performance_log') and len(bot.performance_log) >= 2:
        return bot.performance_log

    history = []
    if not hasattr(bot, 'trade_history') or not bot.trade_history:
        return history

    # Sort trades by timestamp
    trades = sorted(bot.trade_history, key=lambda x: x.get('timestamp', ''))
    
    # Calculate starting timestamp (e.g. 1 hour before first trade)
    if trades:
        try:
            first_time = datetime.fromisoformat(trades[0]['timestamp'])
            init_time = first_time - timedelta(hours=1)
            history.append({
                'timestamp': init_time.isoformat(),
                'realized_pnl': 0.0,
                'expected_pnl': 0.0,
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0
            })
        except Exception:
            pass

    running_pnl = 0.0
    total_trades = 0
    wins = 0
    for t in trades:
        # Generate expected P&L increment: size * edge
        edge = float(t.get('edge', 0.05))
        size = float(t.get('filled_value', t.get('size', 10.0)))
        pnl = size * edge
        running_pnl += pnl
        total_trades += 1
        if pnl >= 0:
            wins += 1
        
        history.append({
            'timestamp': t.get('timestamp'),
            'realized_pnl': 0.0,
            'expected_pnl': running_pnl,
            'total_pnl': running_pnl,
            'total_trades': total_trades,
            'win_rate': wins / total_trades
        })
    return history

def create_portfolio_value_chart(performance_history, timeframe, capital, layout=None, theme_mode="Dark Mode"):
    """Create a clean line chart showing portfolio net worth over time for the selected timeframe"""
    if layout is None:
        layout = DARK_LAYOUT if theme_mode == "Dark Mode" else LIGHT_LAYOUT
    if not performance_history:
        return None

    df = pd.DataFrame(performance_history)
    if 'timestamp' not in df.columns or 'total_pnl' not in df.columns:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    latest_time = df['timestamp'].max()
    if timeframe == "1D":
        threshold = latest_time - timedelta(days=1)
    elif timeframe == "1W":
        threshold = latest_time - timedelta(days=7)
    elif timeframe == "1M":
        threshold = latest_time - timedelta(days=30)
    elif timeframe == "1Y":
        threshold = latest_time - timedelta(days=365)
    elif timeframe == "YTD":
        threshold = datetime(latest_time.year, 1, 1)
    else: # ALL
        threshold = df['timestamp'].min()

    filtered_df = df[df['timestamp'] >= threshold].copy()

    # If filtered data is too sparse, keep original
    if len(filtered_df) < 2:
        filtered_df = df.copy()

    fig = go.Figure()
    
    is_light = theme_mode == "Light Mode"
    line_color = '#3b82f6'
    fill_color = 'rgba(59, 130, 246, 0.08)'

    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['total_pnl'] + capital, # Net Worth = capital + pnl
        mode='lines',
        name='Net Worth ($)',
        line=dict(color=line_color, width=2.5, shape='spline'),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='<b>%{x}</b><br>Net Worth: $%{y:,.2f}<extra></extra>',
    ))

    clean_layout = layout.copy()
    clean_layout.update(
        title="",
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
            fixedrange=True
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=180,
        showlegend=False
    )
    
    fig.update_layout(**clean_layout)
    return fig


def get_bot_pids() -> list:
    """Get the PIDs of running run.py processes."""
    pids = []
    try:
        if sys.platform.startswith("win"):
            # On Windows, query running processes using WMIC command line check
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            cmd = ['wmic', 'process', 'where', "name='python.exe' or name='pythonw.exe'", 'get', 'commandline,processid']
            try:
                output = subprocess.check_output(cmd, startupinfo=startupinfo, text=True, errors='ignore')
                current_pid = str(os.getpid())
                for line in output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if 'run.py' in line and current_pid not in line:
                        # Extract the PID, which is usually the last word in the wmic output
                        parts = line.split()
                        if parts:
                            try:
                                pid = int(parts[-1])
                                pids.append(pid)
                            except ValueError:
                                pass
            except Exception:
                # Fallback to powershell if wmic fails
                cmd = ['powershell', '-NoProfile', '-Command', 
                       'Get-CimInstance Win32_Process -Filter "name like \'python%\'" | Select-Object CommandLine, ProcessId | ConvertTo-Json']
                output = subprocess.check_output(cmd, startupinfo=startupinfo, text=True, errors='ignore')
                if 'run.py' in output:
                    try:
                        import json
                        data = json.loads(output)
                        if isinstance(data, dict):
                            data = [data]
                        for proc in data:
                            cmd_line = proc.get('CommandLine', '') or ''
                            pid = proc.get('ProcessId')
                            if 'run.py' in cmd_line and str(pid) != current_pid:
                                pids.append(int(pid))
                    except Exception:
                        import re
                        for match in re.finditer(r'"ProcessId":\s*(\d+)', output):
                            pids.append(int(match.group(1)))
        else:
            # On Unix systems (Linux / macOS), use ps aux
            output = subprocess.check_output(['ps', 'aux'], text=True, errors='ignore')
            current_pid = str(os.getpid())
            for line in output.splitlines():
                if 'run.py' in line and 'python' in line and current_pid not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pids.append(int(parts[1]))
                        except ValueError:
                            pass
    except Exception as e:
        logger.error(f"Error checking bot PIDs: {e}")
    
    return list(set(pids))

def check_bot_process_running() -> bool:
    """Check if the run.py process is running as a background task."""
    return len(get_bot_pids()) > 0

def start_bot_process(mode: str, interval: int) -> bool:
    """Start the run.py process as a detached background task."""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    run_py_path = project_root / 'run.py'
    
    # Find Python executable
    venv_python = project_root / '.venv' / 'Scripts' / 'python.exe' if sys.platform.startswith('win') else project_root / '.venv' / 'bin' / 'python'
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [python_exe, str(run_py_path), '--mode', mode.lower(), '--interval', str(interval)]
    if mode.lower() == 'live':
        cmd.append('--confirm-live')
        
    try:
        creationflags = 0
        if sys.platform.startswith('win'):
            # DETACHED_PROCESS = 0x00000008
            creationflags = 0x00000008
        
        subprocess.Popen(
            cmd,
            cwd=str(project_root),
            creationflags=creationflags,
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        logger.error(f"Failed to start bot process: {e}")
        return False

def stop_bot_process() -> bool:
    """Stop all running run.py processes."""
    pids = get_bot_pids()
    if not pids:
        return True
        
    success = True
    for pid in pids:
        try:
            if sys.platform.startswith('win'):
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}")
            success = False
            
    return success

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
# UNIFIED THEME CSS — one block per mode, no cascading conflicts
# ─────────────────────────────────────────────────────────────────----
if theme_mode == "Light Mode":
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
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
            --radius-lg: 14px;
            --radius-md: 10px;
        }

        /* ── Global ────────────────────────────────────────────────────── */
        .stApp { background: var(--bg-base) !important; color: var(--text-primary) !important;
                 font-family: 'Inter', sans-serif !important; }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.85) !important;
                                    backdrop-filter: blur(12px) !important;
                                    border-bottom: 1px solid var(--border) !important; }
        .positive { color: var(--accent-green) !important; }
        .negative { color: var(--accent-rose) !important; }
        .neutral  { color: var(--accent-amber) !important; }

        /* ── Sidebar ────────────────────────────────────────────────────── */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div { background: var(--bg-surface) !important;
                                                  border-right: 1px solid var(--border) !important; }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p { color: var(--text-primary) !important; }

        /* ── Typography ────────────────────────────────────────────────── */
        h1, h2, h3, h4, h5, h6 { color: var(--text-bright) !important;
                                  font-family: 'Space Grotesk', sans-serif !important; }
        h1 { background: var(--gradient-hero) !important;
             -webkit-background-clip: text !important;
             -webkit-text-fill-color: transparent !important; background-clip: text !important; }
        p, span, div, label, li { color: var(--text-primary) !important; }
        a { color: var(--accent-purple) !important; } a:hover { color: var(--accent-indigo) !important; }

        /* ── Metric Cards ─────────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1rem 1.1rem !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.9) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stMetric"]:hover {
            border-color: rgba(124,58,237,0.25) !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 0 12px rgba(124,58,237,0.06) !important;
            transform: translateY(-2px) !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important; font-weight: 600 !important;
            font-size: 0.7rem !important; text-transform: uppercase !important;
            letter-spacing: 0.12em !important; font-family: 'Space Grotesk', sans-serif !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: var(--text-bright) !important; font-weight: 700 !important;
            font-size: 1.6rem !important; line-height: 1.2 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important; font-size: 0.85rem !important;
        }

        /* ── Tab Bar ────────────────────────────────────────────────────── */
        [data-testid="stTabs"] {
            background: var(--bg-surface) !important; border-radius: var(--radius-lg) !important;
            padding: 5px !important; border: 1px solid var(--border) !important; gap: 2px !important;
        }
        [data-testid="stTabs"] button { color: var(--text-muted) !important;
                                         border-radius: 10px !important; transition: all 0.25s ease !important; }
        [data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(135deg, rgba(124,58,237,0.08) 0%, rgba(6,182,212,0.04) 100%) !important;
            color: var(--accent-violet) !important;
            border-bottom: 2px solid var(--accent-violet) !important; }
        [data-testid="stTabs"] button[aria-selected="true"] p { color: var(--accent-violet) !important; }
        [data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
            background: rgba(0,0,0,0.02) !important; color: var(--text-secondary) !important; }

        /* ── Selectbox / Dropdown ────────────────────────────────────────── */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: var(--bg-card) !important; color: var(--text-primary) !important;
            border: 1px solid var(--border) !important; border-radius: 8px !important; }
        [data-testid="stSelectbox"] svg { fill: var(--text-muted) !important; }
        [data-testid="stSelectbox"] div[data-baseweb="select"] input {
            color: var(--text-primary) !important; }
        /* Dropdown popover (the menu that opens) */
        div[data-baseweb="popover"] div[data-baseweb="menu"] {
            background-color: #ffffff !important; border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important; }
        div[data-baseweb="popover"] div[data-baseweb="menu"] li,
        div[data-baseweb="popover"] div[data-baseweb="menu"] div[role="option"],
        div[data-baseweb="popover"] div[data-baseweb="menu"] li > div {
            background-color: #ffffff !important; color: #334155 !important; }
        div[data-baseweb="popover"] div[data-baseweb="menu"] li:hover,
        div[data-baseweb="popover"] div[data-baseweb="menu"] li:hover > div,
        div[data-baseweb="popover"] div[data-baseweb="menu"] div[role="option"]:hover {
            background-color: #f1f5f9 !important; color: #7c3aed !important; }
        div[data-baseweb="popover"] div[data-baseweb="menu"] li[aria-selected="true"],
        div[data-baseweb="popover"] div[data-baseweb="menu"] li[aria-selected="true"] > div {
            background-color: rgba(124,58,237,0.08) !important; color: #7c3aed !important; }

        /* ── Multi-select dropdown ──────────────────────────────────────── */
        [data-testid="stMultiSelect"] > div > div {
            background: var(--bg-card) !important; color: var(--text-primary) !important;
            border: 1px solid var(--border) !important; border-radius: 8px !important; }
        [data-testid="stMultiSelect"] div[data-baseweb="tag"] {
            background: rgba(124,58,237,0.08) !important; border: 1px solid rgba(124,58,237,0.2) !important;
            border-radius: 6px !important; }
        [data-testid="stMultiSelect"] div[data-baseweb="tag"] span { color: #7c3aed !important; }

        /* ── Inputs / Number / Text ─────────────────────────────────────── */
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background: var(--bg-card) !important; color: var(--text-primary) !important;
            border: 1px solid var(--border) !important; border-radius: 8px !important; }
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {
            border-color: var(--accent-purple) !important;
            box-shadow: 0 0 0 3px rgba(124,58,237,0.08) !important; }

        /* ── Slider ──────────────────────────────────────────────────────── */
        [data-testid="stSlider"] > div > div > div > div { background: var(--border) !important; }
        [data-testid="stSlider"] > div > div > div > div > div { background: var(--accent-purple) !important; }

        /* ── DataFrames / Tables ──────────────────────────────────────────── */
        [data-testid="stDataFrame"], .stDataFrame {
            border-radius: var(--radius-md) !important; overflow: hidden !important;
            border: 1px solid var(--border) !important; }
        [data-testid="stDataFrame"] td { color: var(--text-primary) !important;
                                          background: #ffffff !important;
                                          border-color: var(--border) !important;
                                          font-family: 'JetBrains Mono', monospace !important;
                                          font-size: 0.82rem !important; }
        [data-testid="stDataFrame"] th { color: var(--accent-violet) !important;
                                          background: var(--bg-elevated) !important;
                                          border-color: var(--border) !important;
                                          font-weight: 600 !important; text-transform: uppercase !important;
                                          font-size: 0.72rem !important; letter-spacing: 0.08em !important;
                                          font-family: 'Space Grotesk', sans-serif !important; }
        [data-testid="stDataFrame"] tr:nth-child(even) td { background: #f8fafc !important; }
        [data-testid="stDataFrame"] tr:hover td { background: var(--bg-hover) !important; }

        /* ── Glow Data Grid (custom tables) ──────────────────────────────── */
        .gdfg-virtualized-scroll { background: #ffffff !important; }
        .gdfg-cell { color: #334155 !important; background: #ffffff !important;
                     border-color: #e2e8f0 !important; }
        .gdfg-header-cell { color: #7c3aed !important; background: #f1f5f9 !important;
                             border-color: #cbd5e1 !important; }

        /* ── Expander ────────────────────────────────────────────────────── */
        [data-testid="stExpander"] { background: var(--bg-card) !important;
                                      border: 1px solid var(--border) !important;
                                      border-radius: var(--radius-md) !important; }
        [data-testid="stExpander"] summary span { color: var(--accent-violet) !important;
                                                    font-weight: 600 !important; }

        /* ── Alert Boxes ────────────────────────────────────────────────── */
        [data-testid="stAlert"] { background: var(--bg-card) !important;
                                   border-radius: var(--radius-md) !important;
                                   border: 1px solid var(--border) !important; }

        /* ── Buttons ────────────────────────────────────────────────────── */
        .stButton > button {
            background: var(--gradient-hero) !important; color: #fff !important;
            font-weight: 600 !important; border: none !important;
            border-radius: 8px !important; padding: 0.5rem 1.4rem !important;
            font-family: 'Space Grotesk', sans-serif !important;
            text-transform: uppercase !important; letter-spacing: 0.05em !important; }
        .stButton > button:hover {
            box-shadow: 0 4px 20px rgba(124,58,237,0.25) !important;
            transform: translateY(-1px) !important; }
        .stButton > button p { color: #fff !important; }

        /* ── Checkbox / Toggle ──────────────────────────────────────────── */
        [data-testid="stCheckbox"] label span { color: var(--text-primary) !important; }
        [data-testid="stToggle"] label span { color: var(--text-primary) !important; }

        /* ── Dividers ──────────────────────────────────────────────────── */
        hr { background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
             border: none !important; height: 1px !important; }
        .section-line {
            height: 2px; border: none; margin: 0.5rem 0 1.25rem; border-radius: 1px;
            background: linear-gradient(90deg, #7c3aed, #06b6d4, transparent) !important; }

        /* ── Status Dots ────────────────────────────────────────────────── */
        .status-dot { display: inline-block; width: 6px; height: 6px;
                      border-radius: 50%; margin-right: 6px;
                      animation: pulse-dot 1.8s infinite ease-in-out; }
        .status-dot.live { background: #10b981 !important; box-shadow: 0 0 8px rgba(16,185,129,0.3) !important; }
        .status-dot.off  { background: #94a3b8 !important; animation: none !important; }
        .status-dot.err  { background: #e11d48 !important; box-shadow: 0 0 8px rgba(225,29,72,0.3) !important; }
        @keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); }
                                50% { opacity: 0.6; transform: scale(1.3); } }

        /* ── Vibe Card ──────────────────────────────────────────────────── */
        .vibe-card { background: #ffffff !important; border: 1px solid #cbd5e1 !important;
                     border-radius: var(--radius-lg) !important;
                     box-shadow: 0 1px 6px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8) !important;
                     transition: all 0.3s ease !important; }
        .vibe-card:hover { border-color: rgba(124,58,237,0.25) !important;
                            box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 0 12px rgba(124,58,237,0.04) !important;
                            transform: translateY(-2px) !important; }

        /* ── Live Indicator ──────────────────────────────────────────────── */
        .live-indicator { background: rgba(225,29,72,0.06) !important;
                          border: 1px solid rgba(225,29,72,0.2) !important;
                          color: #e11d48 !important; border-radius: 8px; padding: 6px 12px;
                          font-size: 0.78rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }

        /* ── Gradient Text / Badge / Footer ────────────────────────────── */
        .gradient-text { background: var(--gradient-hero) !important;
                         -webkit-background-clip: text !important;
                         -webkit-text-fill-color: transparent !important; background-clip: text !important; }
        .header-badge { background: rgba(124,58,237,0.06) !important;
                         border: 1px solid rgba(124,58,237,0.15) !important;
                         border-radius: 20px; padding: 4px 12px; font-size: 0.75rem;
                         color: #7c3aed !important; font-family: 'Space Grotesk', sans-serif; }
        .dashboard-footer { text-align: center; padding: 2rem 0 1rem;
                            color: var(--text-muted) !important; font-size: 0.8rem;
                            font-family: 'Space Grotesk', sans-serif;
                            border-top: 1px solid var(--border) !important; margin-top: 3rem; }
        .dashboard-footer a { color: var(--accent-purple) !important; }

        /* ── Scrollbar ─────────────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: #f8fafc !important; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1 !important; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-violet) !important; }

        /* ── Noise overlay OFF in light mode ────────────────────────────── */
        .stApp::before { display: none !important; }

        /* ── Negative edge highlighting ──────────────────────────────────── */
        .negative-edge { background-color: rgba(225,29,72,0.06) !important; color: #e11d48 !important; }
        .funds-amount { color: #00e676 !important; }

        /* Hide dynamic scripting helper iframe */
        iframe[width="0"][height="0"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg-void:       #08090d;
            --bg-base:       #0c0d14;
            --bg-surface:    #12131c;
            --bg-elevated:   #191b27;
            --bg-card:       #1e2030;
            --bg-hover:      #252840;
            --border:        #2a2d42;
            --border-light:  #363a54;
            --border-glow:   rgba(139, 92, 246, 0.15);
            --text-bright:   #f0f1f5;
            --text-primary:  #d4d6e0;
            --text-secondary:#8b8fa8;
            --text-muted:    #5c6080;
            --accent-purple: #a78bfa;
            --accent-violet: #8b5cf6;
            --accent-indigo: #6366f1;
            --accent-cyan:   #22d3ee;
            --accent-teal:   #2dd4bf;
            --accent-green:  #34d399;
            --accent-emerald:#10b981;
            --accent-amber:  #fbbf24;
            --accent-orange: #f97316;
            --accent-rose:   #fb7185;
            --accent-red:    #ef4444;
            --accent-pink:   #ec4899;
            --gradient-hero:  linear-gradient(135deg, #8b5cf6 0%, #6366f1 40%, #22d3ee 100%);
            --gradient-card:  linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(34,211,238,0.04) 100%);
            --glow-purple:   0 0 20px rgba(139,92,246,0.15), 0 0 60px rgba(139,92,246,0.05);
            --radius-lg: 16px;
            --radius-md: 12px;
        }

        /* ── Global ────────────────────────────────────────────────────── */
        .stApp { background: var(--bg-void) !important; color: var(--text-primary) !important;
                 font-family: 'Inter', sans-serif !important; }
        [data-testid="stHeader"] { background: transparent !important;
                                    border-bottom: 1px solid var(--border) !important;
                                    backdrop-filter: blur(12px) !important; }
        .positive { color: var(--accent-green) !important; }
        .negative { color: var(--accent-rose) !important; }
        .neutral  { color: var(--accent-amber) !important; }

        /* ── Sidebar ────────────────────────────────────────────────────── */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div { background: var(--bg-base) !important;
                                                  border-right: 1px solid var(--border) !important; }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p { color: var(--text-primary) !important; }

        /* ── Typography ────────────────────────────────────────────────── */
        h1, h2, h3, h4, h5, h6 { color: var(--text-bright) !important;
                                  font-family: 'Space Grotesk', sans-serif !important;
                                  letter-spacing: -0.02em !important; }
        h1 { background: var(--gradient-hero) !important;
             -webkit-background-clip: text !important;
             -webkit-text-fill-color: transparent !important;
             background-clip: text !important; font-weight: 800 !important; font-size: 2.2rem !important; }
        p, span, div, label, li { color: var(--text-primary) !important; }
        a { color: var(--accent-cyan) !important; } a:hover { color: var(--accent-purple) !important; }

        /* ── Metric Cards — Glassmorphism ─────────────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--gradient-card),
                        linear-gradient(180deg, rgba(30,32,48,0.95) 0%, rgba(18,19,28,0.98) 100%) !important;
            backdrop-filter: blur(24px) saturate(1.2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.4rem 1.6rem !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04) !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        [data-testid="stMetric"]:hover {
            border-color: rgba(139,92,246,0.35) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4), var(--glow-purple) !important;
            transform: translateY(-3px) !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important; font-weight: 600 !important;
            font-size: 0.7rem !important; text-transform: uppercase !important;
            letter-spacing: 0.12em !important; font-family: 'Space Grotesk', sans-serif !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: var(--text-bright) !important; font-weight: 700 !important;
            font-size: 2rem !important; line-height: 1.2 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important; font-size: 0.85rem !important;
        }

        /* ── Tab Bar ────────────────────────────────────────────────────── */
        [data-testid="stTabs"] {
            background: var(--bg-surface) !important; border-radius: var(--radius-lg) !important;
            padding: 5px !important; border: 1px solid var(--border) !important; gap: 2px !important;
        }
        [data-testid="stTabs"] button { color: var(--text-muted) !important;
                                         border-radius: 10px !important;
                                         transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                                         font-family: 'Space Grotesk', sans-serif !important; }
        [data-testid="stTabs"] button p { color: inherit !important; font-size: 0.85rem !important; }
        [data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(34,211,238,0.08) 100%) !important;
            color: var(--accent-purple) !important;
            border-bottom: 2px solid var(--accent-purple) !important;
            box-shadow: 0 0 20px rgba(139,92,246,0.08) !important; }
        [data-testid="stTabs"] button[aria-selected="true"] p { color: var(--accent-purple) !important; }
        [data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
            background: rgba(255,255,255,0.03) !important; color: var(--text-secondary) !important; }

        /* ── Selectbox / Dropdown ────────────────────────────────────────── */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: var(--bg-card) !important; color: var(--text-primary) !important;
            border: 1px solid var(--border) !important; border-radius: 8px !important; }
        [data-testid="stSelectbox"] svg { fill: var(--text-muted) !important; }
        [data-testid="stSelectbox"] div[data-baseweb="select"] input {
            color: var(--text-primary) !important; }
        /* Dropdown popover */
        div[data-baseweb="popover"] div[data-baseweb="menu"] {
            background-color: #1e2030 !important; border: 1px solid #2a2d42 !important;
            border-radius: 10px !important; box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important; }
        div[data-baseweb="popover"] div[data-baseweb="menu"] li,
        div[data-baseweb="popover"] div[data-baseweb="menu"] div[role="option"],
        div[data-baseweb="popover"] div[data-baseweb="menu"] li > div {
            background-color: #1e2030 !important; color: #d4d6e0 !important; }
        div[data-baseweb="popover"] div[data-baseweb="menu"] li:hover,
        div[data-baseweb="popover"] div[data-baseweb="menu"] li:hover > div,
        div[data-baseweb="popover"] div[data-baseweb="menu"] div[role="option"]:hover {
            background-color: #252840 !important; color: #a78bfa !important; }
        div[data-baseweb="popover"] div[data-baseweb="menu"] li[aria-selected="true"],
        div[data-baseweb="popover"] div[data-baseweb="menu"] li[aria-selected="true"] > div {
            background-color: rgba(139,92,246,0.12) !important; color: #a78bfa !important; }

        /* ── Multi-select dropdown ──────────────────────────────────────── */
        [data-testid="stMultiSelect"] > div > div {
            background: var(--bg-card) !important; color: var(--text-primary) !important;
            border: 1px solid var(--border) !important; border-radius: 8px !important; }
        [data-testid="stMultiSelect"] div[data-baseweb="tag"] {
            background: rgba(139,92,246,0.1) !important; border: 1px solid rgba(139,92,246,0.2) !important;
            border-radius: 6px !important; }
        [data-testid="stMultiSelect"] div[data-baseweb="tag"] span { color: #a78bfa !important; }

        /* ── Inputs / Number / Text ─────────────────────────────────────── */
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background: var(--bg-card) !important; color: var(--text-primary) !important;
            border: 1px solid var(--border) !important; border-radius: 8px !important; }
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {
            border-color: var(--accent-purple) !important;
            box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important; }

        /* ── Slider ──────────────────────────────────────────────────────── */
        [data-testid="stSlider"] > div > div > div > div { background: var(--border) !important; }
        [data-testid="stSlider"] > div > div > div > div > div { background: var(--accent-purple) !important; }

        /* ── DataFrames / Tables ──────────────────────────────────────────── */
        [data-testid="stDataFrame"], .stDataFrame {
            border-radius: var(--radius-md) !important; overflow: hidden !important;
            border: 1px solid var(--border) !important; }
        [data-testid="stDataFrame"] td { color: var(--text-primary) !important;
                                          background: var(--bg-surface) !important;
                                          border-color: var(--border) !important;
                                          font-family: 'JetBrains Mono', monospace !important;
                                          font-size: 0.82rem !important; }
        [data-testid="stDataFrame"] th { color: var(--accent-purple) !important;
                                          background: var(--bg-elevated) !important;
                                          border-color: var(--border) !important;
                                          font-weight: 600 !important; text-transform: uppercase !important;
                                          font-size: 0.72rem !important; letter-spacing: 0.08em !important;
                                          font-family: 'Space Grotesk', sans-serif !important; }
        [data-testid="stDataFrame"] tr:nth-child(even) td { background: rgba(25,27,39,0.6) !important; }
        [data-testid="stDataFrame"] tr:hover td { background: var(--bg-hover) !important; }

        /* ── Glow Data Grid ──────────────────────────────────────────────── */
        .gdfg-virtualized-scroll { background: #12131c !important; }
        .gdfg-cell { color: #d4d6e0 !important; background: #1e2030 !important;
                     border-color: #2a2d42 !important; }
        .gdfg-header-cell { color: #a78bfa !important; background: #191b27 !important;
                             border-color: #2a2d42 !important; }

        /* ── Expander ────────────────────────────────────────────────────── */
        [data-testid="stExpander"] { background: var(--bg-card) !important;
                                      border: 1px solid var(--border) !important;
                                      border-radius: var(--radius-md) !important;
                                      transition: all 0.3s ease !important; }
        [data-testid="stExpander"]:hover { border-color: var(--border-light) !important; }
        [data-testid="stExpander"] summary span { color: var(--accent-cyan) !important;
                                                    font-weight: 600 !important; }

        /* ── Alert Boxes ────────────────────────────────────────────────── */
        [data-testid="stAlert"] { background: var(--bg-card) !important;
                                   border-radius: var(--radius-md) !important;
                                   border: 1px solid var(--border) !important; }

        /* ── Buttons ────────────────────────────────────────────────────── */
        .stButton > button {
            background: var(--gradient-hero) !important; color: #fff !important;
            font-weight: 600 !important; border: none !important;
            border-radius: var(--radius-md) !important; padding: 0.6rem 1.6rem !important;
            font-family: 'Space Grotesk', sans-serif !important;
            text-transform: uppercase !important; letter-spacing: 0.06em !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; }
        .stButton > button:hover {
            box-shadow: 0 0 30px rgba(139,92,246,0.35), 0 0 60px rgba(99,102,241,0.15) !important;
            transform: translateY(-2px) !important; }
        .stButton > button p { color: #fff !important; }

        /* ── Checkbox / Toggle ──────────────────────────────────────────── */
        [data-testid="stCheckbox"] label span { color: var(--text-primary) !important; }
        [data-testid="stToggle"] label span { color: var(--text-primary) !important; }

        /* ── Dividers ──────────────────────────────────────────────────── */
        hr { background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
             border: none !important; height: 1px !important; }
        .section-line {
            height: 2px; border: none; margin: 0.5rem 0 1.25rem; border-radius: 1px;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan), transparent) !important; }

        /* ── Status Dots ────────────────────────────────────────────────── */
        .status-dot { display: inline-block; width: 6px; height: 6px;
                      border-radius: 50%; margin-right: 6px;
                      animation: pulse-dot 1.8s infinite ease-in-out; }
        .status-dot.live { background: var(--accent-green) !important;
                           box-shadow: 0 0 8px rgba(16,185,129,0.5) !important; }
        .status-dot.off  { background: var(--text-muted) !important; animation: none !important; }
        .status-dot.err  { background: var(--accent-rose) !important;
                           box-shadow: 0 0 8px rgba(251,113,133,0.4) !important; }
        @keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); }
                                50% { opacity: 0.6; transform: scale(1.3); } }

        /* ── Vibe Card ──────────────────────────────────────────────────── */
        .vibe-card { background: linear-gradient(135deg, rgba(30,32,48,0.9) 0%, rgba(18,19,28,0.95) 100%) !important;
                     border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important;
                     backdrop-filter: blur(24px) saturate(1.2) !important;
                     box-shadow: 0 2px 12px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04) !important;
                     transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important; }
        .vibe-card:hover { border-color: rgba(139,92,246,0.3) !important;
                            box-shadow: 0 8px 32px rgba(0,0,0,0.35), var(--glow-purple) !important;
                            transform: translateY(-2px) !important; }

        /* ── Live Indicator ──────────────────────────────────────────────── */
        .live-indicator { background: rgba(251,113,133,0.08) !important;
                          border: 1px solid rgba(251,113,133,0.3) !important;
                          color: var(--accent-rose) !important; border-radius: 8px; padding: 6px 12px;
                          font-size: 0.78rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }

        /* ── Gradient Text / Badge / Footer ────────────────────────────── */
        .gradient-text { background: var(--gradient-hero) !important;
                         -webkit-background-clip: text !important;
                         -webkit-text-fill-color: transparent !important; background-clip: text !important; }
        .header-badge { background: rgba(139,92,246,0.1) !important;
                         border: 1px solid rgba(139,92,246,0.2) !important;
                         border-radius: 20px; padding: 4px 12px; font-size: 0.75rem;
                         color: var(--accent-purple) !important; font-family: 'Space Grotesk', sans-serif; }
        .dashboard-footer { text-align: center; padding: 2rem 0 1rem;
                            color: var(--text-muted) !important; font-size: 0.8rem;
                            font-family: 'Space Grotesk', sans-serif;
                            border-top: 1px solid var(--border) !important; margin-top: 3rem; }
        .dashboard-footer a { color: var(--accent-purple) !important; }

        /* ── Scrollbar ─────────────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-void) !important; }
        ::-webkit-scrollbar-thumb { background: var(--border) !important; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-purple) !important; }

        /* ── Noise Overlay ──────────────────────────────────────────────── */
        .stApp::before {
            content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none; z-index: 0; opacity: 0.015;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
            background-size: 128px 128px; }

        /* ── Negative edge highlighting ──────────────────────────────────── */
        .negative-edge { background-color: rgba(239,68,68,0.08) !important; color: #fb7185 !important; }

        /* ── Funds amount — always green ──────────────────────────────────── */
        .funds-amount, .funds-amount * { color: #00e676 !important; }

        /* ── Plotly chart backgrounds ───────────────────────────────────── */
        .js-plotly-plot .plotly .main-svg { background: transparent !important; }

        /* ── Spinner ────────────────────────────────────────────────────── */
        [data-testid="stSpinner"] { color: var(--accent-purple) !important; }

        /* ── Pre / JSON ─────────────────────────────────────────────────── */
        pre { background: var(--bg-card) !important; color: var(--accent-teal) !important;
              border: 1px solid var(--border) !important; border-radius: 8px !important;
              font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }

        /* Hide dynamic scripting helper iframe */
        iframe[width="0"][height="0"] {
            display: none !important;
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
    """Render dynamic live ticking system time in IST and UTC timezones"""
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    ist_time_str = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    utc_time_str = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""
    <div style="font-family: monospace; font-size: 0.78rem; color: #888; display: flex; flex-direction: column; gap: 2px; margin-top: 2px;">
        <div><span style="color: #a78bfa; font-weight: 600;">🕒 IST Time:</span> {ist_time_str}</div>
        <div><span style="color: #6366f1; font-weight: 600;">🌐 UTC Time:</span> {utc_time_str}</div>
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
            min_bet_size=settings.min_bet_size,
            max_category_exposure=settings.category_limits
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
        bot.load_state()
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

def highlight_edge(val):
    """Highlight edge column: green for positive edge."""
    try:
        if isinstance(val, str):
            if '%' in val:
                numeric_val = float(val.replace('%', '')) / 100.0
            else:
                numeric_val = float(val)
        else:
            numeric_val = float(val)
        
        if numeric_val > 0.001:
            return 'background-color: #d4edda; color: #155724;'
    except:
        pass
    return ''

def display_market_table(markets_df):
    """Display the markets data in an interactive table"""
    if markets_df.empty:
        st.warning("No market data available")
        return

    # Select columns to display
    display_columns = ['Market', 'Category', 'Current Price', 'Your Probability', 'Edge', 'Liquidity', 'Volume 24h']
    display_df = markets_df[display_columns].copy()

    styled_df = display_df.style.map(highlight_edge, subset=['Edge'])
    st.dataframe(
        styled_df,
        width="stretch",
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

    styled_df = display_df.style.format({
        'Size ($)': '${:,.2f}',
        'Allocation (%)': '{:.1f}%',
        'Market Price': '{:.1%}',
        'Your Probability': '{:.1%}',
        'Edge': '{:.1%}'
    }).map(highlight_edge, subset=['Edge'])

    st.dataframe(
        styled_df,
        width="stretch"
    )

def fetch_5min_crypto_events():
    """Fetch active 5-minute prediction contracts from Polymarket API (tag_id=102892)"""
    try:
        url = "https://gamma-api.polymarket.com/events?active=true&tag_id=102892&limit=100&order=startDate&ascending=false"
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
                
            # Format time evaluation range in UTC only
            end_date_str = market.get("endDate")
            try:
                from datetime import timezone
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                start_dt = end_dt - timedelta(minutes=5)
                start_str = start_dt.strftime("%I:%M%p").lstrip('0')
                end_str = end_dt.strftime("%I:%M%p").lstrip('0')
                date_str = start_dt.strftime("%B %d")
                utc_question = f"{coin_name} Up or Down - {date_str}, {start_str}-{end_str} UTC"
            except Exception:
                utc_question = market.get("question", f"{coin_name} Price Contract")

            coin_data[coin_name] = {
                "live": True,
                "question": utc_question,
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
            
            # Format timeframe window for simulated data in UTC
            try:
                from datetime import timezone
                now_utc = datetime.now(timezone.utc)
                minute = (now_utc.minute // 5 + 1) * 5
                end_dt = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
                start_dt = end_dt - timedelta(minutes=5)
                start_str = start_dt.strftime("%I:%M%p").lstrip('0')
                end_str = end_dt.strftime("%I:%M%p").lstrip('0')
                date_str = start_dt.strftime("%B %d")
                sim_question = f"{coin_name} Up or Down - {date_str}, {start_str}-{end_str} UTC"
            except Exception:
                sim_question = f"Will {coin_name} be UP or DOWN at the next 5-minute candle close?"

            coin_data[coin_name] = {
                "live": False,
                "question": sim_question,
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
            if st.button(f"🔍 Select {coin_name}", key=f"focus_{coin_name}", width="stretch"):
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
        st.plotly_chart(fig, width="stretch", theme=None)
        
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
    bot_running = check_bot_process_running()

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
        interval_mins = getattr(bot, 'interval', 60)
        
        # Render the badges row using components.html to allow dynamic ticking without sandbox restrictions
        api_connected = conn.get("connected", False)
        api_status = conn.get("status_text", "CONNECTED" if api_connected else "DISCONNECTED")
        
        if api_connected:
            api_badge_html = f'''
            <span class="badge" style="background:rgba(16,185,129,0.1); border:1px solid #10b981; color:#10b981;">
                <span class="status-dot" style="background:#10b981; box-shadow:0 0 8px #10b981;"></span>
                API: {api_status}
            </span>
            '''
        else:
            api_badge_html = f'''
            <span class="badge" style="background:rgba(255,68,68,0.15); border:1px solid #ff4444; color:#ff4444;">
                <span class="status-dot" style="background:#ff4444; box-shadow:0 0 8px #ff4444;"></span>
                API: {api_status}
            </span>
            '''
            
        if bot_running:
            bot_badge_html = '''
            <span class="badge" style="background:rgba(16,185,129,0.1); border:1px solid #10b981; color:#10b981;">
                <span class="status-dot" style="background:#10b981; box-shadow:0 0 8px #10b981;"></span>
                Bot: Active
            </span>
            '''
        else:
            bot_badge_html = '''
            <span class="badge" style="background:rgba(148,163,184,0.1); border:1px solid #94a3b8; color:#94a3b8;">
                <span class="status-dot" style="background:#94a3b8;"></span>
                Bot: Inactive
            </span>
            '''
            
        # Build Mode badge HTML
        is_paper = getattr(bot, 'paper_mode', True)
        if is_paper:
            mode_badge_html = '''
            <span class="badge" style="background:rgba(167,139,250,0.15); border:1px solid #a78bfa; color:#a78bfa;">
                <span class="status-dot" style="background:#a78bfa; box-shadow:0 0 8px #a78bfa;"></span>
                Mode: Paper
            </span>
            '''
        else:
            mode_badge_html = '''
            <span class="badge" style="background:rgba(245,158,11,0.15); border:1px solid #f59e0b; color:#f59e0b;">
                <span class="status-dot" style="background:#f59e0b; box-shadow:0 0 8px #f59e0b;"></span>
                Mode: Live
            </span>
            '''

        next_run_str = getattr(bot, 'next_run_timestamp', None)
        countdown_script = ""
        
        if bot_running and next_run_str:
            try:
                next_run_dt = datetime.fromisoformat(next_run_str)
                next_run_epoch = int(next_run_dt.timestamp() * 1000)
                
                # Calculate initial time on server
                now_dt = datetime.now()
                diff_seconds = int((next_run_dt - now_dt).total_seconds())
                if diff_seconds <= 0:
                    initial_text = "⏱️ Next: Running..."
                else:
                    initial_mins = diff_seconds // 60
                    initial_secs = diff_seconds % 60
                    initial_text = f"⏱️ Next in: {initial_mins}m {initial_secs:02d}s"
                    
                cycle_badge_html = f'''
                <span class="badge" id="countdown-badge" style="background:rgba(6,182,212,0.08); border:1px solid #06b6d4; color:#06b6d4;">
                    <span class="status-dot" style="background:#06b6d4; box-shadow:0 0 8px #06b6d4;"></span>
                    <span id="countdown-text">{initial_text}</span>
                </span>
                '''
                
                countdown_script = f'''
                <script>
                    (function() {{
                        const nextTimeMs = {next_run_epoch};
                        function runTicker() {{
                            const textEl = document.getElementById('countdown-text');
                            if (!textEl) return;
                            const now = Date.now();
                            const diff = nextTimeMs - now;
                            if (diff <= 0) {{
                                textEl.innerHTML = '⏱️ Next: Running...';
                                return;
                            }}
                            const mins = Math.floor(diff / 60000);
                            const secs = Math.floor((diff % 60000) / 1000);
                            const padSecs = String(secs).padStart(2, '0');
                            textEl.innerHTML = '⏱️ Next in: ' + mins + 'm ' + padSecs + 's';
                        }}
                        setInterval(runTicker, 1000);
                        runTicker();
                    }})();
                </script>
                '''
            except Exception:
                cycle_badge_html = '''
                <span class="badge" style="background:rgba(148,163,184,0.1); border:1px solid #94a3b8; color:#94a3b8;">
                    <span class="status-dot" style="background:#94a3b8;"></span>
                    ⏱️ Next: --
                </span>
                '''
        else:
            if bot_running:
                cycle_badge_html = f'''
                <span class="badge" style="background:rgba(6,182,212,0.08); border:1px solid #06b6d4; color:#06b6d4;">
                    <span class="status-dot" style="background:#06b6d4; box-shadow:0 0 8px #06b6d4;"></span>
                    ⏱️ Next: {interval_mins}m
                </span>
                '''
            else:
                cycle_badge_html = '''
                <span class="badge" style="background:rgba(148,163,184,0.1); border:1px solid #94a3b8; color:#94a3b8;">
                    <span class="status-dot" style="background:#94a3b8;"></span>
                    ⏱️ Next: Offline
                </span>
                '''

        badges_row_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                    background: transparent;
                    display: flex;
                    justify-content: flex-end;
                    align-items: center;
                    height: 100%;
                }}
                .badges-container {{
                    display: flex;
                    gap: 6px;
                    flex-wrap: wrap;
                    justify-content: flex-end;
                    font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
                }}
                .badge {{
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                    border-radius: 16px;
                    padding: 2px 7px;
                    font-size: 0.68rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    white-space: nowrap;
                }}
                .status-dot {{
                    width: 5px;
                    height: 5px;
                    border-radius: 50%;
                    display: inline-block;
                }}
            </style>
        </head>
        <body>
            <div class="badges-container">
                {api_badge_html}
                {bot_badge_html}
                {mode_badge_html}
                {cycle_badge_html}
            </div>
            {countdown_script}
        </body>
        </html>
        '''

        # Render the badges component with height=38 to ensure wrapping is safe if width is ever squeezed
        components.html(badges_row_html, height=38)
        
        # Render the proxy details & balance underneath
        proxy_short = f"{conn['proxy_address'][:6]}...{conn['proxy_address'][-4:]}" if conn.get('proxy_address') else "Unknown Proxy"
        if conn.get("connected"):
            balance_html = clean_html(f'''
            <div style="display:flex; flex-direction:column; align-items:flex-end; gap:2px; padding-top:4px;">
                <span style="font-family:monospace; font-size:0.72rem; color:#888;">
                    Proxy: {proxy_short}
                </span>
                <span class="funds-amount" style="font-family:monospace; font-size:1.15rem; color:#00e676; font-weight:800; margin-top:4px; display:inline-flex; align-items:center; gap:5px;">
                    💰 {conn.get('proxy_balance', 0.0):,.2f} USDC
                </span>
            </div>
            ''')
            st.markdown(balance_html, unsafe_allow_html=True)
        else:
            balance_html = clean_html(f'''
            <div style="display:flex; flex-direction:column; align-items:flex-end; gap:2px; padding-top:4px;">
                <span style="font-family:sans-serif; font-size:0.72rem; color:#ff4444;">
                    API Offline
                </span>
                <span class="funds-amount" style="font-family:monospace; font-size:1.15rem; color:#00e676; font-weight:800; margin-top:4px; display:inline-flex; align-items:center; gap:5px;">
                    💰 {conn.get('proxy_balance', 0.0):,.2f} USDC
                </span>
            </div>
            ''')
            st.markdown(balance_html, unsafe_allow_html=True)

    st.markdown("""<hr class="section-line">""", unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    # Wallet Details Expander
    with st.sidebar.expander("💼 Wallet Details", expanded=False):
        eoa = conn.get("eoa_address") or "N/A"
        proxy = conn.get("proxy_address") or "N/A"
        eoa_short = f"{eoa[:6]}...{eoa[-4:]}" if len(eoa) > 10 else eoa
        proxy_short_full = f"{proxy[:6]}...{proxy[-4:]}" if len(proxy) > 10 else proxy

        st.markdown("<span style='font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;'>EOA Wallet</span>", unsafe_allow_html=True)
        st.code(eoa_short, language=None)

        st.markdown("<span style='font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;'>Proxy Wallet</span>", unsafe_allow_html=True)
        st.code(proxy_short_full, language=None)
        
        if conn.get("error"):
            st.warning(f"**Notice:** {conn['error']}")

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
    st.sidebar.subheader("Trading Process Control")
    bot_running = check_bot_process_running()
    
    paper_mode = st.sidebar.checkbox("Paper Trading Mode", value=True, help="Enable for simulated trading")
    bot_interval = st.sidebar.number_input(
        "Bot Run Interval (mins)",
        min_value=1,
        max_value=1440,
        value=60,
        step=5,
        help="How often the background bot runs a cycle (in minutes)"
    )

    if bot_running:
        if st.sidebar.button("🛑 Stop Trading Bot", key="stop_bot_btn", use_container_width=True):
            if stop_bot_process():
                st.toast("Bot process stopped successfully!", icon="🛑")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to stop the bot process.")
    else:
        mode_str = "paper" if paper_mode else "live"
        if st.sidebar.button(f"🚀 Start Trading Bot ({mode_str.upper()})", key="start_bot_btn", use_container_width=True):
            if start_bot_process("paper" if paper_mode else "live", bot_interval):
                st.toast(f"Bot process started in {mode_str.upper()} mode!", icon="🚀")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to start the bot process.")

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
                    st.plotly_chart(fig_edge, width="stretch")

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
                    st.plotly_chart(fig_scatter, width="stretch")

    # Tab 2: Crypto markets
    with tab2:
        render_crypto_markets_tab(bot)

    # Tab 3: Portfolio
    with tab3:
        # 1. Resolve Allocations Data first (so we can display metrics)
        allocations_df = pd.DataFrame()
        status_val = "N/A"
        iterations_val = "N/A"
        fw_gap_val = "N/A"
        final_obj_val = "N/A"
        has_optimizer_data = False

        if not markets_df.empty:
            try:
                markets_for_optimization = markets_df['raw_market'].tolist()
                allocations, status, info = bot.optimizer.optimize(
                    markets_for_optimization,
                    bot.constraints
                )
                status_val = status.value
                iterations_val = info.get('iterations', 'N/A')
                fw_gap_val = info.get('fw_gap', 'N/A')
                final_obj_val = info.get('final_objective', 'N/A')
                has_optimizer_data = True

                allocations_data = []
                for i, (market, alloc) in enumerate(zip(markets_for_optimization, allocations)):
                    if alloc < 0.001:
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
            except Exception as e:
                # Fallback to top markets by edge as placeholder
                if 'Edge' in markets_df.columns:
                    markets_df['Edge_Numeric'] = markets_df['Edge'].str.replace('%', '').astype(float)
                    top_markets = markets_df.nlargest(5, 'Edge_Numeric') if len(markets_df) >= 5 else markets_df
                    total_edge = top_markets['Edge_Numeric'].sum()

                    allocations_data = []
                    for idx, (_, market) in enumerate(top_markets.iterrows()):
                        edge_val = float(market['Edge'].replace('%', ''))
                        allocation_pct = (edge_val / total_edge) * 100 if total_edge > 0 else 0
                        size_usd = (allocation_pct / 100) * float(bot.capital)

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

        # 2. Render Polymarket Profile & Profit/Loss box (Side-by-Side)
        perf_history = get_performance_history(bot)
        if not perf_history:
            perf_history = [{
                'timestamp': datetime.now().isoformat(),
                'realized_pnl': 0.0,
                'expected_pnl': 0.0,
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0
            }]

        col_left, col_right = st.columns([1, 1.25])
        
        with col_left:
            positions_value = allocations_df['Size ($)'].sum() if not allocations_df.empty else 0.0
            biggest_win = bot.performance_metrics.get('biggest_win_history', 0.0)
            predictions_count = bot.performance_metrics.get('total_trades', 0)
            cycles = bot.cycles_completed
            
            st.markdown(
                clean_html(f"""
                <div style="background-color: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1.5rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; min-height: 290px;">
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
                        <div style="position: relative; width: 70px; height: 70px; flex-shrink: 0;">
                            <div style="width: 70px; height: 70px; border-radius: 50%; background: radial-gradient(circle, #f472b6 0%, #3b82f6 70%, #8b5cf6 100%);"></div>
                            <div style="width: 20px; height: 20px; border-radius: 50%; background: #fbbf24; border: 2px solid var(--bg-surface); position: absolute; bottom: 0; right: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: #12131c; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">★</div>
                        </div>
                        <div>
                            <div style="font-size: 1.5rem; font-weight: 800; font-family: Space Grotesk; color: var(--text-bright); line-height: 1.2;">polybot</div>
                            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Joined Mar 2026 • {cycles} cycles</div>
                        </div>
                    </div>
                    <div style="display: flex; border-top: 1px solid var(--border); padding-top: 1.5rem; justify-content: space-between;">
                        <div style="flex: 1; text-align: left;">
                            <div style="font-size: 1.3rem; font-weight: 700; font-family: JetBrains Mono; color: var(--text-bright);">${positions_value:,.2f}</div>
                            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-family: Space Grotesk; font-weight: 600; margin-top: 2px;">Positions Value</div>
                        </div>
                        <div style="width: 1px; background-color: var(--border); margin: 0 0.75rem;"></div>
                        <div style="flex: 1; text-align: left;">
                            <div style="font-size: 1.3rem; font-weight: 700; font-family: JetBrains Mono; color: var(--text-bright);">${biggest_win:,.2f}</div>
                            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-family: Space Grotesk; font-weight: 600; margin-top: 2px;">Biggest Win</div>
                        </div>
                        <div style="width: 1px; background-color: var(--border); margin: 0 0.75rem;"></div>
                        <div style="flex: 1; text-align: left;">
                            <div style="font-size: 1.3rem; font-weight: 700; font-family: JetBrains Mono; color: var(--text-bright);">{predictions_count:,}</div>
                            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-family: Space Grotesk; font-weight: 600; margin-top: 2px;">Predictions</div>
                        </div>
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

        with col_right:
            with st.container(border=True):
                # Header row: Profit/Loss label and ranges
                col_header_title, col_header_range = st.columns([1, 1.25])
                
                with col_header_title:
                    current_pnl = perf_history[-1]['total_pnl'] if perf_history else 0.0
                    pnl_sign = "▲" if current_pnl >= 0 else "▼"
                    pnl_class = "positive" if current_pnl >= 0 else "negative"
                    st.markdown(
                        f"<div class='{pnl_class}' style='font-family: Space Grotesk; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 4px; padding-top: 4px;'>"
                        f"<span>{pnl_sign}</span> Profit/Loss"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                with col_header_range:
                    timeframe = st.segmented_control(
                        "Timeframe Selector",
                        options=["1D", "1W", "1M", "1Y", "YTD", "ALL"],
                        default="ALL",
                        key="portfolio_timeframe",
                        label_visibility="collapsed"
                    )

                # Filter history based on range
                df_perf = pd.DataFrame(perf_history)
                df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
                latest_time = df_perf['timestamp'].max()
                
                if timeframe == "1D":
                    threshold = latest_time - timedelta(days=1)
                elif timeframe == "1W":
                    threshold = latest_time - timedelta(days=7)
                elif timeframe == "1M":
                    threshold = latest_time - timedelta(days=30)
                elif timeframe == "1Y":
                    threshold = latest_time - timedelta(days=365)
                elif timeframe == "YTD":
                    threshold = datetime(latest_time.year, 1, 1)
                else:
                    threshold = df_perf['timestamp'].min()
                
                df_timeframe = df_perf[df_perf['timestamp'] >= threshold].copy()
                if not df_timeframe.empty:
                    start_val = df_timeframe.iloc[0]['total_pnl']
                    end_val = df_timeframe.iloc[-1]['total_pnl']
                    pnl_diff = end_val - start_val
                else:
                    pnl_diff = current_pnl
                
                sign = "+" if pnl_diff >= 0 else ""
                color = "#10b981" if pnl_diff >= 0 else "#ef4444"
                timeframe_label = {
                    "1D": "Past 24 Hours",
                    "1W": "Past Week",
                    "1M": "Past Month",
                    "1Y": "Past Year",
                    "YTD": "Year-To-Date",
                    "ALL": "All-Time"
                }.get(timeframe, "All-Time")
                
                st.markdown(
                    f"<div style='margin-top: 0.2rem;'>"
                    f"<span style='font-size: 1.8rem; font-weight: 800; font-family: Space Grotesk; color: var(--text-bright);'>${abs(pnl_diff):,.2f}</span>"
                    f"<span style='font-size: 1.2rem; font-weight: 600; color: {color}; margin-left: 0.4rem;'>{sign}</span>"
                    f"</div>"
                    f"<div style='font-size: 0.75rem; color: var(--text-muted); font-family: Space Grotesk; margin-top: -2px;'>{timeframe_label}</div>",
                    unsafe_allow_html=True
                )
                
                # Plotly Chart
                fig = create_portfolio_value_chart(
                    perf_history,
                    timeframe,
                    bot.capital,
                    layout=plotly_layout,
                    theme_mode=theme_mode
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 3. Render Positions and Table Below
        if not allocations_df.empty:
            col1, col2 = st.columns([1, 1])

            with col1:
                allocations_fig = create_allocations_chart(allocations_df, layout=plotly_layout, theme_mode=theme_mode)
                if allocations_fig:
                    st.plotly_chart(allocations_fig, width="stretch")
                else:
                    st.info("No significant allocations to display")

            with col2:
                st.subheader("Position Details")
                display_allocations_table(allocations_df)

            if has_optimizer_data:
                with st.expander("Optimization Details"):
                    st.write(f"**Status:** {status_val}")
                    st.write(f"**Iterations:** {iterations_val}")
                    st.write(f"**Expected Log Utility:** {final_obj_val:.6f}" if isinstance(final_obj_val, float) else f"**Expected Log Utility:** {final_obj_val}")
                    st.write(f"**Frank-Wolfe Gap:** {fw_gap_val:.6e}" if isinstance(fw_gap_val, float) else f"**Frank-Wolfe Gap:** {fw_gap_val}")
        else:
            st.info("No current allocations to display")

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
                st.plotly_chart(perf_fig, width="stretch")
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
                        st.dataframe(display_trades, width="stretch")
                    else:
                        st.dataframe(trades_df, width="stretch")
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
                st.plotly_chart(risk_fig, width="stretch")
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
                st.dataframe(quotes_df, width="stretch")
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

                        st.dataframe(formatted_trades, width="stretch")
                    else:
                        st.dataframe(display_trades, width="stretch")

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
                            with open(bot.alert_manager.alert_file, 'w', encoding='utf-8') as f:
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
                            time_str = timestamp
                            if timestamp != 'Unknown time':
                                try:
                                    dt = datetime.fromisoformat(timestamp)
                                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                                except Exception:
                                    pass

                            # Style based on level
                            if level == 'critical':
                                st.error(f"🚨 **{source}** [{time_str}]: {message}")
                            elif level == 'error':
                                st.error(f"❌ **{source}** [{time_str}]: {message}")
                            elif level == 'warning':
                                st.warning(f"⚠️ **{source}** [{time_str}]: {message}")
                            else:
                                st.info(f"ℹ️ **{source}** [{time_str}]: {message}")

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
                        st.dataframe(trade_df, width="stretch")
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

    # Client-side dynamic ticking script using Same-Origin context via onerror hack
    next_run_str = getattr(bot, 'next_run_timestamp', None)
    if bot_running and next_run_str:
        try:
            next_run_dt = datetime.fromisoformat(next_run_str)
            next_run_epoch = int(next_run_dt.timestamp() * 1000)
            st.markdown(f"""
            <img src="does-not-exist" onerror="(function(){{if(window.cycleCountdownIntervalId){{clearInterval(window.cycleCountdownIntervalId);}}function runTicker(){{const badge=document.getElementById('cycle-countdown-badge');if(!badge)return;const nextTimeMs=parseInt(badge.getAttribute('data-next-run-ms'),10);if(isNaN(nextTimeMs))return;const now=Date.now();const diff=nextTimeMs-now;if(diff<=0){{badge.innerHTML='⏱️ Cycle Timeout: Running...';clearInterval(window.cycleCountdownIntervalId);return;}}const mins=Math.floor(diff/60000);const secs=Math.floor((diff%60000)/1000);badge.innerHTML='⏱️ Cycle Timeout in: '+mins+'m '+String(secs).padStart(2,'0')+'s';}}window.cycleCountdownIntervalId=setInterval(runTicker,1000);runTicker();}})()" style="display:none;"/>
            """, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering dynamic countdown component: {e}")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
