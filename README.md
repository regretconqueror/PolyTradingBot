# Polymarket Trading Bot

Advanced prediction market trading bot using ProjectFW (Frank-Wolfe) optimization and Kelly Criterion for optimal position sizing.

## Features

- **Kelly Criterion Optimization**: Maximizes long-term log utility
- **Frank-Wolfe Algorithm**: Efficient convex optimization without projections
- **Risk Management**: Position limits, category constraints, drawdown protection
- **Paper Trading**: Test strategies without real money
- **Live Trading**: Production-ready execution via Polymarket CLOB API

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run paper trading
python run.py --mode paper --capital 10000

# Run live trading (BE CAREFUL!)
python run.py --mode live --capital 10000