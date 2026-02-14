# Polymarket Trading Bot

Advanced prediction market trading bot using ProjectFW (Frank-Wolfe) optimization and Kelly Criterion for optimal position sizing.

## Features

- **Kelly Criterion Optimization**: Maximizes long-term log utility
- **Frank-Wolfe Algorithm**: Efficient convex optimization without projections
- **Risk Management**: Position limits, category constraints, drawdown protection
- **Paper Trading**: Test strategies without real money
- **Live Trading**: Production-ready execution via Polymarket CLOB API (not implemented in this version)
- **Yes/No Sum Arbitrage (Paper)**: Detects YES+NO < 1 opportunities

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
