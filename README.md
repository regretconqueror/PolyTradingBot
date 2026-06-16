# Polymarket Trading Bot

Advanced prediction market trading bot using ProjectFW (Frank-Wolfe) optimization and Kelly Criterion for optimal position sizing.

## Overview

This trading bot implements a sophisticated portfolio optimization framework for Polymarket prediction markets, combining:

- **Kelly Criterion Optimization**: Maximizes long-term growth of capital by optimizing position sizes based on estimated edges
- **Frank-Wolfe Algorithm (ProjectFW)**: Efficient convex optimization that handles complex constraints without projections
- **Risk Management**: Position limits, category constraints, drawdown protection, and VaR monitoring
- **Multiple Strategies**: Ensemble probability modeling plus Yes/No arbitrage scanning
- **Live & Paper Trading**: Seamless transition between testing and production modes
- **Performance Tracking**: Comprehensive metrics and reporting

## Features

### Core Optimization
- **Kelly Criterion**: Mathematically optimal bet sizing for maximizing expected logarithmic utility
- **Frank-Wolfe Solver**: Handles linear constraints (exposure limits, position caps, category limits) efficiently
- **Multi-objective Optimization**: Balances expected return against risk through constraint management

### Risk Management
- **Exposure Limits**: Control overall market exposure and single-position concentration
- **Category Constraints**: Diversify across market categories (Crypto, Politics, Sports, Science)
- **Drawdown Protection**: Monitor and alert on portfolio drawdowns
- **VaR Calculation**: Value-at-Risk monitoring for tail risk
- **Correlation Monitoring**: Detect and alert on excessive position correlations

### Trading Capabilities
- **Paper Trading Mode**: Test strategies with zero risk using simulated execution
- **Live Trading Mode**: Production-ready execution via Polymarket CLOB API (guarded)
- **Yes/No Arbitrage**: Detect and exploit mispriced YES/NO pairs that don't sum to ~1.0
- **Dynamic Rebalancing**: Periodically optimize portfolio as market conditions change
- **Order Management**: Guarded execution with size limits, dry-run protection, and error recovery

### Technical Features
- **Modular Architecture**: Separate concerns for easy maintenance and extension
- **Comprehensive Logging**: Detailed operational logs for debugging and audit
- **Alert System**: Configurable notifications for risk events and operational issues
- **Performance Metrics**: Track win rate, P&L, Sharpe ratio, profit factor, and more
- **Configuration Management**: Environment-driven settings with sensible defaults

## Installation

### Prerequisites
- Python 3.8+
- Git (for cloning the repository)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/PolyTradingBot.git
cd PolyTradingBot

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Polymarket API credentials and preferences
```

### Environment Variables
Copy `.env.example` to `.env` and configure:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POLYMARKET_API_KEY` | Polymarket API key | (empty) | Yes (for live) |
| `POLYMARKET_API_SECRET` | Polymarket API secret | (empty) | Yes (for live) |
| `POLYMARKET_API_PASSPHRASE` | Polymarket API passphrase | (empty) | Yes (for live) |
| `POLYMARKET_PRIVATE_KEY` | Wallet private key for signing | (empty) | Yes (for live) |
| `POLYMARKET_FUNDER_ADDRESS` | Wallet address for funding | (empty) | Yes (for live) |
| `POLYMARKET_SIGNATURE_TYPE` | Signature type (usually 3) | 3 | No |
| `CAPITAL` | Trading capital in USD | 10000 | No |
| `PAPER_MODE` | Enable paper trading (no real orders) | true | No |
| `LIVE_TRADING_ENABLED` | Enable live trading (overrides PAPER_MODE) | false | No |
| `LIVE_DRY_RUN` | Simulate live orders without submission | true | No |
| `MAX_LIVE_ORDER_SIZE` | Maximum USD size per live order | 25.0 | No |
| `MAX_LIVE_ORDERS_PER_CYCLE` | Max live orders per trading cycle | 3 | No |
| `MAX_EXPOSURE` | Maximum total portfolio exposure | 0.75 | No |
| `MAX_POSITION` | Maximum single position size | 0.20 | No |
| `MAX_DRAWDOWN` | Maximum allowed drawdown | 0.15 | No |
| `MIN_BET_SIZE` | Minimum position size | 0.02 | No |

### Category Limits (Optional)
Set category-specific exposure limits in `.env`:
```
CRYPTO=0.30
POLITICS=0.25
SPORTS=0.20
SCIENCE=0.15
```

## Usage

### Paper Trading (Recommended for Testing)
```bash
# Ensure these are set in .env:
# PAPER_MODE=true
# LIVE_TRADING_ENABLED=false
python run.py
```

### Live Trading
```bash
# Ensure these are set in .env:
# PAPER_MODE=false
# LIVE_TRADING_ENABLED=true
# LIVE_DRY_RUN=false  # Set to false for real orders
# All API credentials configured
python run.py
```

### Live Trading Simulation (Dry Run)
```bash
# Test live trading flow without real orders:
# PAPER_MODE=false  
# LIVE_TRADING_ENABLED=true
# LIVE_DRY_RUN=true   # Simulates orders
python run.py
```

## Architecture

```
PolyTradingBot/
├── bot/
│   ├── execution.py      # Order execution engine (guarded live trading)
│   ├── alert_manager.py  # Notification system
│   └── trading_bot.py    # Main bot orchestration
├── core/
│   ├── optimizer.py      # Frank-Wolfe Kelly Criterion optimizer
│   ├── models.py         # Data models and constraints
│   └── risk_manager.py   # Risk monitoring and limits
├── data/
│   └── api_client.py     # Polymarket API interface
├── dashboard/
│   └── streamlit_app.py  # Streamlit dashboard for visualization
├── strategies/
│   ├── __init__.py       # Strategy exports
│   ├── example_strategy.py # Example probability models
│   └── yes_no_arb.py     # Yes/No arbitrage scanner
├── config/
│   └── settings.py       # Environment-based configuration
├── test/                 # Unit tests
├── run.py                # Entry point
└── bot_state.json        # State persistence file (created automatically)
```

## Continuous Operation

The bot supports continuous operation with automatic state persistence and graceful shutdown handling.

### Running Continuously
Use the `--interval` flag to specify minutes between trading cycles:
```bash
# Run every 5 minutes
python run.py --interval 5

# Run every 30 minutes  
python run.py --interval 30
```

### State Persistence
The bot automatically saves its state to `bot_state.json` after each successful cycle and on graceful shutdown. This includes:
- Trade history and positions
- Performance metrics  
- Risk manager data (positions, price history)
- Alert manager state
- Cycle counters and peak portfolio values

On startup, the bot automatically loads `bot_state.json` if it exists, resuming from where it left off.

### Graceful Shutdown
The bot handles SIGINT (Ctrl+C) and SIGTERM signals to save state before exiting, ensuring no data loss.

## Exit Signal Framework

The bot implements five distinct exit strategies that can trigger position closure:

1. **Take Profit**: Close when profit reaches a specified percentage of position value
2. **Stop Loss**: Close when loss reaches a specified percentage of position value  
3. **Momentum Reversal**: Close when the market probability moves significantly against our position
4. **Time Exit**: Close positions that have been open longer than a maximum duration
5. **Stale Position**: Close positions that haven't seen price movement for an extended period

Each exit type is evaluated on every cycle, and the bot will close positions when any exit condition is met.

## Market Settlement

For resolved markets, the bot automatically detects settlement and books realized P&L:

- Markets are checked for settlement status on each cycle
- When a market closes (`closed=true`) or becomes inactive (`active=false`), associated trades are marked as SETTLED
- For binary outcomes:
  - If YES token resolves to 1: realized_pnl = (1 - entry_price) * shares
  - If YES token resolves to 0: realized_pnl = -entry_price * shares
- Settlement P&L is recorded and trades transition from FILLED → SETTLED status
- This ensures accurate tracking of actual vs. expected performance

## Dashboard

An interactive Streamlit dashboard is included for real-time monitoring:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard provides:
- Real-time market data and opportunity scanning
- Portfolio allocation visualization with pie charts and bar graphs
- Performance analytics including win rate, P&L, and risk metrics
- Active order monitoring and trade history
- Risk gauge displays for exposure, position limits, and drawdown
- Detailed trade logs with filtering capabilities

## State File

The `bot_state.json` file enables seamless recovery:
- Automatically created/updated - no manual intervention needed
- Contains all necessary state to resume trading exactly where left off
- Safe to delete - bot will start fresh if file is missing or corrupted
- Included in `.gitignore` to prevent accidental commitment of sensitive data

## Trade Status Lifecycle

Trades progress through a defined lifecycle:
1. **PENDING**: Order submitted but not yet filled
2. **FILLED**: Order executed, position open
3. **EXITED**: Position closed via exit signal (take profit, stop loss, etc.)
4. **SETTLED**: Market resolved, final P&L booked (replaces EXITED for settled markets)
5. **EXPIRED**: Market expired without resolution
6. **FAILED**: Order execution failed
7. **DRY_RUN**: Simulated trade in dry-run mode

Only EXITED and SETTLED trades contribute to realized P&L metrics. FILLED positions show unrealized P&L based on current market prices.

## Configuration Options

### Risk Parameters
Adjust these in `.env` to match your risk tolerance:

- **Risk Per Trade**: Controlled by Kelly criterion via optimizer
- **Max Exposure**: `MAX_EXPOSURE` (default 0.75 = 75% of capital at risk)
- **Max Position**: `MAX_POSITION` (default 0.20 = 20% per position)
- **Category Limits**: Per-category exposure limits for diversification
- **Drawdown Limit**: `MAX_DRAWDOWN` (default 0.15 = 15% max drawdown)
- **Minimum Bet**: `MIN_BET_SIZE` (default 0.02 = 2% minimum position)

### Trading Behavior
- **Capital**: Set your trading capital with `CAPITAL`
- **Order Size**: `MAX_LIVE_ORDER_SIZE` caps individual order sizes
- **Order Frequency**: `MAX_LIVE_ORDERS_PER_CYCLE` limits live orders per cycle
- **Rebalancing**: Automatic every cycle; threshold hardcoded at 5% deviation

## Performance Metrics

The bot tracks and reports:
- **Win Rate**: Percentage of profitable trades
- **Total P&L**: Absolute and percentage profit/loss
- **Profit Factor**: Gross profit / gross loss
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Average Win/Loss**: Typical trade outcomes
- **Expected P&L**: Based on modeled edges (until settlement data integrated)

## Security Notes

1. **Never commit your `.env` file** - it contains sensitive API keys
2. **Start with paper trading** to verify strategy performance
3. **Use small position sizes** when transitioning to live trading
4. **Monitor alerts** - the bot provides warnings for risk conditions
5. **Keep software updated** - check for dependency updates regularly

## Development

### Running Tests
```bash
python -m pytest test/ -v
```

### Adding New Strategies
1. Create a new class in `strategies/` inheriting from `ProbabilityModel`
2. Implement the `estimate_probability(market_data)` method
3. Update the ensemble model in `trading_bot.py` or use directly

### Extending Risk Management
Modify `core/risk_manager.py` to add:
- New risk metrics
- Additional alert conditions
- Enhanced position limit logic

## Disclaimer

This software is for educational and informational purposes only. Trading prediction markets involves risk of loss. Past performance is not indicative of future results. Always trade responsibly and within your means. The developers are not responsible for any trading losses incurred while using this bot.

## License

MIT License - see LICENSE file for details.

## Acknowledgements

- Polymarket team for providing the prediction market platform
- Authors of the Kelly Criterion and Frank-Wolfe algorithm research
- Open-source community for the libraries that make this possible