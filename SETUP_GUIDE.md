# PolyTradingBot Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment

#### For Paper Trading (Recommended for Testing):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and set:
PAPER_MODE=true
# You can leave API credentials empty or use dummy values for paper trading
POLYMARKET_API_KEY=test_key
POLYMARKET_API_SECRET=test_secret
POLYMARKET_PRIVATE_KEY=test_key

# Adjust capital if desired
CAPITAL=10000
```

#### For Live Trading:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual Polymarket credentials:
# Get these from your Polymarket account (may require contacting support or checking developer settings)
POLYMARKET_API_KEY=your_actual_api_key
POLYMARKET_API_SECRET=your_actual_api_secret
POLYMARKET_PRIVATE_KEY=your_ethereum_wallet_private_key

# Important: Never commit your .env file - it's already in .gitignore
PAPER_MODE=false  # Set to false for live trading
CAPITAL=10000     # Set your trading capital
```

### 3. Run the Bot

#### Paper Trading:
```bash
python run.py --mode paper
# OR
python run.py  # Defaults to paper mode if PAPER_MODE=true in .env
```

#### Live Trading:
```bash
python run.py --mode live
# OR
python run.py  # Will use live mode if PAPER_MODE=false in .env
```

### 4. Adjust Trading Parameters (Optional)

You can adjust these in your `.env` file:
- `CAPITAL` - Trading capital in USDC
- `PAPER_MODE` - `true` for simulation, `false` for live trading
- Optional risk parameters:
  - `MAX_EXPOSURE` (default: 0.75 = 75% of capital)
  - `MAX_POSITION` (default: 0.20 = 20% per position)
  - `MAX_DRAWDOWN` (default: 0.15 = 15% max drawdown)
  - `MIN_BET_SIZE` (default: 0.02 = 2% minimum position size)

### 5. Understanding the Output

The bot will display:
- Market selection based on your probability models
- Portfolio optimization using ProjectFW (Frank-Wolfe) algorithm
- Order execution (simulated in paper mode, real in live mode)
- Performance metrics including win rate, P&L, Sharpe ratio
- Risk summaries showing exposure, VaR, and correlation analysis

### Important Notes

1. **Start Small**: Begin with paper trading to validate strategies
2. **Security First**: Never share your private key or commit .env to version control
3. **Test Thoroughly**: Run multiple paper trading sessions before going live
4. **Monitor Performance**: Use the built-in analytics to refine your strategies
5. **Start with Low Capital**: When going live, begin with amounts you can afford to lose

### Troubleshooting

- **"No tradeable markets found"**: Try reducing the `--min-edge` parameter or check your internet connection
- **Authentication errors**: Verify your API credentials are correct
- **Order rejected**: Check that your order sizes comply with market minimums and your risk limits
- **Performance seems off**: Remember that paper trading simulation uses simplified P&L calculations

### Next Steps After Setup

1. Run several paper trading cycles to see how the strategy performs
2. Examine the generated trade history and performance reports
3. Adjust your risk parameters based on observed volatility
4. Consider implementing the pending enhancements (rebalancing, multi-outcome support, etc.)
5. When confident, transition to live trading with small position sizes

For questions or issues, check the code comments or consider implementing additional features from the TODO list.