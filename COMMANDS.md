# PolyTradingBot — Custom Commands

Use these commands naturally in conversation. Each maps to a specific capability or workflow.

---

## /search_net \<query\>
**What it does:** Searches the web for \<query\>, returns the most relevant results.

**When to use:** Finding GitHub repos, looking up token/CLOB API docs, checking recent news about Polymarket, researching trading strategies, looking up market data.

**Examples:**
- `/search_net arbitrage trading python bot github`
- `/search_net Polymarket CLOB API documentation 2026`
- `/search_net binary options market making strategy`

---

## /docs \<topic\>
**What it does:** Fetches documentation for a specific topic — API references, library docs, guides.

**When to use:** Reading up on Polymarket API endpoints, checking how the Gamma API works, looking up risk management libraries.

**Examples:**
- `/docs Polymarket CLOB API order placement`
- `/docs Frank-Wolfe optimization algorithm`
- `/docs Kelly criterion portfolio`

---

## /backtest \<strategy\> \<market\>
**What to say:** "Run a backtest of the yes_no_arb strategy on 2026 election markets"

**What it does:** Uses the `backtest/` module to simulate strategy performance on historical data.

**Examples:**
- `/backtest yes_no_arb on geopolitics markets last 30 days`
- `/backtest ensemble_model on high-volume sports markets`

---

## /risk_check
**What to say:** "run a /risk_check on my current portfolio"

**What it does:** Runs the full risk assessment — VaR, correlation matrix, drawdown, stop-loss triggers.

**Output:** VaR at 95%, max correlation pair, exposure ratios, risk violations.

---

## /edge_scan
**What to say:** "do an /edge_scan for crypto markets"

**What it does:** Scans all available Polymarket markets by category and reports which ones have positive edge (estimated probability > market price minus fees).

**Output:** Ranked list of tradeable edge opportunities, sorted by edge size.

---

## /paper_summary
**What to say:** "give me a /paper_summary"

**What it does:** Prints a full paper trading performance report — total P&L, win rate, open positions, realized vs unrealized, slippage breakdown.

---

## /tune \<model\>
**What to say:** "tune the ensemble model for low-vol markets"

**What it does:** Adjusts probability model parameters (momentum weights, lookback windows, long-shot thresholds) to optimize for the requested market type.

**Example models:** `MarketSentimentModel`, `VolatilityAdjustedModel`, `EnsembleModel`

---

## /deploy
**When to use:** When you're ready to switch from paper to live trading.

**What to say:** "prepare /deploy to live mode with $50 capital"

**What it does:** Walks through the checklist: verify API keys, check risk constraints, confirm min_bet_size, ensure live_trading_enabled=true, prompt before executing first live order.

---

## /alert_history
**What to say:** "show me /alert_history for today"

**What it does:** Reads `alerts.json` and displays all risk alerts, stop-loss triggers, and VaR breaches that occurred in the requested timeframe.