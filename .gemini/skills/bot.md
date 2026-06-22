# /bot — Start the PolyTradingBot

## Description
Starts the Polymarket Trading Bot in paper mode with default settings.

## Instructions

When the user invokes `/bot`, follow these steps:

1. **Activate the virtual environment and start the bot** by running:
   ```
   .venv\Scripts\activate ; python run.py --mode paper --interval 60
   ```
   Run this from the workspace root: `c:\AI_WORK\PolyTradingBot`

2. **Report the startup banner** — The bot prints a banner with Capital, Mode, State file, and Model info. Relay this to the user.

3. **The bot runs continuously** in 60-minute trading cycles. Let the user know:
   - The bot is running in **paper trading** mode
   - It will cycle every **60 minutes**
   - They can ask you to stop it anytime (you'll kill the background task)

4. **If the user wants to customize**, they can say things like:
   - `/bot live` → run with `--mode live --confirm-live`
   - `/bot capital 500` → run with `--capital 500`
   - `/bot interval 30` → run with `--interval 30`
   - These can be combined: `/bot capital 200 interval 15`

   Parse the user's message for these options and adjust the command accordingly:
   - `live` → add `--mode live --confirm-live`
   - `capital <N>` → add `--capital <N>`
   - `interval <N>` → add `--interval <N>`

5. **For live mode**, always warn the user before starting:
   > ⚠️ You are about to start the bot in LIVE trading mode. Real orders will be placed. Confirm you want to proceed.

   Wait for explicit user confirmation before running the live command.

6. **Keep the bot running as a background task** so the user can continue chatting. Use async/background execution.
