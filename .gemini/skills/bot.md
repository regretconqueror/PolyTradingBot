# /bot — Start the PolyTradingBot

## Description
Starts the Polymarket Trading Bot in paper mode along with the Streamlit dashboard.

## Instructions

When the user invokes `/bot`, follow these steps:

1. **Activate the virtual environment, start both the bot and dashboard, and open the browser**:
   - Start the bot:
     ```
     .venv\Scripts\activate ; python -u run.py --mode paper --interval 60
     ```
   - Start the dashboard:
     ```
     .venv\Scripts\activate ; streamlit run dashboard/streamlit_app.py --server.headless true
     ```
   - Open the dashboard in the default browser:
     ```
     start http://localhost:8501
     ```
   Run these from the workspace root: `c:\AI_WORK\PolyTradingBot`. Note: Use `python -u` for unbuffered logs to capture the startup banner, and `--server.headless true` for Streamlit.

2. **Report the startup banner** — The bot prints a banner with Capital, Mode, State file, and Model info. Relay this to the user.

3. **Provide Status and Links** — Let the user know:
   - The bot is running in **paper trading** mode
   - The dashboard is running at [http://localhost:8501](http://localhost:8501)
   - The bot will cycle every **60 minutes**
   - They can ask you to stop either the bot or the dashboard anytime (you'll kill the background task)

4. **If the user wants to customize**, they can say things like:
   - `/bot live` → run with `--mode live --confirm-live`
   - `/bot capital 500` → run with `--capital 500`
   - `/bot interval 30` → run with `--interval 30`
   - These can be combined: `/bot capital 200 interval 15`

   Parse the user's message for these options and adjust the bot startup command accordingly:
   - `live` → add `--mode live --confirm-live`
   - `capital <N>` → add `--capital <N>`
   - `interval <N>` → add `--interval <N>`

5. **For live mode**, always warn the user before starting:
   > ⚠️ You are about to start the bot in LIVE trading mode. Real orders will be placed. Confirm you want to proceed.

   Wait for explicit user confirmation before running the live command.

6. **Keep both tasks running in the background** so the user can continue chatting. Use async/background execution.
