---
name: bot
description: Triggers when the user inputs /bot or requests dashboard/bot execution.
---
# Bot Skill

This skill is triggered when the user executes `/bot` or requests bot execution. It starts both the trading bot and the dashboard simultaneously in parallel.

## Instructions
When the user executes the `/bot` command:
1. Propose and execute the commands to start the trading bot and the Streamlit dashboard simultaneously in parallel. Start them as background tasks:
   * **Trading Bot:**
     ```powershell
     .venv\Scripts\python.exe run.py
     ```
   * **Streamlit Dashboard:**
     ```powershell
     .venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py --browser.gatherUsageStats false
     ```
2. Inform the user that both the trading bot and dashboard have been started in parallel.
3. Use the `browser_subagent` to open a new tab/window pointing to the local URL (typically `http://localhost:8501`) to ensure the dashboard opens in a new tab.
