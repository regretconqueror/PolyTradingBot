#!/usr/bin/env python3
"""
Main entry point for Polymarket Trading Bot
"""
import argparse
import signal
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot import PolymarketTradingBot
from config import load_settings
from core import PortfolioConstraints
from strategies import EnsembleModel


def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument("--capital", type=float, help="Trading capital")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Trading mode")
    parser.add_argument("--interval", type=int, default=60, help="Trading interval (minutes)")
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required with --mode live before real order submission is allowed",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="bot_state.json",
        help="Path to JSON file for bot state persistence (default: bot_state.json)",
    )

    args = parser.parse_args()

    settings = load_settings()

    capital = args.capital or settings.capital
    paper_mode = args.mode == "paper"

    if args.mode == "live" and settings.paper_mode:
        print("WARNING: Live trading requested but PAPER_MODE=true in .env")
        print("Set PAPER_MODE=false in .env to enable live trading")
        return

    if args.mode == "live" and not args.confirm_live:
        print("WARNING: Live mode requires --confirm-live.")
        print("Without this flag the bot refuses to construct a live trading session.")
        return

    constraints = PortfolioConstraints(
        max_total_exposure=settings.max_exposure,
        max_single_position=settings.max_position,
        max_drawdown=settings.max_drawdown,
        min_bet_size=settings.min_bet_size,
    )

    model = EnsembleModel()

    bot = PolymarketTradingBot(
        capital=capital,
        constraints=constraints,
        model=model,
        api_key=settings.api_key,
        api_secret=settings.api_secret,
        passphrase=settings.api_passphrase,
        private_key=settings.private_key,
        funder_address=settings.funder_address,
        signature_type=settings.signature_type,
        live_trading_enabled=settings.live_trading_enabled and args.confirm_live,
        live_dry_run=settings.live_dry_run,
        max_live_order_size=settings.max_live_order_size,
        max_live_orders_per_cycle=settings.max_live_orders_per_cycle,
        paper_mode=paper_mode,
        enable_yes_no_arb=True,
    )

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal. Saving state and exiting...")
        bot.save_state(args.state_file)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"\n{'=' * 80}")
    print("POLYMARKET TRADING BOT")
    print(f"{'=' * 80}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
    if not paper_mode:
        print(f"Live enabled: {settings.live_trading_enabled and args.confirm_live}")
        print(f"Live dry run: {settings.live_dry_run}")
        print(f"Max live order size: ${settings.max_live_order_size:,.2f}")
        print(f"Max live orders/cycle: {settings.max_live_orders_per_cycle}")
    print(f"State file: {args.state_file}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'=' * 80}\n")

    # Load previous state if exists
    if args.state_file:
        bot.load_state(args.state_file)

    if args.interval > 0:
        print(f"Starting continuous trading bot with {args.interval} minute intervals...")
        print("Press Ctrl+C to stop gracefully")
        try:
            while True:
                bot.run()
                print(f"\nCycle completed. Waiting {args.interval} minutes until next cycle...")
                time.sleep(args.interval * 60)  # Convert minutes to seconds
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Saving state and exiting...")
            bot.save_state(args.state_file)
    else:
        print("Running one cycle...")
        bot.run()
        # Save state after single run
        bot.save_state(args.state_file)


if __name__ == "__main__":
    main()
