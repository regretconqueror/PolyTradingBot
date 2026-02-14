#!/usr/bin/env python3
"""
Main entry point for Polymarket Trading Bot
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_settings
from core import PortfolioConstraints
from strategies import SimpleEdgeModel
from bot import PolymarketTradingBot

def main():
    parser = argparse.ArgumentParser(description='Polymarket Trading Bot')
    parser.add_argument('--capital', type=float, help='Trading capital')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper', 
                       help='Trading mode')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Trading interval (minutes)')
    
    args = parser.parse_args()
    
    # Load settings
    settings = load_settings()
    
    # Override with command line args
    capital = args.capital or settings.capital
    paper_mode = args.mode == 'paper'
    
    if args.mode == 'live' and settings.paper_mode:
        print("⚠️  WARNING: Live trading requested but PAPER_MODE=true in .env")
        print("Set PAPER_MODE=false in .env to enable live trading")
        return
    
    # Setup constraints
    constraints = PortfolioConstraints(
        max_total_exposure=settings.max_exposure,
        max_single_position=settings.max_position,
        max_drawdown=settings.max_drawdown,
        min_bet_size=settings.min_bet_size
    )
    
    model = SimpleEdgeModel(edge_factor=0.04)
    
    bot = PolymarketTradingBot(
        capital=capital,
        constraints=constraints,
        model=model,
        api_key=settings.api_key,
        api_secret=settings.api_secret,
        paper_mode=paper_mode,
        enable_yes_no_arb=True
    )
    
    print(f"\n{'='*80}")
    print("POLYMARKET TRADING BOT")
    print(f"{'='*80}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*80}\n")
    
    if args.interval > 0:
        bot.start(interval_minutes=args.interval)
    else:
        bot.run()

if __name__ == "__main__":
    main()
