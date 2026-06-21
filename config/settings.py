"""Configuration management"""
import os
from dataclasses import dataclass
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # API Credentials
    api_key: str
    api_secret: str
    api_passphrase: str
    private_key: str
    funder_address: str
    signature_type: int
    
    # Trading
    capital: float
    paper_mode: bool
    live_trading_enabled: bool
    live_dry_run: bool
    max_live_order_size: float
    max_live_orders_per_cycle: int
    
    # Constraints
    max_exposure: float
    max_position: float
    max_drawdown: float
    min_bet_size: float
    
    # Category limits
    category_limits: Dict[str, float]
    
    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv('POLYMARKET_API_KEY', ''),
            api_secret=os.getenv('POLYMARKET_API_SECRET', ''),
            api_passphrase=os.getenv('POLYMARKET_API_PASSPHRASE', ''),
            private_key=os.getenv('POLYMARKET_PRIVATE_KEY', ''),
            funder_address=os.getenv('POLYMARKET_FUNDER_ADDRESS', ''),
            signature_type=int(os.getenv('POLYMARKET_SIGNATURE_TYPE', 3)),
            capital=float(os.getenv('CAPITAL', 10000)),
            paper_mode=os.getenv('PAPER_MODE', 'true').lower() == 'true',
            live_trading_enabled=os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true',
            live_dry_run=os.getenv('LIVE_DRY_RUN', 'true').lower() == 'true',
            max_live_order_size=float(os.getenv('MAX_LIVE_ORDER_SIZE', 25)),
            max_live_orders_per_cycle=int(os.getenv('MAX_LIVE_ORDERS_PER_CYCLE', 3)),
            max_exposure=float(os.getenv('MAX_EXPOSURE', 0.75)),
            max_position=float(os.getenv('MAX_POSITION', 0.20)),
            max_drawdown=float(os.getenv('MAX_DRAWDOWN', 0.15)),
            min_bet_size=float(os.getenv('MIN_BET_SIZE', 0.02)),
            category_limits={
                'Crypto': 0.30,
                'Politics': 0.25,
                'Sports': 0.20,
                'Science': 0.15,
                'default': 0.25
            }
        )

def load_settings():
    load_dotenv(override=True)
    return Settings.from_env()
