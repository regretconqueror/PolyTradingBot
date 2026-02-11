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
    private_key: str
    
    # Trading
    capital: float
    paper_mode: bool
    
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
            private_key=os.getenv('POLYMARKET_PRIVATE_KEY', ''),
            capital=float(os.getenv('CAPITAL', 10000)),
            paper_mode=os.getenv('PAPER_MODE', 'true').lower() == 'true',
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
    return Settings.from_env()