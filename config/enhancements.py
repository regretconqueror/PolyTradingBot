"""
Configuration enhancements for PolyTradingBot
Provides utilities for parameter tuning, validation, and configuration management
"""
import os
import sys
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

# Add the current directory to the path so we can import settings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import Settings

@dataclass
class ParameterRange:
    """Defines a valid range for a parameter"""
    min_val: float
    max_val: float
    step: float = 0.01
    description: str = ""

@dataclass
class TuningRecommendation:
    """Provides tuning recommendations based on market conditions"""
    parameter: str
    current_value: float
    recommended_value: float
    reason: str
    confidence: float  # 0.0 to 1.0

class ConfigEnhancer:
    """Enhanced configuration management with tuning capabilities"""

    # Define valid ranges for key parameters
    PARAMETER_RANGES = {
        'max_exposure': ParameterRange(0.1, 0.9, 0.05, "Maximum total exposure as fraction of capital"),
        'max_position': ParameterRange(0.05, 0.5, 0.05, "Maximum single position as fraction of capital"),
        'max_drawdown': ParameterRange(0.05, 0.3, 0.05, "Maximum allowed drawdown before stopping"),
        'min_bet_size': ParameterRange(0.01, 0.1, 0.01, "Minimum bet size as fraction of capital"),
        'capital': ParameterRange(1000, 1000000, 1000, "Trading capital in USDC"),
    }

    # Market condition based tuning rules
    TUNING_RULES = {
        'high_volatility': {
            'condition': lambda stats: stats.get('avg_volatility', 0) > 0.3,
            'adjustments': [
                ('max_position', -0.05, "Reduce position size in volatile markets"),
                ('max_drawdown', -0.05, "Tighten drawdown limits in volatile markets"),
            ]
        },
        'low_liquidity': {
            'condition': lambda stats: stats.get('avg_liquidity', 100000) < 50000,
            'adjustments': [
                ('max_position', -0.03, "Reduce position size in low liquidity markets"),
                ('min_bet_size', -0.01, "Reduce minimum bet size for low liquidity"),
            ]
        },
        'high_opportunity': {
            'condition': lambda stats: stats.get('avg_edge', 0) > 0.15,
            'adjustments': [
                ('max_exposure', 0.1, "Increase exposure when high-quality opportunities exist"),
                ('max_position', 0.05, "Increase position size for high-edge opportunities"),
            ]
        }
    }

    @classmethod
    def validate_settings(cls, settings: Settings) -> List[str]:
        """Validate settings and return list of issues"""
        issues = []

        for param_name, param_range in cls.PARAMETER_RANGES.items():
            if hasattr(settings, param_name):
                value = getattr(settings, param_name)
                if not (param_range.min_val <= value <= param_range.max_val):
                    issues.append(
                        f"{param_name}: {value} is outside valid range "
                        f"[{param_range.min_val}, {param_range.max_val}]"
                    )

        # Check logical constraints
        if settings.max_position > settings.max_exposure:
            issues.append(
                f"max_position ({settings.max_position}) cannot exceed "
                f"max_exposure ({settings.max_exposure})"
            )

        if settings.min_bet_size > settings.max_position:
            issues.append(
                f"min_bet_size ({settings.min_bet_size}) cannot exceed "
                f"max_position ({settings.max_position})"
            )

        return issues

    @classmethod
    def get_tuning_recommendations(cls, settings: Settings, market_stats: Dict[str, float]) -> List[TuningRecommendation]:
        """Get tuning recommendations based on market statistics"""
        recommendations = []

        for rule_name, rule in cls.TUNING_RULES.items():
            if rule['condition'](market_stats):
                for param, adjustment, reason in rule['adjustments']:
                    if hasattr(settings, param) and param in cls.PARAMETER_RANGES:
                        current_value = getattr(settings, param)
                        param_range = cls.PARAMETER_RANGES[param]

                        # Apply adjustment
                        recommended_value = current_value + adjustment

                        # Clamp to valid range
                        recommended_value = max(
                            param_range.min_val,
                            min(param_range.max_val, recommended_value)
                        )

                        # Only recommend if there's a meaningful change
                        if abs(recommended_value - current_value) >= param_range.step:
                            confidence = 0.8  # Base confidence
                            if rule_name == 'high_opportunity':
                                confidence = 0.9
                            elif rule_name == 'high_volatility':
                                confidence = 0.85

                            recommendations.append(TuningRecommendation(
                                parameter=param,
                                current_value=current_value,
                                recommended_value=recommended_value,
                                reason=reason,
                                confidence=confidence
                            ))

        return recommendations

    @classmethod
    def create_optimized_config(cls, base_settings: Settings, market_stats: Dict[str, float]) -> Settings:
        """Create an optimized configuration based on market stats"""
        # Start with current settings
        optimized = Settings.from_env()  # Fresh copy from env

        # Apply validated tuning recommendations
        recommendations = cls.get_tuning_recommendations(base_settings, market_stats)

        for rec in recommendations:
            setattr(optimized, rec.parameter, rec.recommended_value)

        # Validate the result
        issues = cls.validate_settings(optimized)
        if issues:
            print(f"Warning: Optimized config has issues: {issues}")
            # Fall back to base settings if optimization creates invalid config
            optimized = base_settings

        return optimized

    @classmethod
    def print_config_summary(cls, settings: Settings):
        """Print a formatted summary of current configuration"""
        print("\n" + "="*60)
        print("POLYTRADINGBOT CONFIGURATION SUMMARY")
        print("="*60)

        print(f"Trading Capital: ${settings.capital:,.2f} USDC")
        print(f"Paper Trading Mode: {settings.paper_mode}")
        print()

        print("Risk Management:")
        print(f"  Max Total Exposure: {settings.max_exposure:.1%}")
        print(f"  Max Single Position: {settings.max_position:.1%}")
        print(f"  Max Drawdown: {settings.max_drawdown:.1%}")
        print(f"  Min Bet Size: {settings.min_bet_size:.1%}")
        print()

        print("Category Limits:")
        for category, limit in settings.category_limits.items():
            print(f"  {category}: {limit:.1%}")
        print()

        # Show parameter validity
        issues = cls.validate_settings(settings)
        if issues:
            print("*** CONFIGURATION ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("+++ Configuration is valid")

        print("="*60)

    @classmethod
    def save_config_template(cls, filepath: str = ".env.optimized"):
        """Save a configuration template with current environment values"""
        template = f"""# PolyTradingBot Optimized Configuration Template
# Copy this to .env and adjust as needed

# Polymarket API Credentials
# Get these from your Polymarket account settings
POLYMARKET_API_KEY=your_api_key_here
POLYMARKET_API_SECRET=your_api_secret_here
POLYMARKET_PRIVATE_KEY=your_private_key_here

# Trading Configuration
CAPITAL=10000
PAPER_MODE=true

# Risk Management Parameters
MAX_EXPOSURE=0.75
MAX_POSITION=0.20
MAX_DRAWDOWN=0.15
MIN_BET_SIZE=0.02

# Category Limits (optional - will use defaults if not set)
# Format: CATEGORY=limit (0.0-1.0)
# CRYPTO=0.30
# POLITICS=0.25
# SPORTS=0.20
# SCIENCE=0.15
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(template)
        print(f"Configuration template saved to {filepath}")

def load_and_enhance_config() -> Settings:
    """Load configuration and apply enhancements"""
    settings = Settings.from_env()

    # Validate and report issues
    issues = ConfigEnhancer.validate_settings(settings)
    if issues:
        print("⚠️  Configuration Issues Detected:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    return settings

if __name__ == "__main__":
    # Demo usage
    settings = Settings.from_env()
    ConfigEnhancer.print_config_summary(settings)

    # Example market stats for tuning demo
    demo_stats = {
        'avg_volatility': 0.25,
        'avg_liquidity': 75000,
        'avg_edge': 0.08
    }

    print("\n[TUNING] Example Tuning Recommendations:")
    recommendations = ConfigEnhancer.get_tuning_recommendations(settings, demo_stats)
    for rec in recommendations:
        print(f"  {rec.parameter}: {rec.current_value:.3f} → {rec.recommended_value:.3f}")
        print(f"    Reason: {rec.reason} (confidence: {rec.confidence:.1%})")