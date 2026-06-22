"""
Alert management for PolyTradingBot
Handles logging and storing alerts for monitoring and dashboard display
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    """Manages alerts for the trading bot"""

    def __init__(self, alert_file: str = "alerts.json"):
        self.alert_file = alert_file
        # Ensure alert file exists
        if not os.path.exists(self.alert_file):
            with open(self.alert_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def _load_alerts(self) -> List[Dict]:
        """Load existing alerts from file"""
        try:
            with open(self.alert_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_alerts(self, alerts: List[Dict]):
        """Save alerts to file"""
        try:
            with open(self.alert_file, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def add_alert(self, level: str, message: str, source: str = "bot",
                  metadata: Optional[Dict] = None) -> Dict:
        """
        Add an alert

        Args:
            level: Alert level (info, warning, error, critical)
            message: Alert message
            source: Source of the alert (default: "bot")
            metadata: Additional data to store with the alert

        Returns:
            The alert dictionary that was added
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level.lower(),
            "source": source,
            "message": message,
            "metadata": metadata or {}
        }

        # Log the alert
        if level.lower() == "critical":
            logger.critical(f"[{source}] {message}")
        elif level.lower() == "error":
            logger.error(f"[{source}] {message}")
        elif level.lower() == "warning":
            logger.warning(f"[{source}] {message}")
        else:
            logger.info(f"[{source}] {message}")

        # Add to alerts file
        alerts = self._load_alerts()
        alerts.append(alert)
        # Keep only last 100 alerts to prevent file from growing too large
        if len(alerts) > 100:
            alerts = alerts[-100:]
        self._save_alerts(alerts)

        return alert

    def info(self, message: str, source: str = "bot", metadata: Optional[Dict] = None):
        """Add an info alert"""
        return self.add_alert("info", message, source, metadata)

    def warning(self, message: str, source: str = "bot", metadata: Optional[Dict] = None):
        """Add a warning alert"""
        return self.add_alert("warning", message, source, metadata)

    def error(self, message: str, source: str = "bot", metadata: Optional[Dict] = None):
        """Add an error alert"""
        return self.add_alert("error", message, source, metadata)

    def critical(self, message: str, source: str = "bot", metadata: Optional[Dict] = None):
        """Add a critical alert"""
        return self.add_alert("critical", message, source, metadata)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        alerts = self._load_alerts()
        return alerts[-limit:] if alerts else []

# Global alert manager instance
alert_manager = AlertManager()