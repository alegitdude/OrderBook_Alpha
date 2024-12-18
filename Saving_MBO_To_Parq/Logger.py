# logger.py
import logging
from pathlib import Path
from typing import Optional

class MarketDataLogger:
    """
    Centralized logging for market data processing components
    Handles log creation, formatting, and management
    """
    
    def __init__(
        self,
        name: str,
        base_dir: Path,
        log_level: int = logging.INFO,
        log_to_console: bool = True,
        log_to_file: bool = True
    ):
        self.name = name
        self.log_dir = base_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Create formatters
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add handlers
        if log_to_console:
            self._add_console_handler()
        
        if log_to_file:
            self._add_file_handler()
    
    def _add_console_handler(self):
        """Add console output handler"""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add file output handler"""
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger