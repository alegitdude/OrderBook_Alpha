from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import sys
import time
from contextlib import contextmanager
from typing import Generator

@dataclass
class LogConfig:
    """Configuration for logging setup"""
    log_level: str = "INFO"
    log_dir: Optional[Path] = None
    log_to_file: bool = True
    log_to_console: bool = True
    performance_tracking: bool = True

class OrderBookLogger:
    """Centralized logging for sequence detection system"""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.logger = logging.getLogger('price_sequence_detector')
        self.performance_logs = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Initialize logging configuration"""
        self.logger.setLevel(getattr(logging, self.config.log_level))
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file and self.config.log_dir:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_handler = logging.FileHandler(
                self.config.log_dir / f'sequence_detection_{timestamp}.log'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    @contextmanager
    def track_performance(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for tracking operation performance"""
        if not self.config.performance_tracking:
            yield
            return
            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            if operation_name not in self.performance_logs:
                self.performance_logs[operation_name] = []
            self.performance_logs[operation_name].append(duration)
    
    def log_milestone(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a processing milestone with optional data"""
        if data:
            self.logger.info(f"{message} - {json.dumps(data, default=str)}")
        else:
            self.logger.info(message)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context"""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str, context: str = ""):
        """Log a warning with context"""
        self.logger.warning(f"{context}: {message}")
    
    def save_performance_summary(self, output_path: Path):
        """Save performance tracking summary to file"""
        if not self.config.performance_tracking:
            return
        
        summary = {}
        for operation, durations in self.performance_logs.items():
            summary[operation] = {
                'count': len(durations),
                'total_time': sum(durations),
                'average_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations)
            }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Performance summary saved to {output_path}")