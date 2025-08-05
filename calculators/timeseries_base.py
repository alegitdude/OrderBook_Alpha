from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta

from ..core.types import TimeSeriesConfig, LookbackType
from ..utils.logging import OrderBookLogger

class TimeSeriesFeatureCalculator(ABC):
    """Base class for time-series feature calculators supporting both time and message modes"""
    
    def __init__(
        self,
        config: TimeSeriesConfig,
        logger: OrderBookLogger
    ):
        self.config = config
        self.logger = logger
        self.buffer = deque(maxlen=self.config.num_points)
        
        # Time-based state
        self.last_update = None
        
        # Message-based state
        self.message_counter = 0
        self.message_accumulator = []
        
        self._initialize_state()
    
    @abstractmethod
    def _initialize_state(self):
        """Initialize calculator-specific state"""
        pass
    
    @abstractmethod
    def _compute_value(self, messages: List[pd.Series]) -> float:
        """Compute single value from a set of messages"""
        pass
    
    def update(self, message: pd.Series) -> Optional[np.ndarray]:
        """Update time series with new message"""
        if self.config.lookback_type == LookbackType.TIME:
            return self._update_time_based(message)
        else:
            return self._update_message_based(message)
    
    def _update_time_based(self, message: pd.Series) -> Optional[np.ndarray]:
        """Update using time-based intervals"""
        current_time = message['ts_event']
        
        # Initialize if first message
        if self.last_update is None:
            self.last_update = current_time
            return None
        
        # Calculate how many intervals have passed
        time_diff = (current_time - self.last_update).total_seconds() * 1000
        intervals_passed = int(time_diff // self.config.granularity_ms)
        
        if intervals_passed > 0:
            # Fill any missed intervals with the last value
            last_value = self.buffer[-1] if self.buffer else 0
            for _ in range(intervals_passed - 1):
                self.buffer.append(last_value)
            
            # Compute new value
            self.buffer.append(self._compute_value([message]))
            self.last_update = current_time
        
        # Return full time series if buffer is full
        if len(self.buffer) == self.config.num_points:
            return np.array(self.buffer)
        
        return None
    
    def _update_message_based(self, message: pd.Series) -> Optional[np.ndarray]:
        """Update using message-based intervals"""
        self.message_counter += 1
        self.message_accumulator.append(message)
        
        # Check if we've accumulated enough messages for this interval
        if self.message_counter >= self.config.granularity_messages:
            # Compute value from accumulated messages
            value = self._compute_value(self.message_accumulator)
            self.buffer.append(value)
            
            # Reset accumulator
            self.message_counter = 0
            self.message_accumulator = []
        
        # Return full time series if buffer is full
        if len(self.buffer) == self.config.num_points:
            return np.array(self.buffer)
        
        return None
    
    def get_series(self) -> np.ndarray:
        """Get current time series"""
        return np.array(self.buffer)
    
    def reset(self):
        """Reset calculator state"""
        self.buffer.clear()
        self.last_update = None
        self.message_counter = 0
        self.message_accumulator = []
        self._initialize_state()
