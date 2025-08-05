from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union, Dict, Any

class LookbackType(Enum):
    TIME = "time"
    TRADES = "trades"
    MESSAGES = "messages"

@dataclass
class TimeSeriesConfig:
    """Configuration for time-series feature calculation supporting both time and message modes"""
    feature_name: str
    lookback_type: LookbackType
    
    # Time-based configuration
    granularity_ms: Optional[int] = None     # Milliseconds per sample
    history_ms: Optional[int] = None         # Total history in milliseconds
    
    # Message-based configuration
    granularity_messages: Optional[int] = None  # Messages per sample
    history_messages: Optional[int] = None      # Total history in messages
    
    def __post_init__(self):
        if self.lookback_type == LookbackType.TIME:
            if self.granularity_ms is None or self.history_ms is None:
                raise ValueError("Time-based config requires granularity_ms and history_ms")
        elif self.lookback_type == LookbackType.MESSAGES:
            if self.granularity_messages is None or self.history_messages is None:
                raise ValueError("Message-based config requires granularity_messages and history_messages")
    
    @property
    def num_points(self) -> int:
        """Calculate number of points in time series"""
        if self.lookback_type == LookbackType.TIME:
            return self.history_ms // self.granularity_ms
        else:
            return self.history_messages // self.granularity_messages

@dataclass
class PriceSequence:
    """Represents a sequence of consecutive price moves in the same direction"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    total_ticks: float           # Total movement in ticks
    direction: int               # 1 for up, -1 for down
    start_index: int            # Message index where sequence starts
    end_index: int              # Message index where sequence ends
    num_moves: int              # Number of moves in sequence
    moves_per_second: float     # Rate of movement
    volume_during: float        # Total volume during sequence
    avg_trade_size: float       # Average trade size during sequence
    max_retrace: float         # Maximum retracement during sequence
    
    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def total_return(self) -> float:
        """Calculate total return of the sequence"""
        return (self.end_price / self.start_price) - 1
    
    @property
    def ticks_per_second(self) -> float:
        """Calculate tick movement rate"""
        return self.total_ticks / self.duration_seconds if self.duration_seconds > 0 else 0

@dataclass
class SequenceConfig:
    """Configuration for sequence identification"""
    min_moves: int              # Minimum number of consecutive moves required
    min_ticks: float           # Minimum size for each move in ticks
    max_duration_ms: int       # Maximum milliseconds for sequence
    min_volume: float          # Minimum volume during sequence
    max_retrace_ticks: float   # Maximum allowed retracement in ticks
    tick_size: float           # Size of one tick

@dataclass
class FeatureConfig:
    """Configuration for feature calculation"""
    name: str
    lookback_type: LookbackType
    lookback_value: Union[int, float]
    compute_frequency: Optional[int] = 1

@dataclass
class AnalysisConfig:
    """Configuration for sequence analysis"""
    lookback_messages: int      # Messages to analyze before sequence
    lookback_intervals: List[int]  # Different intervals to analyze
    required_calm_ticks: float  # Required price calmness in ticks
    min_confidence: float      # Minimum confidence for prediction

@dataclass
class TimeSeriesConfig:
    """Configuration for time-series feature calculation"""
    granularity_ms: int           # Base granularity in milliseconds
    history_ms: int              # How much history to maintain
    feature_name: str            # Name of this feature
    
    @property
    def num_points(self) -> int:
        """Calculate number of points in time series"""
        return self.history_ms // self.granularity_ms