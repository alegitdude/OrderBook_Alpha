from typing import List
import pandas as pd
from calculators.timeseries_base import TimeSeriesFeatureCalculator


class TradeIntensityTimeSeries(TimeSeriesFeatureCalculator):
    """Trade intensity and aggression time series calculator"""
    
    def _initialize_state(self):
        pass
    
    def _compute_value(self, messages: List[pd.Series]) -> float:
        if not messages:
            return 0.0
            
        aggressive_volume = 0
        total_volume = 0
        large_trades = 0
        total_trades = 0
        
        for msg in messages:
            if msg['is_trade']:
                total_volume += msg['size']
                total_trades += 1
                
                # Track aggressive trades
                if 'event_subtype' in msg and msg['event_subtype'].startswith('aggressive'):
                    aggressive_volume += msg['size']
                
                # Track large trades (above threshold)
                if msg['size'] > 100:  # Configurable threshold
                    large_trades += 1
        
        if total_trades == 0:
            return 0.0
        
        # Calculate multiple intensity metrics
        aggression_ratio = aggressive_volume / total_volume if total_volume > 0 else 0
        large_trade_ratio = large_trades / total_trades
        trade_rate = min(total_trades / 10, 1)  # Normalized trade rate
        
        # Combine into single intensity score
        return (aggression_ratio + large_trade_ratio + trade_rate) / 3