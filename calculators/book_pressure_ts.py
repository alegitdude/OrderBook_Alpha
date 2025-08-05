from typing import List
import pandas as pd
from calculators.timeseries_base import TimeSeriesFeatureCalculator


class BookPressureTimeSeries(TimeSeriesFeatureCalculator):
    """Book pressure time series calculator"""
    
    def _initialize_state(self):
        pass
    
    def _compute_value(self, messages: List[pd.Series]) -> float:
        # Use most recent message for book state
        if not messages:
            return 0.0
            
        message = messages[-1]
        book_state = message['book_state']
        
        # Calculate weighted pressure at different levels
        bid_pressure = 0
        ask_pressure = 0
        
        # Look at top 5 levels with decreasing weights
        for i, level in enumerate(book_state['bid_side'][:5]):
            weight = 1 / (i + 1)  # Higher weight for closer levels
            bid_pressure += level['volume'] * weight
        
        for i, level in enumerate(book_state['ask_side'][:5]):
            weight = 1 / (i + 1)
            ask_pressure += level['volume'] * weight
            
        total_pressure = bid_pressure + ask_pressure
        return (bid_pressure - ask_pressure) / total_pressure if total_pressure > 0 else 0