from collections import deque
from typing import List
import numpy as np
import pandas as pd
from calculators.timeseries_base import TimeSeriesFeatureCalculator

class MomentumTimeSeries(TimeSeriesFeatureCalculator):
    """Price momentum time series calculator"""
    
    def _initialize_state(self):
        self.price_history = deque(maxlen=10)
    
    def _compute_value(self, messages: List[pd.Series]) -> float:
        if not messages:
            return 0.0
            
        # Use most recent price
        current_price = messages[-1]['mid_price']
        self.price_history.append(current_price)
        
        if len(self.price_history) < 3:
            return 0.0
        
        # Calculate price changes
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate directional consistency
        if len(returns) > 1:
            direction_changes = (returns[1:] * returns[:-1] < 0).sum()
            consistency = 1 - (direction_changes / (len(returns) - 1))
        else:
            consistency = 1.0
        
        # Calculate momentum strength
        momentum_strength = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        
        return consistency * np.sign(momentum_strength)