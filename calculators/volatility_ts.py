from collections import deque
from typing import List
import numpy as np
import pandas as pd
from calculators.timeseries_base import TimeSeriesFeatureCalculator


class VolatilityTimeSeries(TimeSeriesFeatureCalculator):
    """Price volatility time series calculator"""
    
    def _initialize_state(self):
        self.price_history = deque(maxlen=20)
    
    def _compute_value(self, messages: List[pd.Series]) -> float:
        if not messages:
            return 0.0
            
        # Use most recent price
        current_price = messages[-1]['mid_price']
        self.price_history.append(current_price)
        
        if len(self.price_history) < 3:
            return 0.0
        
        # Calculate returns
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate realized volatility
        realized_vol = np.std(returns)
        
        # Calculate directional volatility skew
        up_returns = returns[returns > 0]
        down_returns = returns[returns < 0]
        
        up_vol = np.std(up_returns) if len(up_returns) > 0 else 0
        down_vol = np.std(down_returns) if len(down_returns) > 0 else 0
        
        # Return volatility skew
        if down_vol == 0:
            return up_vol
        return up_vol / down_vol - 1 if down_vol != 0 else realized_vol