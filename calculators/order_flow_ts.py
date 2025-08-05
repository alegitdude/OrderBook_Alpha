from typing import List
import pandas as pd
from calculators.timeseries_base import TimeSeriesFeatureCalculator

class OrderFlowTimeSeries(TimeSeriesFeatureCalculator):
    """Order flow imbalance time series calculator"""
    
    def _initialize_state(self):
        pass
    
    def _compute_value(self, messages: List[pd.Series]) -> float:
        buy_volume = 0
        sell_volume = 0
        buy_trades = 0
        sell_trades = 0
        
        for msg in messages:
            if msg['is_trade']:
                if msg['side'] == 'buy':
                    buy_volume += msg['size']
                    buy_trades += 1
                else:
                    sell_volume += msg['size']
                    sell_trades += 1
        
        total_volume = buy_volume + sell_volume
        total_trades = buy_trades + sell_trades
        
        # Volume imbalance
        volume_imbalance = ((buy_volume - sell_volume) / 
                           total_volume if total_volume > 0 else 0)
        
        # Trade count imbalance
        trade_imbalance = ((buy_trades - sell_trades) / 
                          total_trades if total_trades > 0 else 0)
        
        # Combine metrics
        return (volume_imbalance + trade_imbalance) / 2