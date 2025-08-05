from datetime import timedelta
from typing import Dict, Optional
import numpy as np
import pandas as pd
from calculators.book_pressure_ts import BookPressureTimeSeries
from calculators.momentum_ts import MomentumTimeSeries
from calculators.order_flow_ts import OrderFlowTimeSeries
from calculators.trade_intensity_ts import TradeIntensityTimeSeries
from calculators.volatility_ts import VolatilityTimeSeries
from core.types import TimeSeriesConfig, LookbackType
from utils.logging import OrderBookLogger


class SequenceFeatureCollector:
    """Collects time series features before sequence events"""
    
    def __init__(
        self,
        logger: OrderBookLogger,
        configs: Dict[str, TimeSeriesConfig]
    ):
        self.logger = logger
        self.calculators = {
            'order_flow': OrderFlowTimeSeries(
                configs['order_flow'], logger
            ),
            'book_pressure': BookPressureTimeSeries(
                configs['book_pressure'], logger
            ),
            'trade_intensity': TradeIntensityTimeSeries(
                configs['trade_intensity'], logger
            ),
            'momentum': MomentumTimeSeries(
                configs['momentum'], logger
            ),
            'volatility': VolatilityTimeSeries(
                configs['volatility'], logger
            )
        }
    
    def collect_features(
        self,
        data: pd.DataFrame,
        sequence_start_idx: int
    ) -> Dict[str, np.ndarray]:
        """Collect all feature time series before a sequence"""
        # Reset all calculators
        for calc in self.calculators.values():
            calc.reset()
        
        # Determine lookback range based on calculator configurations
        lookback_range = self._calculate_lookback_range(data, sequence_start_idx)
        
        if lookback_range is None:
            return {}
        
        start_idx, end_idx = lookback_range
        
        # Process messages
        features = {}
        for idx in range(start_idx, end_idx):
            if idx >= len(data):
                break
                
            message = data.iloc[idx]
            
            # Update each calculator
            for name, calc in self.calculators.items():
                series = calc.update(message)
                if series is not None:
                    features[name] = series
        
        return features
    
    def _calculate_lookback_range(
        self,
        data: pd.DataFrame,
        sequence_start_idx: int
    ) -> Optional[tuple]:
        """Calculate the range of data to process based on calculator configurations"""
        
        # Check if we have any time-based calculators
        time_based_configs = [
            calc.config for calc in self.calculators.values()
            if calc.config.lookback_type == LookbackType.TIME
        ]
        
        # Check if we have any message-based calculators
        message_based_configs = [
            calc.config for calc in self.calculators.values()
            if calc.config.lookback_type == LookbackType.MESSAGES
        ]
        
        start_indices = []
        
        # Handle time-based calculators
        if time_based_configs:
            max_history_ms = max(config.history_ms for config in time_based_configs)
            sequence_start_time = data.iloc[sequence_start_idx]['ts_event']
            start_time = sequence_start_time - timedelta(milliseconds=max_history_ms)
            
            # Find starting index for time-based
            time_start_indices = data[data['ts_event'] >= start_time].index
            if len(time_start_indices) > 0:
                start_indices.append(time_start_indices[0])
        
        # Handle message-based calculators
        if message_based_configs:
            max_history_messages = max(config.history_messages for config in message_based_configs)
            message_start_idx = max(0, sequence_start_idx - max_history_messages)
            start_indices.append(message_start_idx)
        
        if not start_indices:
            return None
        
        # Use the earliest start index to ensure we have enough data for all calculators
        start_idx = min(start_indices)
        return (start_idx, sequence_start_idx)