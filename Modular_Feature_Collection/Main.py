import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable, Union, Optional
from enum import Enum
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
import scipy.stats as stats

class LookbackType(Enum):
    TIME = "time"
    TRADES = "trades"
    MIDPRICE_CHANGES = "midprice_changes"
    MESSAGES = "messages"

@dataclass
class FeatureConfig:
    """Configuration for feature calculation"""
    lookback_type: LookbackType
    lookback_value: Union[int, float]  # Number of trades/messages or time in seconds
    feature_name: str
    aggregation_method: str = "last"  # last, mean, std, min, max, etc.

@dataclass
class PredictionTarget:
    """Configuration for prediction target generation"""
    target_type: str  # mid_price_direction, mid_price_return, etc.
    horizon_type: LookbackType
    horizon_value: Union[int, float]
    
class OrderBookFeatureEngine:
    def __init__(self):
        self.feature_calculators = {
            # Queue/Book Pressure Metrics
            'queue_position': self._calculate_queue_position,
            'total_orders': self._calculate_total_orders,
            'cumulative_volume': self._calculate_cumulative_volume,
            'relative_queue_position': self._calculate_relative_queue_position,
            
            # Book Velocity Metrics
            'order_count_delta': self._calculate_order_count_delta,
            'volume_delta': self._calculate_volume_delta,
            'price_level_velocity': self._calculate_price_level_velocity,
            'book_update_rate': self._calculate_book_update_rate,
            
            # Order Flow Pattern Metrics
            'sweep_detection': self._detect_sweeps,
            'iceberg_detection': self._detect_icebergs,
            'order_flow_imbalance': self._calculate_order_flow_imbalance,
            'replenishment_rate': self._calculate_replenishment_rate,
            
            # Price Level Activity Metrics
            'price_level_turnover': self._calculate_price_level_turnover,
            'price_level_lifetime': self._calculate_price_level_lifetime,
            'price_level_stability': self._calculate_price_level_stability,
            'relative_level_dominance': self._calculate_relative_level_dominance,
            
            # Order Clustering Metrics
            'cluster_size': self._calculate_cluster_size,
            'cluster_imbalance': self._calculate_cluster_imbalance,
            'time_between_clusters': self._calculate_time_between_clusters,
            'cluster_impact': self._calculate_cluster_impact,
            
            # Market Maker Activity Metrics
            'maker_volumes': self._calculate_maker_volumes,
            'taker_volumes': self._calculate_taker_volumes,
            'cancellation_rates': self._calculate_cancellation_rates,
            
            # Inter-level Relationship Metrics
            'level_correlation': self._calculate_level_correlation,
            'level_gaps': self._calculate_level_gaps,
            'volume_concentration': self._calculate_volume_concentration,
            'level_symmetry': self._calculate_level_symmetry,
            
            # Trade Impact Metrics
            'trade_price_impact': self._calculate_trade_price_impact,
            'trade_book_impact': self._calculate_trade_book_impact,
            'trade_queue_impact': self._calculate_trade_queue_impact,
            'post_trade_recovery': self._calculate_post_trade_recovery,
            
            # Order Resting Time Metrics
            'resting_time_stats': self._calculate_resting_time_stats,
            'quick_cancel_ratio': self._calculate_quick_cancel_ratio
        }
        
    def _get_lookback_window(self, data: pd.DataFrame, 
                            current_idx: int,
                            config: FeatureConfig) -> pd.DataFrame:
        """Get data window based on lookback type and value"""
        if config.lookback_type == LookbackType.TRADES:
            trade_indices = data[data['is_trade']].index
            current_trade_idx = trade_indices.searchsorted(current_idx)
            start_trade_idx = max(0, current_trade_idx - config.lookback_value)
            start_idx = trade_indices[start_trade_idx]
            return data.loc[start_idx:current_idx]
            
        elif config.lookback_type == LookbackType.TIME:
            current_time = data.loc[current_idx, 'ts_event']
            start_time = current_time - pd.Timedelta(seconds=config.lookback_value)
            return data[
                (data['ts_event'] >= start_time) & 
                (data['ts_event'] <= current_time)
            ]
            
        elif config.lookback_type == LookbackType.MIDPRICE_CHANGES:
            price_changes = data['mid_price'] != data['mid_price'].shift(1)
            change_indices = data[price_changes].index
            current_change_idx = change_indices.searchsorted(current_idx)
            start_change_idx = max(0, current_change_idx - config.lookback_value)
            start_idx = change_indices[start_change_idx]
            return data.loc[start_idx:current_idx]
            
        else:  # MESSAGES
            start_idx = max(0, current_idx - config.lookback_value)
            return data.loc[start_idx:current_idx]

    def _calculate_queue_position(self, window: pd.DataFrame) -> float:
        """Calculate queue position metrics"""
        # Implementation details here
        pass
    
    def _calculate_total_orders(self, window: pd.DataFrame) -> float:
        """Calculate total orders at price levels"""
        # Implementation details here
        pass
    
    # ... Additional feature calculation methods ...
    
    def _generate_prediction_target(self, data: pd.DataFrame,
                                  current_idx: int,
                                  target_config: PredictionTarget) -> float:
        """Generate prediction target based on configuration"""
        if target_config.target_type == 'mid_price_direction':
            future_window = self._get_lookback_window(
                data,
                min(current_idx + target_config.horizon_value, len(data) - 1),
                FeatureConfig(
                    lookback_type=target_config.horizon_type,
                    lookback_value=target_config.horizon_value,
                    feature_name='target'
                )
            )
            current_mid = data.loc[current_idx, 'mid_price']
            future_mid = future_window['mid_price'].iloc[-1]
            return np.sign(future_mid - current_mid)
            
        # Add other target types as needed
        
    def calculate_features(self, data: pd.DataFrame,
                          feature_configs: List[FeatureConfig],
                          target_config: Optional[PredictionTarget] = None,
                          sample_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """Calculate features and optionally prediction targets for specified indices"""
        if sample_indices is None:
            sample_indices = data.index
            
        features = defaultdict(list)
        
        for idx in sample_indices:
            for config in feature_configs:
                window = self._get_lookback_window(data, idx, config)
                feature_value = self.feature_calculators[config.feature_name](window)
                features[config.feature_name].append(feature_value)
                
            if target_config:
                target = self._generate_prediction_target(data, idx, target_config)
                features['target'].append(target)
                
        return pd.DataFrame(features, index=sample_indices)

class DataLoader:
    """Handle loading and preprocessing of parquet files"""
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        
    def load_batch(self, batch_size: int = 100000) -> pd.DataFrame:
        """Load and yield batches of data"""
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                yield batch.to_pandas()

def main():
    # Example usage
    file_paths = ['path/to/orderbook/data/*.parquet']
    data_loader = DataLoader(file_paths)
    feature_engine = OrderBookFeatureEngine()
    
    # Configure features
    feature_configs = [
        FeatureConfig(
            lookback_type=LookbackType.TRADES,
            lookback_value=100,
            feature_name='order_flow_imbalance'
        ),
        FeatureConfig(
            lookback_type=LookbackType.TIME,
            lookback_value=1.0,  # 1 second
            feature_name='volume_delta'
        ),
        # Add more feature configurations
    ]
    
    # Configure prediction target
    target_config = PredictionTarget(
        target_type='mid_price_direction',
        horizon_type=LookbackType.TRADES,
        horizon_value=100
    )
    
    # Process data in batches
    for batch in data_loader.load_batch():
        # Sample indices (e.g., every 100th message)
        sample_indices = batch.index[::100]
        
        # Calculate features and targets
        features_df = feature_engine.calculate_features(
            batch,
            feature_configs,
            target_config,
            sample_indices
        )
        
        # Store or process features_df as needed

if __name__ == "__main__":
    main()