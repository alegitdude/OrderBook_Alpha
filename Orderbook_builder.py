import databento as db
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import os
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OrderBookFeatureRecorder:
    """
    Comprehensive orderbook feature recording system
    Optimized for high-performance data handling
    """
    instrument: str
    output_dir: str = './orderbook_features/'
    
    # Feature extraction configuration
    depth_levels: int = 10
    feature_window: int = 50  # Number of snapshots per feature record
    
    # Internal state tracking
    _current_features: List[List[float]] = field(default_factory=list)
    _current_book: Dict[str, Dict[float, float]] = field(default_factory=lambda: {
        'buy_orders': {},
        'sell_orders': {}
    })
    _trade_data: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_databento_message(self, msg):
        """
        Process each Databento message to update orderbook
        
        Supports:
        - Book updates (MBO messages)
        - Trade messages
        - Incremental book updates
        """
        if msg.record_type == 'mbo':
            self._update_orderbook_from_mbo(msg)
        elif msg.record_type == 'trade':
            self._record_trade(msg)
        
        # Periodically extract features
        if len(self._current_features) >= self.feature_window:
            self._save_feature_batch()
    
    def _update_orderbook_from_mbo(self, msg):
        """
        Update orderbook based on Market By Order (MBO) message
        
        Handles:
        - New orders
        - Order modifications
        - Order deletions
        """
        price = msg.price
        side = 'buy_orders' if msg.side == 'bid' else 'sell_orders'
        
        # Handle different MBO actions
        if msg.action == 'add':
            self._current_book[side][price] = msg.size
        elif msg.action == 'update':
            self._current_book[side][price] = msg.size
        elif msg.action == 'delete':
            self._current_book[side].pop(price, None)
        
        # Extract features if window is full
        if len(self._current_features) < self.feature_window:
            features = self._extract_orderbook_features()
            self._current_features.append(features)
    
    def _record_trade(self, trade_msg):
        """
        Record trade information for microstructure analysis
        """
        self._trade_data.append({
            'price': trade_msg.price,
            'size': trade_msg.size,
            'side': 'buy' if trade_msg.aggressor_side == 'bid' else 'sell',
            'timestamp': trade_msg.ts_event
        })
    
    def _extract_orderbook_features(self):
        """
        Extract comprehensive orderbook features
        
        Reuses feature extraction logic from previous implementation
        """
        # Sort and limit to top depth levels
        buy_orders = dict(sorted(self._current_book['buy_orders'].items(), reverse=True)[:self.depth_levels])
        sell_orders = dict(sorted(self._current_book['sell_orders'].items())[:self.depth_levels])
        
        # Combine feature extraction methods
        features = (
            # Existing feature extraction methods
            self._calculate_weighted_imbalance(buy_orders, sell_orders) +
            self._calculate_volume_profile(buy_orders, sell_orders) +
            self._calculate_market_microstructure(buy_orders, sell_orders, self._trade_data) +
            self._calculate_liquidity_metrics(buy_orders, sell_orders)
        )
        
        return features
    
    def _save_feature_batch(self):
        """
        Save extracted features to Parquet for efficient storage
        """
        # Convert features to Polars DataFrame
        df = pl.DataFrame(
            self._current_features,
            schema=[
                f'feature_{i}' for i in range(len(self._current_features[0]))
            ]
        )
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{self.output_dir}/{self.instrument}_features_{timestamp}.parquet'
        
        # Write to Parquet
        df.write_parquet(filename)
        
        # Reset feature collection
        self._current_features = []
        self._trade_data = []
    
    # Reuse feature calculation methods from previous implementation
    def _calculate_weighted_imbalance(self, buy_orders, sell_orders):
        # Implementation from previous example
        pass
    
    def _calculate_volume_profile(self, buy_orders, sell_orders):
        # Implementation from previous example
        pass
    
    def _calculate_market_microstructure(self, buy_orders, sell_orders, trade_data):
        # Implementation from previous example
        pass
    
    def _calculate_liquidity_metrics(self, buy_orders, sell_orders):
        # Implementation from previous example
        pass

class DatabentoBatchProcessor:
    """
    High-performance batch processor for Databento data
    """
    def __init__(self, api_key, dataset='glbx.md', data_range=None):
        """
        Initialize Databento client and data retrieval
        
        Parameters:
        - api_key: Databento API key
        - dataset: Market data dataset (default: GLBX market data)
        - data_range: Optional date/time range for data retrieval
        """
        self.client = db.Historical(key=api_key)
        self.dataset = dataset
        self.data_range = data_range
    
    def process_instrument(self, instrument, recorder):
        """
        Process data for a specific instrument
        
        Parameters:
        - instrument: Futures instrument symbol
        - recorder: OrderBookFeatureRecorder instance
        """
        # Retrieve Databento data
        data_request = self.client.retrieve(
            dataset=self.dataset,
            symbols=[instrument],
            record_type=['mbo', 'trade'],
            start=self.data_range[0] if self.data_range else None,
            end=self.data_range[1] if self.data_range else None
        )
        
        # Process each message
        for msg in data_request:
            recorder.process_databento_message(msg)

# Example Usage
def main():
    # Configuration
    API_KEY = 'your_databento_api_key'
    INSTRUMENT = 'ES.CM'  # E.g., E-mini S&P 500 futures
    
    # Initialize processors
    batch_processor = DatabentoBatchProcessor(
        api_key=API_KEY,
        data_range=(
            datetime(2024, 1, 1),  # Start date
            datetime(2024, 1, 2)   # End date
        )
    )
    
    # Create feature recorder
    feature_recorder = OrderBookFeatureRecorder(
        instrument=INSTRUMENT,
        output_dir='./futures_orderbook_features/'
    )
    
    # Process data
    batch_processor.process_instrument(INSTRUMENT, feature_recorder)

if __name__ == "__main__":
    main()