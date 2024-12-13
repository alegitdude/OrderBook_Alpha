import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

@dataclass
class OrderBookSnapshotRecorder:
    """
    Records atomic orderbook data for futures market
    Focuses on raw, non-parametric data points for maximum reusability
    """
    instrument: str
    output_dir: str = './raw_orderbook_snapshots/'
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._schema = pa.schema([
            # Timestamps and identifiers
            ('ts_event', pa.timestamp('ns')),
            ('ts_recv', pa.timestamp('ns')),
            ('instrument', pa.string()),
            ('exchange', pa.string()),
            ('sequence_number', pa.int64()),
            
            # Order reference data
            ('order_id', pa.string()),
            ('original_order_id', pa.string()),
            ('priority', pa.int64()),         # Order priority/sequence at price level
            
            # Event classification
            ('event_type', pa.string()),      # 'new', 'modify', 'cancel', 'trade', 'trade_cancel'
            ('side', pa.string()),            # 'bid' or 'ask'
            
            # Order details
            ('price', pa.float64()),
            ('size', pa.float64()),
            ('old_price', pa.float64()),      # For modifications
            ('old_size', pa.float64()),       # For modifications
            ('entry_time', pa.timestamp('ns')), # When order first entered book
            
            # Trade-specific information
            ('is_trade', pa.bool_()),
            ('trade_id', pa.string()),
            ('aggressor_side', pa.string()),    # Which side initiated the trade
            ('resting_order_id', pa.string()),  # ID of the passive order
            ('aggressive_order_id', pa.string()),# ID of the aggressive order
            ('trade_size', pa.float64()),       # Size of the trade
            
            # Cancellation details
            ('is_cancel', pa.bool_()),
            ('cancel_type', pa.string()),       # 'user', 'ioc', 'timeout', etc.
            ('original_entry_time', pa.timestamp('ns')), # When cancelled order entered
            
            # Full orderbook state (raw snapshots)
            ('bid_prices', pa.list_(pa.float64())),    # All bid prices
            ('bid_volumes', pa.list_(pa.float64())),   # Volume at each bid
            ('ask_prices', pa.list_(pa.float64())),    # All ask prices
            ('ask_volumes', pa.list_(pa.float64())),   # Volume at each ask
            ('bid_order_counts', pa.list_(pa.int32())), # Orders at each bid
            ('ask_order_counts', pa.list_(pa.int32())), # Orders at each ask
            
            # Orderbook metrics at event time (atomic)
            ('best_bid', pa.float64()),
            ('best_ask', pa.float64()),
            ('best_bid_volume', pa.float64()),
            ('best_ask_volume', pa.float64()),
            ('bid_order_count', pa.int32()),   # Total orders on bid side
            ('ask_order_count', pa.int32()),   # Total orders on ask side
            
            # Market Impact (pre/post state)
            ('mid_price_before', pa.float64()),
            ('mid_price_after', pa.float64()),
            ('spread_before', pa.float64()),
            ('spread_after', pa.float64()),
            
            # Session markers
            ('session_date', pa.date32()),
            ('is_session_start', pa.bool_()),
            ('is_session_end', pa.bool_())
        ])
        
        # Initialize state tracking
        self._active_orders: Dict[str, Dict] = {}
        self._last_mid_price: Optional[float] = None
        self._last_spread: Optional[float] = None
    def create_snapshot_record(self, msg) -> Dict[str, Any]:
        """
        Create enhanced snapshot record with order flow attribution
        """
        # Basic event info
        snapshot = {
            'ts_event': msg.ts_event,
            'ts_recv': msg.ts_recv,
            'instrument': msg.symbol,
            'exchange': msg.exchange,
            'sequence_number': msg.sequence_number,
            'is_trade': False,
            'is_cancel': False
        }
        
        # Calculate market state before event
        best_bid = max(self._active_orders.get('bid', {}).keys(), default=None)
        best_ask = min(self._active_orders.get('ask', {}).keys(), default=None)
        if best_bid and best_ask:
            self._last_mid_price = (best_bid + best_ask) / 2
            self._last_spread = best_ask - best_bid
        
        if msg.record_type == 'mbo':
            # Handle Market By Order message
            snapshot.update(self._process_order_event(msg))
        elif msg.record_type == 'trade':
            # Handle Trade message
            snapshot.update(self._process_trade_event(msg))
        
        # Add market impact data
        new_best_bid = max(self._active_orders.get('bid', {}).keys(), default=None)
        new_best_ask = min(self._active_orders.get('ask', {}).keys(), default=None)
        if new_best_bid and new_best_ask:
            new_mid_price = (new_best_bid + new_best_ask) / 2
            new_spread = new_best_ask - new_best_bid
            snapshot.update({
                'mid_price_before': self._last_mid_price,
                'mid_price_after': new_mid_price,
                'spread_before': self._last_spread,
                'spread_after': new_spread
            })
        
        # Add current orderbook state
        snapshot.update(self._get_orderbook_state())
        
        return snapshot
    
    def _process_order_event(self, msg) -> Dict[str, Any]:
        """
        Process and classify order events
        """
        event_data = {
            'order_id': msg.order_id,
            'side': msg.side,
            'price': msg.price,
            'size': msg.size
        }
        
        if msg.action == 'add':
            event_data.update({
                'event_type': 'new',
                'event_subtype': f'passive_{msg.side}',
                'time_in_book': 0
            })
            self._active_orders[msg.order_id] = {
                'side': msg.side,
                'price': msg.price,
                'size': msg.size,
                'entry_time': msg.ts_event
            }
            
        elif msg.action == 'modify':
            old_order = self._active_orders.get(msg.order_id, {})
            event_data.update({
                'event_type': 'modify',
                'old_price': old_order.get('price'),
                'old_size': old_order.get('size'),
                'time_in_book': (msg.ts_event - old_order.get('entry_time', msg.ts_event)).nanoseconds
            })
            if old_order:
                self._active_orders[msg.order_id].update({
                    'price': msg.price,
                    'size': msg.size
                })
                
        elif msg.action == 'delete':
            old_order = self._active_orders.get(msg.order_id, {})
            event_data.update({
                'event_type': 'cancel',
                'is_cancel': True,
                'cancel_type': 'user',  # Could be refined based on additional data
                'time_in_book': (msg.ts_event - old_order.get('entry_time', msg.ts_event)).nanoseconds
            })
            self._active_orders.pop(msg.order_id, None)
        
        return event_data
    
    def _process_trade_event(self, msg) -> Dict[str, Any]:
        """
        Process and attribute trade events
        """
        return {
            'event_type': 'trade',
            'is_trade': True,
            'trade_id': msg.trade_id,
            'aggressor_side': msg.aggressor_side,
            'price': msg.price,
            'size': msg.size,
            'resting_order_id': msg.resting_order_id,
            'aggressive_order_id': msg.aggressive_order_id
        }
    
    def _get_orderbook_state(self) -> Dict[str, Any]:
        """
        Get current full orderbook state
        """
        # Organize orders by side and price
        bids = {}
        asks = {}
        for order_id, order in self._active_orders.items():
            if order['side'] == 'bid':
                bids[order['price']] = bids.get(order['price'], 0) + order['size']
            else:
                asks[order['price']] = asks.get(order['price'], 0) + order['size']
        
        # Sort and limit to depth levels
        sorted_bids = sorted(bids.items(), reverse=True)[:self.depth_levels]
        sorted_asks = sorted(asks.items())[:self.depth_levels]
        
        return {
            'bid_prices': [price for price, _ in sorted_bids],
            'bid_volumes': [volume for _, volume in sorted_bids],
            'ask_prices': [price for price, _ in sorted_asks],
            'ask_volumes': [volume for _, volume in sorted_asks],
            'best_bid': sorted_bids[0][0] if sorted_bids else None,
            'best_ask': sorted_asks[0][0] if sorted_asks else None,
            'bid_volume': sum(bids.values()),
            'ask_volume': sum(asks.values())
        }