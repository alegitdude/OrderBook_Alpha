import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from Logger import MarketDataLogger
import os
import logging

@dataclass
class OrderBookSnapshotRecorder:
    """
    Records and stores orderbook snapshots from Databento market data
    Optimized for futures market data with relative price levels
    """
    instrument: str
    tick_size: float
    output_dir: str = './raw_orderbook_snapshots/'
    max_levels: int = 50
    batch_size: int = 100_000
    
    def __post_init__(self):
        # Initialize logger
        self.logger = MarketDataLogger(
            name=f"SnapshotRecorder_{self.instrument}",
            base_dir=Path(self.output_dir).parent
        ).get_logger()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define schema for Parquet storage
        self._schema = pa.schema([
            # Raw Databento message fields (in CSV order)
            ('ts_recv', pa.timestamp('ns')),
            ('ts_event', pa.timestamp('ns')),
            ('rtype', pa.string()),
            ('publisher_id', pa.string()),
            ('instrument_id', pa.int64()),
            ('action', pa.string()),
            ('side', pa.string()),
            ('price', pa.float64()),
            ('size', pa.float64()),
            ('channel_id', pa.int64()),
            ('order_id', pa.string()),
            ('flags', pa.int64()),
            ('ts_in_delta', pa.int64()),
            ('sequence', pa.int64()),
            ('symbol', pa.string()),
            
            # Additional derived fields for order book tracking
            ('event_type', pa.string()),      # 'new', 'modify', 'cancel', 'trade'
            ('event_subtype', pa.string()),   # 'aggressive_buy', 'passive_sell', etc.
            ('is_trade', pa.bool_()),
            ('is_cancel', pa.bool_()),
            
            # Order lifecycle fields
            ('original_order_id', pa.string()),
            ('old_price', pa.float64()),
            ('old_size', pa.float64()),
            ('entry_time', pa.timestamp('ns')),
            ('original_entry_time', pa.timestamp('ns')),
            
            # Market state
            ('mid_price_before', pa.float64()),
            ('mid_price_after', pa.float64()),
            ('spread_before', pa.float64()),
            ('spread_after', pa.float64()),
            
            # Book state
            ('best_bid', pa.float64()),
            ('best_ask', pa.float64()),
            ('mid_price', pa.float64()),
            ('spread_ticks', pa.int32()),
            
            # Structured orderbook state
            ('book_state', pa.struct([
                ('bid_side', pa.list_(pa.struct([
                    ('relative_level', pa.int32()),
                    ('price', pa.float64()),
                    ('volume', pa.float64()),
                    ('order_count', pa.int32())
                ]))),
                ('ask_side', pa.list_(pa.struct([
                    ('relative_level', pa.int32()),
                    ('price', pa.float64()),
                    ('volume', pa.float64()),
                    ('order_count', pa.int32())
                ])))
            ])),
            
            # Session markers
            ('is_session_start', pa.bool_()),
            ('is_session_end', pa.bool_())
        ])
        
        # Initialize state tracking
        self._active_orders: Dict[str, Dict] = {}
        self._current_batch: List[Dict[str, Any]] = []
        self._current_file = self._create_new_file()
    
    def _create_new_file(self) -> str:
        """Create new Parquet file name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'{self.output_dir}/{self.instrument}_snapshots_{timestamp}.parquet'
    
    def process_databento_message(self, msg):
        """Process a single Databento message"""
        try:
            # Create basic snapshot with raw message data
            snapshot = {
                'ts_recv': msg['ts_recv'],
                'ts_event': msg['ts_event'],
                'rtype': msg['rtype'],
                'publisher_id': msg['publisher_id'],
                'instrument_id': msg['instrument_id'],
                'action': msg['action'],
                'side': msg['side'],
                'price': msg['price'],
                'size': msg['size'],
                'channel_id': msg['channel_id'],
                'order_id': msg['order_id'],
                'flags': msg['flags'],
                'ts_in_delta': msg['ts_in_delta'],
                'sequence': msg['sequence'],
                'symbol': msg['symbol'],
                'is_trade': False,
                'is_cancel': False
        }
            
            # Calculate market state before event
            best_bid = max(self._active_orders.get('bid', {}).keys(), default=None)
            best_ask = min(self._active_orders.get('ask', {}).keys(), default=None)
            
            if best_bid and best_ask:
                mid_price_before = (best_bid + best_ask) / 2
                spread_before = best_ask - best_bid
                snapshot.update({
                    'mid_price_before': mid_price_before,
                    'spread_before': spread_before
                })
            
            # Process by message type
            order_data = self._process_order_event(msg)
            snapshot.update(order_data)
            
            # Get updated orderbook state
            book_state = self._get_orderbook_state()
            snapshot.update(book_state)
            
            # Add market impact data
            if best_bid and best_ask and book_state['best_bid'] and book_state['best_ask']:
                snapshot.update({
                    'mid_price_after': book_state['mid_price'],
                    'spread_after': book_state['best_ask'] - book_state['best_bid']
                })
            
            # Session markers
            is_first_msg = not bool(self._active_orders)
            snapshot['is_session_start'] = is_first_msg
            
            # Add to batch and write if needed
            self._current_batch.append(snapshot)
            if len(self._current_batch) >= self.batch_size:
                self._write_snapshot()
                
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise
    
    def _process_order_event(self, msg) -> Dict[str, Any]:
        """Process and classify order events"""
        event_data = {
            'order_id': msg['order_id'],
            'side': msg['side'],
            'price': msg['price'],
            'size': msg['size'],
            'flags': msg['flags'],
            'action': msg['action']
        }
        
        if msg['action'] == 'add':
            event_data.update({
                'event_type': 'new',
                'event_subtype': f'passive_{msg["side"]}',
                'entry_time': msg['ts_event'],
                'original_entry_time': msg['ts_event']
            })
            self._active_orders[msg['order_id']] = {
                'side': msg['side'],
                'price': msg['price'],
                'size': msg['size'],
                'entry_time': msg['ts_event'],
                'flags': msg['flags']
            }
        
        elif msg['action'] == 'modify':
            old_order = self._active_orders.get(msg['order_id'], {})
            event_data.update({
                'event_type': 'modify',
                'event_subtype': f'modify_{msg["side"]}',
                'old_price': old_order.get('price'),
                'old_size': old_order.get('size'),
                'original_entry_time': old_order.get('entry_time'),
                'original_order_id': msg['order_id']
            })
            if old_order:
                self._active_orders[msg['order_id']].update({
                    'price': msg['price'],
                    'size': msg['size'],
                    'flags': msg['flags']
                })
                
        elif msg['action'] == 'cancel':
            old_order = self._active_orders.get(msg['order_id'], {})
            event_data.update({
                'event_type': 'cancel',
                'event_subtype': f'cancel_{msg["side"]}',
                'is_cancel': True,
                'original_entry_time': old_order.get('entry_time'),
                'original_order_id': msg['order_id']
            })
            self._active_orders.pop(msg['order_id'], None)
        
        return event_data
    
    def _process_trade_event(self, msg) -> Dict[str, Any]:
        """Process and classify trade events"""
        return {
            'event_type': 'trade',
            'event_subtype': f'aggressive_{msg.aggressor_side}',
            'is_trade': True,
            'trade_id': msg.trade_id,
            'aggressor_side': msg.aggressor_side,
            'trade_size': msg.size,
            'resting_order_id': msg.resting_order_id,
            'aggressive_order_id': msg.aggressive_order_id,
            'price': msg.price
        }
    
    def _get_orderbook_state(self) -> Dict[str, Any]:
        """Get current full orderbook state with relative price levels"""
        # First pass: collect orders
        bids = {}
        asks = {}
        bid_counts = {}
        ask_counts = {}
        
        for order_id, order in self._active_orders.items():
            if order['side'] == 'bid':
                bids[order['price']] = bids.get(order['price'], 0) + order['size']
                bid_counts[order['price']] = bid_counts.get(order['price'], 0) + 1
            else:
                asks[order['price']] = asks.get(order['price'], 0) + order['size']
                ask_counts[order['price']] = ask_counts.get(order['price'], 0) + 1
        
        # Find best bid/ask
        best_bid = max(bids.keys()) if bids else None
        best_ask = min(asks.keys()) if asks else None
        
        if best_bid is None or best_ask is None:
            return {
                'best_bid': None,
                'best_ask': None,
                'mid_price': None,
                'spread_ticks': None,
                'book_state': {'bid_side': [], 'ask_side': []}
            }
        
        # Create structured book state
        book_state = {
            'bid_side': [
                {
                    'relative_level': int(round((best_bid - price) / self.tick_size)),
                    'price': float(price),
                    'volume': float(bids[price]),
                    'order_count': int(bid_counts[price])
                }
                for price in sorted(bids.keys(), reverse=True)
                if int(round((best_bid - price) / self.tick_size)) <= self.max_levels
            ],
            'ask_side': [
                {
                    'relative_level': int(round((price - best_ask) / self.tick_size)),
                    'price': float(price),
                    'volume': float(asks[price]),
                    'order_count': int(ask_counts[price])
                }
                for price in sorted(asks.keys())
                if int(round((price - best_ask) / self.tick_size)) <= self.max_levels
            ]
        }
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': (best_bid + best_ask) / 2,
            'spread_ticks': int(round((best_ask - best_bid) / self.tick_size)),
            'book_state': book_state
        }
    
    def _write_snapshot(self):
        """Write current batch to Parquet file"""
        if not self._current_batch:
            return
        
        try:
            # Convert to PyArrow table
            table = pa.Table.from_pylist(self._current_batch, schema=self._schema)
            
            # Write with compression
            pq.write_table(
                table,
                self._current_file,
                compression='snappy',
                write_statistics=True
            )
            
            # Reset batch and create new file
            self._current_batch = []
            self._current_file = self._create_new_file()
            
        except Exception as e:
            self.logger.error(f"Error writing snapshot: {str(e)}")
            raise
    
    def finalize(self):
        """Finalize processing and write any remaining data"""
        if self._current_batch:
            self._write_snapshot() 