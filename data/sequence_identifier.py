# =============================================================================
# data/sequence_identifier.py
# =============================================================================
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..core.types import PriceSequence, SequenceConfig
from ..utils.logging import OrderBookLogger

class SequenceIdentifier:
    """Identifies sequences of consecutive price moves"""
    
    def __init__(
        self,
        config: SequenceConfig,
        logger: OrderBookLogger
    ):
        self.config = config
        self.logger = logger
    
    def identify_sequences(self, data: pd.DataFrame) -> List[PriceSequence]:
        """Find sequences of consecutive price moves in the same direction"""
        sequences = []
        current_direction = 0
        moves_count = 0
        sequence_start_idx = 0
        sequence_volume = 0
        trade_sizes = []
        max_retrace = 0
        last_price = data.iloc[0]['mid_price']
        
        self.logger.log_milestone("Starting sequence identification")
        
        for idx, row in data.iterrows():
            current_price = row['mid_price']
            price_change_ticks = (current_price - last_price) / self.config.tick_size
            
            # Check if we have a significant move
            if abs(price_change_ticks) >= self.config.min_ticks:
                move_direction = np.sign(price_change_ticks)
                
                # Continue or start new sequence
                if move_direction == current_direction:
                    moves_count += 1
                    if row['is_trade']:
                        sequence_volume += row['size']
                        trade_sizes.append(row['size'])
                    
                    # Update max retrace
                    start_price = data.iloc[sequence_start_idx]['mid_price']
                    expected_price = start_price + (current_direction * moves_count * 
                                                  self.config.min_ticks * 
                                                  self.config.tick_size)
                    retrace = abs(current_price - expected_price) / self.config.tick_size
                    max_retrace = max(max_retrace, retrace)
                    
                    # Check if sequence is complete
                    if moves_count >= self.config.min_moves:
                        sequence = self._create_sequence(
                            data=data,
                            start_idx=sequence_start_idx,
                            end_idx=idx,
                            direction=current_direction,
                            volume=sequence_volume,
                            trade_sizes=trade_sizes,
                            max_retrace=max_retrace
                        )
                        
                        if self._validate_sequence(sequence):
                            sequences.append(sequence)
                        
                        # Reset tracking
                        moves_count = 0
                        current_direction = 0
                        sequence_volume = 0
                        trade_sizes = []
                        max_retrace = 0
                
                # Start new sequence
                else:
                    current_direction = move_direction
                    moves_count = 1
                    sequence_start_idx = idx
                    sequence_volume = row['size'] if row['is_trade'] else 0
                    trade_sizes = [row['size']] if row['is_trade'] else []
                    max_retrace = 0
            
            # Check sequence timeout
            if (idx - sequence_start_idx > 0 and 
                (data.iloc[idx]['ts_event'] - 
                 data.iloc[sequence_start_idx]['ts_event']).total_seconds() * 1000 > 
                self.config.max_duration_ms):
                moves_count = 0
                current_direction = 0
                sequence_volume = 0
                trade_sizes = []
                max_retrace = 0
            
            # Check retracement limit
            if max_retrace > self.config.max_retrace_ticks:
                moves_count = 0
                current_direction = 0
                sequence_volume = 0
                trade_sizes = []
                max_retrace = 0
            
            last_price = current_price
        
        self.logger.log_milestone("Sequence identification complete", {
            'sequences_found': len(sequences),
            'avg_moves': np.mean([s.num_moves for s in sequences]),
            'avg_ticks': np.mean([s.total_ticks for s in sequences]),
            'avg_duration': np.mean([s.duration_seconds for s in sequences])
        })
        
        return sequences
    
    def _create_sequence(
        self,
        data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        direction: int,
        volume: float,
        trade_sizes: List[float],
        max_retrace: float
    ) -> PriceSequence:
        """Create PriceSequence object from identified sequence"""
        start_price = data.iloc[start_idx]['mid_price']
        end_price = data.iloc[end_idx]['mid_price']
        duration = (data.iloc[end_idx]['ts_event'] - 
                   data.iloc[start_idx]['ts_event']).total_seconds()
        
        return PriceSequence(
            start_time=data.iloc[start_idx]['ts_event'],
            end_time=data.iloc[end_idx]['ts_event'],
            start_price=start_price,
            end_price=end_price,
            total_ticks=abs(end_price - start_price) / self.config.tick_size,
            direction=direction,
            start_index=start_idx,
            end_index=end_idx,
            num_moves=self.config.min_moves,
            moves_per_second=self.config.min_moves / duration if duration > 0 else 0,
            volume_during=volume,
            avg_trade_size=np.mean(trade_sizes) if trade_sizes else 0,
            max_retrace=max_retrace
        )
    
    def _validate_sequence(self, sequence: PriceSequence) -> bool:
        """Validate identified sequence meets all criteria"""
        # Check minimum volume
        if sequence.volume_during < self.config.min_volume:
            return False
        
        # Check duration
        if (sequence.duration_seconds * 1000) > self.config.max_duration_ms:
            return False
        
        # Check retracement
        if sequence.max_retrace > self.config.max_retrace_ticks:
            return False
            
        return True