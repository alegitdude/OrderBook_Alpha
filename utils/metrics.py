from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import timedelta

from ..core.types import PriceSequence

def calculate_sequence_metrics(sequences: List[PriceSequence]) -> Dict[str, Any]:
    """Calculate statistics about identified sequences"""
    if not sequences:
        return {}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            'direction': s.direction,
            'total_ticks': s.total_ticks,
            'num_moves': s.num_moves,
            'duration_seconds': s.duration_seconds,
            'moves_per_second': s.moves_per_second,
            'volume': s.volume_during,
            'avg_trade_size': s.avg_trade_size,
            'max_retrace': s.max_retrace,
            'return': s.total_return
        }
        for s in sequences
    ])
    
    metrics = {
        'total_sequences': len(sequences),
        'up_sequences': (df['direction'] == 1).sum(),
        'down_sequences': (df['direction'] == -1).sum(),
        'avg_ticks': df['total_ticks'].mean(),
        'avg_moves': df['num_moves'].mean(),
        'avg_duration': df['duration_seconds'].mean(),
        'avg_moves_per_second': df['moves_per_second'].mean(),
        'avg_volume': df['volume'].mean(),
        'avg_trade_size': df['avg_trade_size'].mean(),
        'avg_retrace': df['max_retrace'].mean(),
        'avg_return': df['return'].mean(),
        'max_return': df['return'].max(),
        'min_return': df['return'].min()
    }
    
    # Add time-based metrics
    sequences_sorted = sorted(sequences, key=lambda x: x.start_time)
    time_diffs = [
        (sequences_sorted[i+1].start_time - sequences_sorted[i].end_time).total_seconds()
        for i in range(len(sequences_sorted)-1)
    ]
    
    if time_diffs:
        metrics.update({
            'avg_time_between': np.mean(time_diffs),
            'min_time_between': min(time_diffs),
            'max_time_between': max(time_diffs)
        })
    
    return metrics