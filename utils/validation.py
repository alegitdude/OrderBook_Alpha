from typing import Dict, Any
import pandas as pd
import numpy as np

def validate_orderbook_data(data: pd.DataFrame) -> bool:
    """Validate orderbook data structure and content"""
    required_columns = {
        'ts_event', 'mid_price', 'book_state',
        'is_trade', 'side', 'size'
    }
    
    # Check required columns
    missing_cols = required_columns - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(data['ts_event']):
        raise ValueError("ts_event must be datetime type")
    
    if not pd.api.types.is_float_dtype(data['mid_price']):
        raise ValueError("mid_price must be float type")
    
    # Check for null values
    null_counts = data[list(required_columns)].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Found null values: {null_counts[null_counts > 0]}")
    
    # Validate book state structure
    if not all(isinstance(state, dict) for state in data['book_state']):
        raise ValueError("Invalid book_state structure")
    
    return True

def validate_sequences(sequences_df: pd.DataFrame) -> bool:
    """Validate identified price sequences"""
    required_columns = {
        'start_time', 'end_time', 'total_ticks',
        'direction', 'num_moves', 'volume_during'
    }
    
    # Check required columns
    missing_cols = required_columns - set(sequences_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate sequence properties
    if any(sequences_df['end_time'] <= sequences_df['start_time']):
        raise ValueError("Found sequences with end_time <= start_time")
    
    if any(sequences_df['total_ticks'] <= 0):
        raise ValueError("Found sequences with non-positive tick movement")
    
    if not all(np.isin(sequences_df['direction'], [-1, 1])):
        raise ValueError("Invalid sequence directions found")
    
    return True