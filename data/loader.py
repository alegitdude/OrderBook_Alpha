from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import numpy as np

from ..utils.logging import OrderBookLogger

class OrderBookDataLoader:
    """Handles loading and preprocessing of orderbook data"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        logger: OrderBookLogger,
        file_pattern: str = "*.parquet"
    ):
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        self.file_metadata = {}
        
        self._scan_files()
    
    def _scan_files(self):
        """Scan directory for data files and cache metadata"""
        self.logger.log_milestone("Scanning data files")
        file_paths = sorted(self.data_dir.glob(self.file_pattern))
        
        for file_path in file_paths:
            try:
                parquet_file = pq.ParquetFile(file_path)
                metadata = parquet_file.metadata
                
                # Read first and last row for time range
                first_batch = next(parquet_file.iter_batches(batch_size=1))
                last_batch = next(parquet_file.iter_batches(batch_size=1, start_rows=metadata.num_rows-1))
                
                df_first = first_batch.to_pandas()
                df_last = last_batch.to_pandas()
                
                self.file_metadata[file_path] = {
                    'num_rows': metadata.num_rows,
                    'start_time': df_first['ts_event'].iloc[0],
                    'end_time': df_last['ts_event'].iloc[0],
                    'symbol': df_first['symbol'].iloc[0]
                }
                
                self.logger.log_milestone("File scanned", {
                    'file': str(file_path),
                    'rows': metadata.num_rows,
                    'time_range': f"{df_first['ts_event'].iloc[0]} to {df_last['ts_event'].iloc[0]}"
                })
                
            except Exception as e:
                self.logger.log_error(e, f"Error scanning file {file_path}")
    
    def load_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Load orderbook data within specified range"""
        relevant_files = []
        
        for file_path, metadata in self.file_metadata.items():
            if start_time and metadata['end_time'] < start_time:
                continue
            if end_time and metadata['start_time'] > end_time:
                continue
            if symbol and metadata['symbol'] != symbol:
                continue
            relevant_files.append(file_path)
        
        dfs = []
        for file_path in sorted(relevant_files):
            try:
                df = pd.read_parquet(file_path)
                
                # Apply time filters
                if start_time:
                    df = df[df['ts_event'] >= start_time]
                if end_time:
                    df = df[df['ts_event'] <= end_time]
                
                dfs.append(df)
                
                self.logger.log_milestone("Loaded file", {
                    'file': str(file_path),
                    'rows': len(df)
                })
                
            except Exception as e:
                self.logger.log_error(e, f"Error loading file {file_path}")
        
        if not dfs:
            raise ValueError("No data found matching criteria")
        
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.sort_values('ts_event', inplace=True)
        
        self.logger.log_milestone("Data loading complete", {
            'total_rows': len(final_df),
            'time_range': f"{final_df['ts_event'].iloc[0]} to {final_df['ts_event'].iloc[-1]}"
        })
        
        return final_df