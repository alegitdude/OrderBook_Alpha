from typing import Iterator, Union
import databento as db
import pandas as pd
from pathlib import Path
from Logger import MarketDataLogger

class DatabentoImporter:
    """
    Handles importing Databento market data from either:
    1. Databento CSV files (with standard MBO futures schema)
    2. DBN files (with zstd compression )
    """
    
    def __init__(self):
        # Initialize logger
        self.logger = MarketDataLogger(
            name="DatabentoImporter",
            base_dir=Path("./data")  # Could be parameterized
        ).get_logger()
        
        # Define Databento MBO futures schema
        self.mbo_columns = [
            'ts_recv',          # Timestamp of receipt
            'ts_event',         # Timestamp of event
            'rtype',            # Record type
            'publisher_id',     # Publisher identifier
            'instrument_id',    # Instrument identifier
            'action',           # add, modify, cancel, trade
            'side',            # bid or ask
            'price',           # Order price
            'size',            # Order size
            'channel_id',      # Channel identifier
            'order_id',        # Order ID
            'flags',           # Order flags
            'ts_in_delta',     # Time delta for inbound message
            'sequence',        # Message sequence number
            'symbol'           # Instrument symbol
        ]
    
    def create_event_stream(self, file_path: str) -> Iterator:
        """
        Create unified event stream from either CSV or DBN file
        
        Parameters:
        - file_path: Path to input file (either .csv or .dbn)
        
        Returns:
        - Iterator of standardized market data events
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return self._process_csv(file_path)
        elif file_path.suffix.lower() == '.dbn':
            return self._process_dbn(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def _process_csv(self, csv_path: Path) -> Iterator:
        """
        Process Databento CSV file
        """
        try:
            # Skip header if it exists
            header_row = pd.read_csv(csv_path, nrows=0)
            skip_rows = 1 if any(col in header_row.columns for col in ['ts_recv', 'ts_event']) else 0
            
            # Read CSV in chunks to handle large files efficiently
            for chunk in pd.read_csv(
                csv_path,
                names=self.mbo_columns,
                dtype={
                    'ts_recv': 'int64',
                    'ts_event': 'int64',
                    'rtype': 'str',
                    'publisher_id': 'str',
                    'instrument_id': 'int64',
                    'action': 'str',
                    'side': 'str',
                    'price': 'float64',
                    'size': 'float64',
                    'channel_id': 'int64',
                    'order_id': 'str',
                    'flags': 'int64',
                    'ts_in_delta': 'int64',
                    'sequence': 'int64',
                    'symbol': 'str'
                },
                skiprows=skip_rows,
                chunksize=10000
            ):
                # Convert timestamps from nanoseconds to datetime
                chunk['ts_event'] = pd.to_datetime(chunk['ts_event'], unit='ns')
                chunk['ts_recv'] = pd.to_datetime(chunk['ts_recv'], unit='ns')
                
                # Yield each row as a dictionary
                for _, row in chunk.iterrows():
                    yield row.to_dict()
                    
        except Exception as e:
            self.logger.error(f"Error processing CSV file {csv_path}: {str(e)}")
            raise
    
    def _process_dbn(self, dbn_path: Path) -> Iterator:
        """
        Process Databento DBN file (with zstd compression)
        """
        try:
            # Use Databento's DBZReader for DBN files
            with db.DBZReader(str(dbn_path)) as reader:
                # Iterate through messages
                for msg in reader:
                    # DBZReader already provides messages in correct format
                    yield msg
                    
        except Exception as e:
            self.logger.error(f"Error processing DBN file {dbn_path}: {str(e)}")
            raise
    
    def validate_event(self, event: dict) -> bool:
        """
        Validate that event has all required fields
        """
        required_fields = set(self.mbo_columns)  # Use all columns as required fields
        return all(field in event for field in required_fields)