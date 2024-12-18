from Logger import MarketDataLogger
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from Data_Import import DatabentoImporter
from Event_to_Packet_Capture import OrderBookSnapshotRecorder

class OrderBookDataProcessor:
    """
    Controls the entire lifecycle of orderbook data processing:
    1. Data import (CSV or DBN)
    2. Orderbook state management
    3. Snapshot creation and storage
    """
    
    def __init__(
        self,
        instrument: str,
        tick_size: float,
        base_dir: str = "./data",
        batch_size: int = 100_000,
        max_levels: int = 50
    ):
        self.instrument = instrument
        self.tick_size = tick_size
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self.max_levels = max_levels
        
        # Set up logging
        self.logger = MarketDataLogger(
            name=f"OrderBookProcessor_{instrument}",
            base_dir=self.base_dir
        ).get_logger()
        
        # Initialize directories
        self._initialize_directories()
        
        # Initialize components
        self.importer = DatabentoImporter()
        self.recorder = None  # Will be initialized per processing session

           
    
    def _initialize_directories(self):
        """Create necessary directory structure"""
        # Main directories
        self.raw_data_dir = self.base_dir / 'raw_data'
        self.processed_data_dir = self.base_dir / 'processed_data'
        self.archive_dir = self.base_dir / 'archive'
        
        # Create all directories
        for directory in [self.raw_data_dir, self.processed_data_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_data_file(
        self,
        input_file: str,
        session_date: Optional[str] = None
    ):
        """
        Process a single data file (CSV or DBN)
        
        Parameters:
        - input_file: Path to input file
        - session_date: Optional trading session date (YYYYMMDD)
        """
        try:
            input_path = Path(input_file)
            self.logger.info(f"Starting processing of {input_path}")
            
            # Determine session date if not provided
            if not session_date:
                session_date = datetime.now().strftime("%Y%m%d")
            
            # Create output directory for this session
            output_dir = self.processed_data_dir / self.instrument / session_date
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize recorder for this session
            self.recorder = OrderBookSnapshotRecorder(
                instrument=self.instrument,
                tick_size=self.tick_size,
                output_dir=str(output_dir),
                max_levels=self.max_levels,
                batch_size=self.batch_size
            )
            
            # Create event stream
            event_stream = self.importer.create_event_stream(str(input_path))
            
            # Process events
            processed_count = 0
            for event in event_stream:
                if self.importer.validate_event(event):
                    try:
                        self.recorder.process_databento_message(event)
                        processed_count += 1
                        
                        if processed_count % 100000 == 0:
                            self.logger.info(f"Processed {processed_count} messages")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing event: {e}")
                        continue
            
            # Ensure final batch is written
            self.recorder.finalize()
            
            # Move input file to archive
            # self._archive_input_file(input_path)
            
            self.logger.info(
                f"Completed processing {processed_count} messages from {input_path}"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing file {input_file}: {str(e)}")
            raise
    
    def process_directory(
        self,
        input_dir: str,
        pattern: str = "*.csv"
    ):
        """
        Process all matching files in a directory
        """
        input_path = Path(input_dir)
        self.logger.info(f"Processing all {pattern} files in {input_path}")
        
        for file_path in sorted(input_path.glob(pattern)):
            self.process_data_file(str(file_path))
    
    def _archive_input_file(self, file_path: Path):
        """Archive processed input file"""
        archive_path = self.archive_dir / file_path.name
        file_path.rename(archive_path)
        self.logger.info(f"Archived {file_path} to {archive_path}")

# Usage example
def main():
    # Initialize processor for E-mini S&P 500 futures
    processor = OrderBookDataProcessor(
        instrument="ES",
        tick_size=0.25,
        base_dir="./market_data",
        batch_size=100_000,
        max_levels=50
    )
    
    # Process a single file
    processor.process_data_file("path/to/ES_20240314.csv")
    
    # Or process all files in a directory
    processor.process_directory(
        "path/to/data_directory",
        pattern="ES_*.csv"
    )

if __name__ == "__main__":
    main()