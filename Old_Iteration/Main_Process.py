from typing import Iterator, Union
import databento as db
# import pandas as pd
# from pathlib import Path
from Data_Import import DatabentoImporter
# Usage example
def main():
    # Initialize importer
    importer = DatabentoImporter()
    
    # Process CSV file
    csv_file = "path/to/databento_mbo_data.csv"
    for event in importer.create_event_stream(csv_file):
        if importer.validate_event(event):
            # Process the event
            print(f"Processing event: {event['action']} order {event['order_id']}")
    
    # Process DBN file
    dbn_file = "path/to/databento_mbo_data.dbn"
    for event in importer.create_event_stream(dbn_file):
        if importer.validate_event(event):
            # Process the event
            print(f"Processing event: {event['action']} order {event['order_id']}")

if __name__ == "__main__":
    main()