import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

# Internal imports
from core import (
    load_config,
    validate_config
)
from data import (
    OrderBookDataLoader,
    SequenceIdentifier
)
from engine import (
    TimeSeriesSequenceAnalyzer,
    create_transformer_dataset
)
from utils import (
    OrderBookLogger,
    LogConfig,
    validate_orderbook_data,
    calculate_sequence_metrics
)

def setup_feature_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Setup feature configurations for the analyzer"""
    return config.get('timeseries', {})

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Price Sequence Detection System')
    parser.add_argument('config', type=Path, help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--log-dir', type=Path, default=Path('logs'), help='Log directory')
    args = parser.parse_args()
    
    # Initialize logging
    log_config = LogConfig(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_to_file=True,
        log_to_console=True,
        performance_tracking=True
    )
    logger = OrderBookLogger(log_config)
    
    try:
        # Load and validate configuration
        logger.log_milestone("Loading configuration")
        config = load_config(args.config)
        validate_config(config)
        
        # Initialize data loader
        logger.log_milestone("Initializing data loader")
        data_loader = OrderBookDataLoader(
            data_dir=config['data']['input_dir'],
            logger=logger,
            file_pattern=config['data']['file_pattern']
        )
        
        # Load orderbook data
        logger.log_milestone("Loading orderbook data")
        data = data_loader.load_data(
            start_time=datetime.fromisoformat(config['processing']['start_time']),
            end_time=datetime.fromisoformat(config['processing']['end_time']),
            symbol=config['processing']['symbol']
        )
        
        # Validate data
        logger.log_milestone("Validating data")
        validate_orderbook_data(data)
        
        # Initialize sequence identifier
        logger.log_milestone("Initializing sequence identifier")
        sequence_identifier = SequenceIdentifier(
            config=config['sequence'],
            logger=logger
        )
        
        # Identify sequences
        logger.log_milestone("Identifying price sequences")
        with logger.track_performance("sequence_identification"):
            sequences = sequence_identifier.identify_sequences(data)
        
        # Calculate and log sequence statistics
        sequence_stats = calculate_sequence_metrics(sequences)
        logger.log_milestone("Sequence identification complete", sequence_stats)
        
        if not sequences:
            logger.log_warning("No sequences found matching criteria")
            return
        
        # Setup feature configurations
        feature_configs = setup_feature_configs(config)
        
        # Initialize time-series sequence analyzer
        logger.log_milestone("Initializing time-series analyzer")
        analyzer = TimeSeriesSequenceAnalyzer(
            logger=logger,
            feature_configs=feature_configs
        )
        
        # Process sequences to extract features
        logger.log_milestone("Extracting time-series features")
        sequence_features = []
        
        with logger.track_performance("feature_extraction"):
            for i, sequence in enumerate(sequences):
                features = analyzer.analyze_sequence(data, sequence)
                if features:
                    sequence_features.append({
                        'sequence': sequence,
                        'features': features
                    })
                
                if (i + 1) % 100 == 0:
                    logger.log_milestone(f"Processed {i + 1}/{len(sequences)} sequences")
        
        # Create transformer dataset
        logger.log_milestone("Creating transformer dataset")
        with logger.track_performance("transformer_dataset_creation"):
            transformer_dataset = create_transformer_dataset(
                data=data,
                sequences=[sf['sequence'] for sf in sequence_features],
                logger=logger
            )
        
        # Save the dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol = config['processing']['symbol']
        
        output_dir = Path(config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save transformer dataset
        dataset_path = output_dir / f"{symbol}_transformer_dataset_{timestamp}.pt"
        
        try:
            import torch
            torch.save({
                'dataset': transformer_dataset,
                'feature_configs': feature_configs,
                'config': config,
                'sequence_stats': sequence_stats,
                'timestamp': timestamp
            }, dataset_path)
            logger.log_milestone("Saved PyTorch dataset", {'path': str(dataset_path)})
        except ImportError:
            import pickle
            pickle_path = dataset_path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'dataset': transformer_dataset,
                    'feature_configs': feature_configs,
                    'config': config,
                    'sequence_stats': sequence_stats,
                    'timestamp': timestamp
                }, f)
            logger.log_milestone("Saved pickle dataset", {'path': str(pickle_path)})
        
        # Save metadata
        metadata = {
            'config': config,
            'feature_configs': feature_configs,
            'sequence_statistics': sequence_stats,
            'dataset_info': {
                'num_samples': len(transformer_dataset),
                'feature_dimensions': transformer_dataset.get_feature_dimensions(),
                'lookback_modes': {
                    collector: {
                        name: {
                            'type': feat_config['lookback_type'],
                            'granularity': (feat_config.get('granularity_ms') or 
                                          feat_config.get('granularity_messages')),
                            'history': (feat_config.get('history_ms') or 
                                      feat_config.get('history_messages'))
                        }
                        for name, feat_config in collector_config.items()
                    }
                    for collector, collector_config in feature_configs.items()
                }
            },
            'timestamp': timestamp
        }
        
        with open(output_dir / f"{symbol}_metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save performance summary
        logger.save_performance_summary(
            output_dir / f"performance_{timestamp}.json"
        )
        
        logger.log_milestone("Processing completed successfully", {
            'sequences_found': len(sequences),
            'sequences_with_features': len(sequence_features),
            'dataset_samples': len(transformer_dataset),
            'feature_modes': list(feature_configs.keys()),
            'output_files': [
                str(dataset_path),
                str(output_dir / f"{symbol}_metadata_{timestamp}.json")
            ]
        })
        
    except Exception as e:
        logger.log_error(e, "Fatal error in main process")
        raise

if __name__ == "__main__":
    main()