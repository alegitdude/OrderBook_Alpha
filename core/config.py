import yaml
from pathlib import Path
from typing import Dict, Any, Union

from .types import (
    SequenceConfig,
    FeatureConfig,
    AnalysisConfig,
    LookbackType
)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse configuration file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")
    
    return parse_config(config)

def parse_config(config: Dict) -> Dict[str, Any]:
    """Parse raw config dict into typed config objects"""
    
    # Parse sequence identification config
    sequence_config = SequenceConfig(
        min_moves=config['sequence']['min_moves'],
        min_ticks=config['sequence']['min_ticks'],
        max_duration_ms=config['sequence']['max_duration_ms'],
        min_volume=config['sequence']['min_volume'],
        max_retrace_ticks=config['sequence']['max_retrace_ticks'],
        tick_size=config['sequence']['tick_size']
    )
    
    # Parse feature configs
    feature_configs = {}
    for name, feat_config in config['features'].items():
        feature_configs[name] = FeatureConfig(
            name=name,
            lookback_type=LookbackType[feat_config['lookback_type'].upper()],
            lookback_value=feat_config['lookback_value'],
            compute_frequency=feat_config.get('compute_frequency', 1)
        )
    
    # Parse analysis config
    analysis_config = AnalysisConfig(
        lookback_messages=config['analysis']['lookback_messages'],
        lookback_intervals=config['analysis']['lookback_intervals'],
        required_calm_ticks=config['analysis']['required_calm_ticks'],
        min_confidence=config['analysis']['min_confidence']
    )
    
    # Parse data paths
    data_config = {
        'input_dir': Path(config['data']['input_dir']),
        'output_dir': Path(config['data']['output_dir']),
        'file_pattern': config['data'].get('file_pattern', '*.parquet')
    }
    
    return {
        'sequence': sequence_config,
        'features': feature_configs,
        'analysis': analysis_config,
        'data': data_config
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration values"""
    sequence_config = config['sequence']
    
    # Validate sequence parameters
    if sequence_config.min_moves < 2:
        raise ValueError("min_moves must be at least 2")
    if sequence_config.min_ticks <= 0:
        raise ValueError("min_ticks must be positive")
    if sequence_config.max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be positive")
    if sequence_config.tick_size <= 0:
        raise ValueError("tick_size must be positive")
    
    # Validate analysis parameters
    analysis_config = config['analysis']
    if analysis_config.lookback_messages <= 0:
        raise ValueError("lookback_messages must be positive")
    if not analysis_config.lookback_intervals:
        raise ValueError("lookback_intervals cannot be empty")
    if min(analysis_config.lookback_intervals) <= 0:
        raise ValueError("lookback_intervals must be positive")
    
    # Validate data paths
    if not config['data']['input_dir'].exists():
        raise ValueError(f"Input directory does not exist: {config['data']['input_dir']}")
    
    return True