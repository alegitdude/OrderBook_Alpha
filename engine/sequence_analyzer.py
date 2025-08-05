from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..core.types import (
    PriceSequence,
    TimeSeriesConfig,
    LookbackType
)
from ..calculators.sequence_feature_collector import SequenceFeatureCollector
from ..utils.logging import OrderBookLogger

class TimeSeriesSequenceAnalyzer:
    """Analyzes market state before price sequences using configurable time-series features"""
    
    def __init__(
        self,
        logger: OrderBookLogger,
        feature_configs: Dict[str, Dict[str, Any]]
    ):
        self.logger = logger
        self.feature_collectors = {}
        self._setup_collectors(feature_configs)
    
    def _setup_collectors(self, feature_configs: Dict[str, Dict[str, Any]]):
        """Setup feature collectors based on configuration"""
        
        for collector_name, config_set in feature_configs.items():
            configs = {}
            
            for feature_name, feature_params in config_set.items():
                lookback_type = LookbackType[feature_params['lookback_type'].upper()]
                
                if lookback_type == LookbackType.TIME:
                    configs[feature_name] = TimeSeriesConfig(
                        feature_name=f"{feature_name}_{collector_name}",
                        lookback_type=lookback_type,
                        granularity_ms=feature_params['granularity_ms'],
                        history_ms=feature_params['history_ms']
                    )
                else:  # MESSAGES
                    configs[feature_name] = TimeSeriesConfig(
                        feature_name=f"{feature_name}_{collector_name}",
                        lookback_type=lookback_type,
                        granularity_messages=feature_params['granularity_messages'],
                        history_messages=feature_params['history_messages']
                    )
            
            self.feature_collectors[collector_name] = SequenceFeatureCollector(
                logger=self.logger,
                configs=configs
            )
    
    def analyze_sequence(
        self,
        data: pd.DataFrame,
        sequence: PriceSequence
    ) -> Optional[Dict[str, np.ndarray]]:
        """Analyze market state before a price sequence"""
        try:
            all_features = {}
            
            # Collect features from each collector
            for collector_name, collector in self.feature_collectors.items():
                features = collector.collect_features(
                    data=data,
                    sequence_start_idx=sequence.start_index
                )
                
                # Add collector name to feature keys
                for name, series in features.items():
                    feature_key = f"{name}_{collector_name}"
                    all_features[feature_key] = series
            
            if not all_features:
                self.logger.log_warning("No features extracted for sequence")
                return None
            
            self.logger.log_milestone("Sequence analysis complete", {
                'sequence_start': sequence.start_time,
                'features_extracted': list(all_features.keys()),
                'feature_lengths': {k: len(v) for k, v in all_features.items()}
            })
            
            return all_features
            
        except Exception as e:
            self.logger.log_error(e, "Error analyzing sequence")
            return None