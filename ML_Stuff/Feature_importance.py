import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class FeatureImportanceAnalyzer:
    def __init__(self, features, targets):
        """
        Initialize feature importance analysis
        
        Parameters:
        - features: numpy array of extracted features
        - targets: numpy array of prediction targets
        """
        self.features = features
        self.targets = targets
        self.scaler = StandardScaler()
        
    def correlation_analysis(self):
        """
        Analyze feature correlations and their relationship with target
        
        Returns:
        - Correlation heatmap
        - Feature-target correlation coefficients
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(self.features)
        
        # Create DataFrame for correlation analysis
        feature_names = [
            f'Imbalance_{i}' for i in range(4)] + \
            [f'VolumeProfile_{i}' for i in range(4)] + \
            [f'Microstructure_{i}' for i in range(4)] + \
            [f'Liquidity_{i}' for i in range(4)]
        
        df = pd.DataFrame(scaled_features, columns=feature_names)
        df['target'] = self.targets[:, 0]  # Assuming binary target
        
        # Correlation matrix
        correlation_matrix = df.corr()
        
        # Visualize correlation
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Target correlations
        target_correlations = correlation_matrix['target'][:-1]
        return target_correlations
    
    def mutual_information_analysis(self):
        """
        Calculate mutual information between features and target
        
        Returns:
        - Mutual information scores
        """
        # Calculate mutual information
        mi_scores = mutual_info_regression(self.features, self.targets[:, 0])
        
        # Visualize
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(mi_scores)), mi_scores)
        plt.title('Mutual Information Scores')
        plt.xlabel('Feature Index')
        plt.ylabel('Mutual Information')
        plt.show()
        
        return mi_scores
    
    def random_forest_feature_importance(self):
        """
        Use Random Forest to determine feature importance
        
        Returns:
        - Feature importance scores
        """
        # Prepare data
        scaled_features = self.scaler.fit_transform(self.features)
        
        # Binary classification approach
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(scaled_features, self.targets[:, 0])
        
        # Feature importance
        importances = clf.feature_importances_
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.show()
        
        return importances
    
    def shap_feature_importance(self):
        """
        Use SHAP (SHapley Additive exPlanations) for feature importance
        
        Returns:
        - SHAP values and visualization
        """
        # Prepare data
        scaled_features = self.scaler.fit_transform(self.features)
        
        # Train a model for SHAP analysis
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(scaled_features, self.targets[:, 0])
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(scaled_features)
        
        # Summary plot
        shap.summary_plot(shap_values, scaled_features)
        
        # Compute mean absolute SHAP values
        mean_shap_values = np.abs(shap_values[1]).mean(axis=0)
        
        return mean_shap_values
    
    def comprehensive_feature_importance(self):
        """
        Combine multiple feature importance techniques
        
        Returns:
        - Comprehensive feature importance summary
        """
        # Run different importance analysis techniques
        correlation_importance = self.correlation_analysis()
        mi_importance = self.mutual_information_analysis()
        rf_importance = self.random_forest_feature_importance()
        shap_importance = self.shap_feature_importance()
        
        # Combine results into a DataFrame
        feature_names = [
            f'Imbalance_{i}' for i in range(4)] + \
            [f'VolumeProfile_{i}' for i in range(4)] + \
            [f'Microstructure_{i}' for i in range(4)] + \
            [f'Liquidity_{i}' for i in range(4)]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Correlation_Importance': correlation_importance,
            'Mutual_Information': mi_importance,
            'Random_Forest_Importance': rf_importance,
            'SHAP_Importance': shap_importance
        })
        
        # Normalize and aggregate
        for col in ['Correlation_Importance', 'Mutual_Information', 
                    'Random_Forest_Importance', 'SHAP_Importance']:
            importance_df[col] = (importance_df[col] - importance_df[col].min()) / \
                                  (importance_df[col].max() - importance_df[col].min())
        
        importance_df['Aggregate_Importance'] = importance_df[
            ['Correlation_Importance', 'Mutual_Information', 
             'Random_Forest_Importance', 'SHAP_Importance']
        ].mean(axis=1)
        
        # Sort by aggregate importance
        importance_df = importance_df.sort_values('Aggregate_Importance', ascending=False)
        
        # Visualize
        plt.figure(figsize=(12, 6))
        importance_df.plot(x='Feature', y='Aggregate_Importance', kind='bar')
        plt.title('Aggregate Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df

# Usage example
def validate_feature_importance(orderbook_data, targets):
    """
    Main function to validate feature importance
    
    Parameters:
    - orderbook_data: Extracted features from orderbook
    - targets: Prediction targets
    """
    # Extract features using previous implementation
    features = enhanced_orderbook_feature_extraction(orderbook_data)
    
    # Create feature importance analyzer
    analyzer = FeatureImportanceAnalyzer(features, targets)
    
    # Run comprehensive analysis
    feature_importance = analyzer.comprehensive_feature_importance()
    
    return feature_importance

# Example usage
sample_orderbook_data = [
    {
        'buy_orders': {100.5: 1000, 100.4: 800, 100.3: 600},
        'sell_orders': {100.6: 1200, 100.7: 900, 100.8: 700},
        'trade_data': [
            {'price': 100.55, 'volume': 50, 'side': 'buy'},
            {'price': 100.65, 'volume': 45, 'side': 'sell'}
        ]
    }
    # More snapshots...
]

# Simulated targets (binary price direction)
targets = np.random.randint(0, 2, (len(sample_orderbook_data), 2))

# Validate feature importance
importance_results = validate_feature_importance(sample_orderbook_data, targets)