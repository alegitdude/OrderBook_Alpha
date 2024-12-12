import numpy as np
import scipy.stats as stats

def enhanced_orderbook_feature_extraction(orderbook_data):
    """
    Comprehensive feature extraction including:
    1. Volume Profile Metrics
    2. Market Microstructure Indicators
    3. Liquidity Metrics
    4. Advanced Orderbook Imbalance Measurements
    """
    features = []

    def calculate_volume_profile(buy_orders, sell_orders, price_levels=10):
        """
        Generate volume profile features
        """
        # Organize orders by price
        sorted_buy = sorted(buy_orders.items(), reverse=True)
        sorted_sell = sorted(sell_orders.items())

        # Volume concentration metrics
        volume_profile_features = []

        # Market Value of Depth (MVD)
        mvd_buy = sum(price * volume for price, volume in sorted_buy[:price_levels])
        mvd_sell = sum(price * volume for price, volume in sorted_sell[:price_levels])
        volume_profile_features.extend([mvd_buy, mvd_sell])

        # Volume Concentration Ratio
        total_buy_volume = sum(volume for _, volume in sorted_buy)
        total_sell_volume = sum(volume for _, volume in sorted_sell)
        top_levels_buy_volume = sum(volume for _, volume in sorted_buy[:3])
        top_levels_sell_volume = sum(volume for _, volume in sorted_sell[:3])
        
        volume_concentration_buy = top_levels_buy_volume / total_buy_volume
        volume_concentration_sell = top_levels_sell_volume / total_sell_volume
        volume_profile_features.extend([volume_concentration_buy, volume_concentration_sell])

        return volume_profile_features

    def calculate_market_microstructure_indicators(buy_orders, sell_orders, trade_data):
        """
        Market microstructure indicators
        """
        microstructure_features = []

        # Order Book Slope
        buy_prices = [price for price, _ in sorted(buy_orders.items(), reverse=True)]
        sell_prices = [price for price, _ in sorted(sell_orders.items())]
        
        # Linear regression of order book depth
        buy_slope, _ = np.polyfit(range(len(buy_prices)), buy_prices, 1)
        sell_slope, _ = np.polyfit(range(len(sell_prices)), sell_prices, 1)
        microstructure_features.extend([buy_slope, sell_slope])

        # Bid-Ask Spread
        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())
        spread = best_ask - best_bid
        relative_spread = spread / ((best_bid + best_ask) / 2)
        microstructure_features.extend([spread, relative_spread])

        # Trade Aggression Ratio
        if trade_data:
            buy_aggressive_trades = sum(1 for trade in trade_data if trade['side'] == 'buy' and trade['price'] >= best_ask)
            sell_aggressive_trades = sum(1 for trade in trade_data if trade['side'] == 'sell' and trade['price'] <= best_bid)
            total_trades = len(trade_data)
            
            aggression_ratio = (buy_aggressive_trades + sell_aggressive_trades) / total_trades if total_trades > 0 else 0
            microstructure_features.append(aggression_ratio)

        return microstructure_features

    def calculate_liquidity_metrics(buy_orders, sell_orders):
        """
        Comprehensive liquidity metrics
        """
        liquidity_features = []

        # Total Depth
        total_buy_depth = sum(buy_orders.values())
        total_sell_depth = sum(sell_orders.values())
        liquidity_features.extend([total_buy_depth, total_sell_depth])

        # Market Impact Depth
        # Estimate volume needed to move price by X%
        def calculate_market_impact_depth(orders, impact_percentage=0.01):
            cumulative_volume = 0
            total_value = sum(price * volume for price, volume in orders.items())
            for price, volume in sorted(orders.items()):
                cumulative_volume += volume
                if cumulative_volume / total_value >= impact_percentage:
                    return cumulative_volume
            return cumulative_volume

        buy_market_impact = calculate_market_impact_depth(buy_orders)
        sell_market_impact = calculate_market_impact_depth(sell_orders)
        liquidity_features.extend([buy_market_impact, sell_market_impact])

        # Liquidity Concentration (Herfindahl-Hirschman Index variant)
        def calculate_liquidity_concentration(orders):
            total_volume = sum(orders.values())
            concentration = sum((volume / total_volume) ** 2 for volume in orders.values())
            return concentration

        buy_concentration = calculate_liquidity_concentration(buy_orders)
        sell_concentration = calculate_liquidity_concentration(sell_orders)
        liquidity_features.extend([buy_concentration, sell_concentration])

        return liquidity_features

    # Process each orderbook snapshot
    for snapshot in orderbook_data:
        # Extract core data
        buy_orders = snapshot.get('buy_orders', {})
        sell_orders = snapshot.get('sell_orders', {})
        trade_data = snapshot.get('trade_data', [])

        # Combine all feature extraction methods
        snapshot_features = (
            # Orderbook Imbalance (from previous implementation)
            calculate_weighted_imbalance(buy_orders, sell_orders) +
            
            # Volume Profile Metrics
            calculate_volume_profile(buy_orders, sell_orders) +
            
            # Market Microstructure Indicators
            calculate_market_microstructure_indicators(buy_orders, sell_orders, trade_data) +
            
            # Liquidity Metrics
            calculate_liquidity_metrics(buy_orders, sell_orders)
        )

        features.append(snapshot_features)

    return np.array(features)

# Example usage within the previous FuturesOrderbookPredictor
class EnhancedFuturesOrderbookPredictor(FuturesOrderbookPredictor):
    def extract_orderbook_features(self, orderbook_data):
        """
        Override the feature extraction method with enhanced version
        """
        return enhanced_orderbook_feature_extraction(orderbook_data)

# Sample data structure
sample_orderbook_data = [
    {
        'buy_orders': {
            100.5: 1000,
            100.4: 800,
            100.3: 600
        },
        'sell_orders': {
            100.6: 1200,
            100.7: 900,
            100.8: 700
        },
        'trade_data': [
            {'price': 100.55, 'volume': 50, 'side': 'buy'},
            {'price': 100.65, 'volume': 45, 'side': 'sell'}
        ]
    }
    # More snapshots...
]