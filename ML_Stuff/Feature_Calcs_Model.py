import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FuturesOrderbookPredictor:
    def __init__(self, look_back_periods=50, prediction_horizon=10):
        self.look_back_periods = look_back_periods
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()

    def extract_orderbook_features(self, orderbook_data):
        """
        Extract comprehensive orderbook features
        
        Features include:
        1. Weighted Orderbook Imbalance
        2. Extreme Orderbook Imbalance Indicators
        3. Orderbook Imbalance Momentum
        4. Imbalance Change Rate
        5. Mid-Price Resistance Indicators
        """
        features = []
        
        # Weighted Orderbook Imbalance over multiple depth levels
        def calculate_weighted_imbalance(buy_orders, sell_orders, depth_levels=[1, 3, 5, 10]):
            imbalances = []
            for depth in depth_levels:
                top_buy = sorted(buy_orders, reverse=True)[:depth]
                top_sell = sorted(sell_orders)[:depth]
                
                buy_volume = sum(vol * (0.9 ** i) for i, vol in enumerate(top_buy))
                sell_volume = sum(vol * (0.9 ** i) for i, vol in enumerate(top_sell))
                
                imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
                imbalances.append(imbalance)
            
            return imbalances
        
        # Extreme Imbalance Detection
        def detect_extreme_imbalance(imbalances, threshold=0.8):
            return [
                1 if abs(imb) > threshold else 0 
                for imb in imbalances
            ]
        
        # Imbalance Change over Trades
        def calculate_imbalance_momentum(historical_imbalances, window=5):
            if len(historical_imbalances) < window:
                return [0] * window
            
            momentum = []
            for i in range(window):
                change_rate = (
                    historical_imbalances[-1] - historical_imbalances[-(i+2)]
                ) / (i + 1)
                momentum.append(change_rate)
            
            return momentum
        
        # Mid-Price Resistance Feature
        def mid_price_resistance(imbalances, mid_prices, resistance_threshold=0.1):
            resistances = []
            for imb, mid_price in zip(imbalances, mid_prices):
                # Check if imbalance fails to move mid-price
                resistance_score = 1 if abs(imb) > resistance_threshold else 0
                resistances.append(resistance_score)
            
            return resistances
        
        # Comprehensive Feature Extraction
        for i in range(len(orderbook_data)):
            current_orderbook = orderbook_data[i]
            
            imbalances = calculate_weighted_imbalance(
                current_orderbook['buy_orders'], 
                current_orderbook['sell_orders']
            )
            
            extreme_imbalance_flags = detect_extreme_imbalance(imbalances)
            
            # Assume we have historical context
            imbalance_momentum = calculate_imbalance_momentum(
                current_orderbook.get('historical_imbalances', [])
            )
            
            mid_price_resistance_features = mid_price_resistance(
                imbalances, 
                current_orderbook.get('mid_prices', [])
            )
            
            # Combine all features
            comprehensive_features = (
                imbalances + 
                extreme_imbalance_flags + 
                imbalance_momentum + 
                mid_price_resistance_features
            )
            
            features.append(comprehensive_features)
        
        return np.array(features)
    
    def prepare_model_input(self, orderbook_data):
        """
        Prepare input data for deep learning model
        """
        features = self.extract_orderbook_features(orderbook_data)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(features) - self.look_back_periods - self.prediction_horizon):
            X.append(features[i:i+self.look_back_periods])
            
            # Predict future price direction and duration
            future_segment = features[i+self.look_back_periods:i+self.look_back_periods+self.prediction_horizon]
            
            # Example target: binary price direction and duration
            price_direction = 1 if np.mean(future_segment) > 0 else 0
            price_duration = len([x for x in future_segment if x > 0])
            
            y.append([price_direction, price_duration])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Create LSTM-based deep learning model
        """
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(2, activation='sigmoid')  # Binary classification + duration
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, orderbook_data):
        """
        Train the deep learning model
        """
        X, y = self.prepare_model_input(orderbook_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        
        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)
        
        # Build and train model
        self.model = self.build_model(X_train.shape[1:])
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=50,
            batch_size=32
        )
        
        return history
    
    def predict(self, new_orderbook_data):
        """
        Make predictions on new orderbook data
        """
        features = self.extract_orderbook_features(new_orderbook_data)
        
        # Prepare and scale features
        X_new = features[-self.look_back_periods:].reshape(1, *features[-self.look_back_periods:].shape)
        X_new_scaled = self.scaler.transform(
            X_new.reshape(-1, X_new.shape[-1])
        ).reshape(X_new.shape)
        
        # Predict
        prediction = self.model.predict(X_new_scaled)
        
        return {
            'price_direction': 'Up' if prediction[0][0] > 0.5 else 'Down',
            'expected_duration': prediction[0][1] * self.prediction_horizon
        }

# Example Usage
if __name__ == "__main__":
    # Simulated orderbook data
    sample_orderbook_data = [
        {
            'buy_orders': [1000, 800, 600, 400, 200],
            'sell_orders': [1200, 900, 700, 500, 300],
            'historical_imbalances': [0.2, 0.3, 0.1],
            'mid_prices': [100.5, 100.6, 100.7]
        }
        # ... more orderbook snapshots
    ]
    
    predictor = FuturesOrderbookPredictor()
    predictor.train(sample_orderbook_data)
    
    # Make a prediction
    new_data = sample_orderbook_data[-50:]  # Last 50 snapshots
    prediction = predictor.predict(new_data)
    print(prediction)