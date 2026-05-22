import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import StockPredictionNN
from data_processor import load_and_preprocess_orders
class StockPredictor:
    def __init__(self, window_size=5, hidden_size=32, num_layers=1):
        self.window_size = window_size
        # Inputs: first derivative (velocity) and second derivative (acceleration)
        self.input_size = 2 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize the neural network
        self.model = StockPredictionNN(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=1, 
            num_layers=self.num_layers
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def calculate_differentials(self, series):
        """
        Calculates the first (velocity) and second (acceleration) derivatives.
        Instead of predicting based on raw historical data, we predict based on
        how the usage is changing.
        """
        # First derivative: rate of change of sales
        dy = np.gradient(series)
        # Second derivative: acceleration of sales
        d2y = np.gradient(dy)
        return dy, d2y

    def calculate_moving_average_mean(self, series, window):
        """
        Calculates the moving average mean for short-term prediction insight.
        """
        return pd.Series(series).rolling(window=window).mean().bfill().values

    def prepare_data(self, sales_data):
        """
        Prepares sequences of differentials to feed into the LSTM.
        """
        dy, d2y = self.calculate_differentials(sales_data)
        
        # Combine differentials into features
        features = np.column_stack((dy, d2y))
        
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            # Target is the next day's differential (velocity)
            y.append(dy[i+self.window_size])
            
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def train(self, sales_data, epochs=100):
        """
        Trains the neural network on the differentials.
        """
        X_train, y_train = self.prepare_data(sales_data)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, recent_sales_data):
        """
        Predicts future stock usage using a combination of Neural Network (differentials)
        and Moving Average Mean.
        """
        self.model.eval()
        
        # 1. Moving Average prediction (Short-term trend of raw data)
        ma_prediction = np.mean(recent_sales_data[-self.window_size:])
        
        # 2. Neural Network prediction based on differentials
        dy, d2y = self.calculate_differentials(recent_sales_data)
        recent_features = np.column_stack((dy, d2y))[-self.window_size:]
        recent_tensor = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            predicted_dy = self.model(recent_tensor).item()
        
        # Reconstruct the predicted raw value using the predicted differential
        last_actual_value = recent_sales_data[-1]
        nn_prediction = last_actual_value + predicted_dy
        
        # Holistic approach: Combine Neural Network (complex pattern) and Moving Average (smooth trend)
        # Weight them: e.g., 60% NN, 40% MA
        final_prediction = (0.6 * nn_prediction) + (0.4 * ma_prediction)
        
        return {
            'moving_average_prediction': ma_prediction,
            'neural_network_predicted_value': nn_prediction,
            'predicted_differential': predicted_dy,
            'final_holistic_prediction': final_prediction
        }

def generate_sample_data(days=100):
    """
    Generates synthetic time-series sales data.
    Simulates base sales with some trend and seasonality.
    """
    time = np.arange(days)
    base_sales = 50
    trend = time * 0.5
    seasonality = 10 * np.sin(2 * np.pi * time / 7) # Weekly seasonality
    noise = np.random.normal(0, 5, days)
    
    sales = base_sales + trend + seasonality + noise
    return np.maximum(sales, 0) # Sales can't be negative

if __name__ == "__main__":
    sales_data = None
    
    # Try loading real data if orders.csv exists
    if os.path.exists('orders.csv'):
        print("Found 'orders.csv'! Processing raw orders data...")
        daily, monthly = load_and_preprocess_orders('orders.csv')
        if daily is not None and len(daily) > 10:
            sales_data = daily
            print(f"Successfully loaded {len(daily)} days of real sales data.")
        else:
            print("Failed to load sufficient real data. Falling back to synthetic.")
    
    if sales_data is None:
        print("Generating sample sales data...")
        sales_data = generate_sample_data(days=150)
    
    # Use the first 80% for training, last 20% for testing
    split_idx = int(len(sales_data) * 0.8)
    train_data = sales_data[:split_idx]
    test_data = sales_data[split_idx:]
    
    print("\nInitializing Stock Predictor...")
    predictor = StockPredictor(window_size=7)
    
    print("\nTraining Neural Network on Differentials...")
    predictor.train(train_data, epochs=100)
    
    print("\nTesting Predictions...")
    # Let's predict the next few days using a sliding window over the test data
    test_steps = min(5, len(test_data))
    for i in range(test_steps):
        current_window = sales_data[split_idx - 7 + i : split_idx + i]
        actual_next_day = sales_data[split_idx + i]
        
        prediction_results = predictor.predict(current_window)
        
        print(f"\nDay {split_idx + i + 1}:")
        print(f"  Actual Sales: {actual_next_day:.2f}")
        print(f"  Predicted Differental: {prediction_results['predicted_differential']:.2f}")
        print(f"  NN Projected Value: {prediction_results['neural_network_predicted_value']:.2f}")
        print(f"  Moving Avg Value: {prediction_results['moving_average_prediction']:.2f}")
        print(f"  Final Holistic Prediction: {prediction_results['final_holistic_prediction']:.2f}")
