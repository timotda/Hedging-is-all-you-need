import torch
import numpy as np
import pandas as pd

class market_data():

    def __init__(self, sequence_length, dt, train_stock_returns, val_stock_prices, test_stock_prices, K, S0, idx_assets_to_hedge):

        self.sequence_length = sequence_length
        self.dt = dt 

        self.T = self.sequence_length*self.dt
        self.K = K
        self.S0 = S0
        self.train_stock_returns = train_stock_returns
        self.val_stock_prices = val_stock_prices
        self.test_stock_prices = test_stock_prices
        self.idx_assets_to_hedge = idx_assets_to_hedge

    def build_data(self):
        """
        Build training data using rolling windows from historical returns.
        Returns normalized price paths from the training period.
        """
        # Get returns for assets to hedge
        returns = self.train_stock_returns.iloc[:, self.idx_assets_to_hedge]
        
        # Convert returns to prices starting from S0
        # Cumulative product of (1 + return) gives price evolution
        price_data = self.S0 * (1 + returns).cumprod()
        
        # Create rolling windows
        T = self.sequence_length
        price_paths = []
        
        for start in range(0, len(price_data) - T):
            end = start + T
            # Extract window
            S_window = price_data.iloc[start:end].values
            # Normalize to start at S0
            S_norm = S_window / S_window[0] * self.S0
            price_paths.append(S_norm)
        
        # Convert to numpy array (num_windows, T, num_assets)
        train_data = np.stack(price_paths, axis=0)
        
        # Build validation and test data using same approach
        val_data = self.make_validation_data()
        test_data = self.make_test_data()

        return train_data, test_data, val_data

    def make_validation_data(self):
        """
        Build validation set from real price data using rolling windows.
        Returns:
            prices_val -> (batch, T, N) where N is number of assets
        """
        # Get validation prices for assets to hedge
        asset_prices = self.val_stock_prices.iloc[:, self.idx_assets_to_hedge]
        
        T = self.sequence_length
        price_paths = []
        
        for start in range(0, len(asset_prices) - T):
            end = start + T
            # Extract price window
            S_window = asset_prices.iloc[start:end].values
            # Normalize path to start at S0 (matches training normalization)
            S_norm = S_window / S_window[0] * self.S0
            price_paths.append(S_norm)
        
        # Convert to numpy array (num_windows, T, num_assets)
        prices_val = np.stack(price_paths, axis=0)
        
        return prices_val
    
    def make_test_data(self):
        """
        Build test set from real price data using rolling windows.
        Returns:
            prices_test -> (batch, T, N) where N is number of assets
        """
        # Get test prices for assets to hedge
        asset_prices = self.test_stock_prices.iloc[:, self.idx_assets_to_hedge]
        
        T = self.sequence_length
        price_paths = []
        
        for start in range(0, len(asset_prices) - T):
            end = start + T
            # Extract price window
            S_window = asset_prices.iloc[start:end].values
            # Normalize path to start at S0 (matches training normalization)
            S_norm = S_window / S_window[0] * self.S0
            price_paths.append(S_norm)
        
        # Convert to numpy array (num_windows, T, num_assets)
        prices_test = np.stack(price_paths, axis=0)
        
        return prices_test
        
