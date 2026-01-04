import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import torch
import torch.nn as nn
import optuna
import numpy as np
import pandas as pd
from optuna.visualization import plot_optimization_history, plot_param_importances
import time

from MarketData.Market_data_generator import market_data
from MarketData.Market_data_util import RNN_BN_simple, loss_exp_OCE


def create_objective(stock_prices, asset_hedged, assets_to_hedge, n_trials_per_study=10):
    """
    This function creates an objective function for hyperparameter optimization for the MarketData pipeline. (necessary for Optuna).
    The objective function that is returned will be called by Optuna with different hyperparameter combinations.    
    Args:
        stock_prices: DataFrame with stock price data
        asset_hedged: String name of asset to hedge (e.g., 'AAPL')
        assets_to_hedge: List of asset names to use for hedging
        n_trials_per_study: Number of training iterations per trial
    
    Returns:
        objective: Function that Optuna will optimize
    """
    
    # Setup data
    assets = ['AAPL','GOOGL','MSFT','AMZN','BRK-B']
    idx_asset_hedged = assets.index(asset_hedged)
    idx_assets_to_hedge = [assets.index(asset) for asset in assets_to_hedge]
    
    # split the data : 80% training and validation, 20% for testing
    # split the 80% in training (80%) and validation (20%)
    n_samples = stock_prices.shape[0]
    train_val_end = int(0.8 * n_samples)

    
    train_val_stock_prices = stock_prices.iloc[:train_val_end,:]
    test_stock_prices = stock_prices.iloc[train_val_end:,:]

    # now split the train_val into train and val
    val_end = int(0.2 * train_val_stock_prices.shape[0])
    train_stock_prices = train_val_stock_prices.iloc[val_end:,:]
    val_stock_prices = train_val_stock_prices.iloc[:val_end,:]
    train_stock_returns = np.log(train_stock_prices / train_stock_prices.shift(1)).dropna()
    # Data generation parameters
    sequence_length = 30
    dt = 1/365
    K = 100
    S0 = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    data_generator = market_data(sequence_length, dt, train_stock_returns, val_stock_prices, test_stock_prices, K, S0, idx_assets_to_hedge)
    
    def objective(trial):
        """
        Objective function for a single Optuna trial.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            val_loss: Validation loss (lower is better)
        """
        
        # Suggest hyperparameters
        hidden_dim_1 = trial.suggest_int('hidden_dim_1', 10, 64, step=2)
        hidden_dim_2 = trial.suggest_int('hidden_dim_2', 10, 64, step=2)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [100, 200, 500, 1000])
        bn_eps = trial.suggest_float('bn_eps', 1e-5, 1e-2, log=True)
        bn_momentum = trial.suggest_float('bn_momentum', 0.01, 0.5)
        
        # Number of assets for hedging
        number_assets = len(assets_to_hedge)
        
        # Apply the trial parameters to the RNN
        network = RNN_BN_simple(
            price_input_size=number_assets,
            sequence_length=sequence_length,
            device=device,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum
        ).to(device)
        
        epoch_num = n_trials_per_study
        
        prices_train_data, prices_test_data, _ = data_generator.build_data()
        
        # Training setup
        T = dt * sequence_length
        loss_fn = loss_exp_OCE(K, T, 1.3, idx_asset_hedged, X_max=True).to(device)
        
        opt = torch.optim.Adam([
            {'params': network.parameters()}
        ], lr=learning_rate)
        
        # Training loop with reduced epochs for faster optimization
        # Convert all training data to tensor
        price_tensor = torch.tensor(prices_train_data, dtype=torch.float32)
        train_data = torch.utils.data.TensorDataset(price_tensor)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        total_train_loss = 0.0
        iterations = 0
        
        for epoch in range(epoch_num):
            network.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                price = batch[0].to(device)
                
                # Log returns as features
                eps = 1e-6
                price_feature = torch.log((price[:, 1:, :] + eps) / (price[:, :-1, :] + eps))
                holding = network(price_feature)
                loss = loss_fn(holding, price, mode='train')
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            total_train_loss += epoch_loss
            iterations += 1
            
            # Report intermediate results for pruning
            trial.report(epoch_loss, iterations)
            
            # Handle pruning based on intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        avg_train_loss = total_train_loss / max(iterations, 1)  # Avoid division by zero
        
        # Validation on test data
        price_test = torch.tensor(prices_test_data, dtype=torch.float32).to(device)
        
        network.eval()
        with torch.no_grad():
            eps = 1e-6
            returns = torch.log((price_test[:, 1:, :] + eps) / (price_test[:, :-1, :] + eps))
            holding = network(returns)
            val_loss = loss_fn(holding, price_test, mode='test').item()
        
        print(f"Trial {trial.number}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return val_loss
    
    return objective


def run_optimization(stock_prices, asset_hedged='AAPL', assets_to_hedge=['AAPL', 'GOOGL'],
                    n_trials=50, n_trials_per_study=10, study_name='deep_hedging_rnn_marketdata'):
    """
    Run Optuna hyperparameter optimization study.
    
    Args:
        stock_prices: DataFrame with stock price data
        asset_hedged: Asset to hedge
        assets_to_hedge: List of assets to use for hedging
        n_trials: Number of Optuna trials to run
        n_trials_per_study: Number of epochs per trial
        study_name: Name of the Optuna study
    
    Returns:
        study: Optuna study object with results
    """
    
    print(f"Starting Optuna optimization for {asset_hedged} hedging with Market Data")
    print(f"Hedging with: {assets_to_hedge}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per trial: {n_trials_per_study}")
    print("=" * 60)
    
    objective = create_objective(stock_prices, asset_hedged, assets_to_hedge, n_trials_per_study)
    
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run optimization
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed_time = time.time() - start_time
    
    print("=" * 60)
    print(f"Optimization completed in {elapsed_time/60:.2f} minutes")
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study


def save_study_results(study, save_path=None):
    """
    Save Optuna study results and visualizations.
    
    Args:
        study: Optuna study object
        save_path: Directory to save results (if None, uses default path)
    """
    import os
    
    # If save_path is not provided, construct the default path
    if save_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, 'Cross_validation_results')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save study to database
    study_path = os.path.join(save_path, f'{study.study_name}.db')
    print(f"\nSaving study to: {study_path}")
    
    # Save best parameters to CSV
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df['best_value'] = study.best_value
    best_params_df['best_trial'] = study.best_trial.number
    csv_path = os.path.join(save_path, f'{study.study_name}_best_params.csv')
    best_params_df.to_csv(csv_path, index=False)
    print(f"Best parameters saved to: {csv_path}")
    

def get_marketdata_study_name(asset_hedged, assets_to_hedge):
    study_name = f'deep_hedging_rnn_marketdata_{asset_hedged}'
    for ticker in assets_to_hedge:
        study_name += "_" + str(ticker)
    return study_name


def run_marketdata_crossval(asset_hedged, assets_to_hedge, stock_prices_path, n_trials=3, n_trials_per_study=3):

    data_path = parent_dir / 'Data' / 'stocks_close_prices_2008_2025.csv'
    stock_prices = pd.read_csv(data_path)
    stock_prices.set_index('Date', inplace=True)
    study_name = get_marketdata_study_name(asset_hedged, assets_to_hedge)
    # Run optimization study
    study = run_optimization(
        stock_prices=stock_prices,
        asset_hedged=asset_hedged,
        assets_to_hedge=assets_to_hedge,
        n_trials=n_trials,  # Number of different hyperparameter combinations to try
        n_trials_per_study=n_trials_per_study,  # Number of epochs per trial
        study_name=study_name
    )
    save_study_results(study)

if __name__ == "__main__":
    # run this to test the cross_validation
    run_marketdata_crossval('AAPL', ['AAPL', 'GOOGL'], stock_prices_path='stocks_close_prices_2008_2025.csv')
    
