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

from BS.BS_generator import Black_Scholes
from BS.BS_util import RNN_BN_simple, loss_exp_OCE
from Deephedging import DeepHedging


def create_objective(stock_prices, asset_hedged, assets_to_hedge, n_trials_per_study=10):
    """
    This function creates an objective function for hyperparameter optimization for the BS pipeline. (necessary for Optuna).
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
    
    train_stock_prices = stock_prices.iloc[:int(0.7*stock_prices.shape[0]),:]
    test_stock_prices = stock_prices.iloc[int(0.7*stock_prices.shape[0]):,:]
    #train_stock_returns = np.log(train_stock_prices / train_stock_prices.shift(1)).dropna()
    
    # Data generation parameters
    sequence_length = 30
    dt = 1/252
    K = 100
    S0 = 100
    train_size = 10000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    data_generator = Black_Scholes(sequence_length, dt, train_stock_prices, test_stock_prices, K, S0, train_size, idx_asset_hedged, idx_assets_to_hedge)
    
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
        regime_embed_dim = trial.suggest_int('regime_embed_dim', 2, 16)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1000, 2000, 5000, 10000])
        bn_eps = trial.suggest_float('bn_eps', 1e-5, 1e-2, log=True)
        bn_momentum = trial.suggest_float('bn_momentum', 0.01, 0.5)
        
        # Number of assets for hedging
        number_assets = len(assets_to_hedge)
        
        # Apply the trial parameters to the RNN
        network = RNN_BN_simple(
            price_input_size=number_assets,
            n_regimes=data_generator.n_regimes,
            sequence_length=sequence_length,
            device=device,
            regime_embed_dim=regime_embed_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum
        ).to(device)
        
        epoch_num = n_trials_per_study
        batch_num = 20
        
        prices_train_data, regimes_train_data = data_generator.build_data(idx_assets_to_hedge)
        
        # Training setup
        T = dt * sequence_length
        sigma = data_generator.sigma
        loss_fn = loss_exp_OCE(K, sigma, T, 1.3, idx_asset_hedged, X_max=True).to(device)
        
        opt = torch.optim.Adam([
            {'params': network.parameters()},
        ], lr=learning_rate)
        
        # Training loop with reduced epochs for faster optimization
        N = 10000
        num_partitions = min(int(1e5/N), len(prices_train_data) // N)
        
        total_train_loss = 0.0
        iterations = 0
        
        for part in range(num_partitions):
            index1 = part * N
            index2 = (part + 1) * N
            
            price_batch = prices_train_data[index1:index2]
            regime_batch = regimes_train_data[index1:index2]
            train_data = torch.utils.data.TensorDataset(price_batch, regime_batch)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epoch_num):
                network.train()
                epoch_loss = 0.0
                
                for batch in train_loader:
                    price, regimes = batch
                    price = price.to(device)
                    regimes = regimes.to(device)
                    
                    price_feature = torch.log(price[:, :-1])
                    holding = network(price_feature, regimes[:, :-1])
                    loss = loss_fn(holding, price)
                    
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
        
        avg_train_loss = total_train_loss / iterations
        
        # Validation on beggining of market data TODO

        val_data = data_generator.make_val_data(idx_assets_to_hedge)
        price_val, regimes_val = val_data
        price_val = price_val.to(device)
        regimes_val = regimes_val.to(device)
        
        network.eval()
        with torch.no_grad():
            eps = 1e-6
            returns = torch.log((price_val[:, 1:, :] + eps) / (price_val[:, :-1, :] + eps))
            holding = network(returns, regimes_val[:, :-1])
            val_loss = loss_fn(holding, price_val).item()
        
        print(f"Trial {trial.number}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return val_loss
    
    return objective


def run_optimization(stock_prices, asset_hedged='AAPL', assets_to_hedge=['AAPL', 'GOOGL'],
                    n_trials=50, n_trials_per_study=10, study_name='deep_hedging_rnn'):
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
    
    print(f"Starting Optuna optimization for {asset_hedged} hedging")
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
    

def get_BS_study_name(asset_hedged, assets_to_hedge):
    """
    Generate study name for BS model cross-validation.
    
    Args:
        asset_hedged (str): Name of the asset being hedged
        assets_to_hedge (list): List of assets used for hedging
    
    Returns:
        str: Study name in format 'deep_hedging_rnn_bs_ASSET_HEDGE1_HEDGE2_...'
    """
    study_name = f'deep_hedging_rnn_bs_{asset_hedged}'
    for ticker in assets_to_hedge:
        study_name += "_" + str(ticker)
    return study_name


def run_BS_crossval(asset_hedged, assets_to_hedge, stock_prices_path, n_trials=3, n_trials_per_study=3):
    """
    Run cross-validation for BS deep hedging model.
    
    Args:
        asset_hedged (str): Name of the asset being hedged
        assets_to_hedge (list): List of assets used for hedging
        stock_prices_path (str): Path to stock prices CSV file
        n_trials (int): Number of different hyperparameter combinations to try
        n_trials_per_study (int): Number of epochs per trial
    """
    data_path = parent_dir / 'Data' / 'stocks_close_prices_2008_2025.csv'
    stock_prices = pd.read_csv(data_path)
    stock_prices.set_index('Date', inplace=True)
    study_name = get_BS_study_name(asset_hedged, assets_to_hedge)
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
