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

from Diffusion.Diffusion_generator import diffusion_data
from Diffusion.Diffusion_util import RNN_BN_simple, loss_exp_OCE


def create_objective(stock_prices, asset_hedged, assets_to_hedge, n_trials_per_study=10):
    """
    This function creates an objective function for hyperparameter optimization for the Diffusion pipeline. (necessary for Optuna).
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
    
    # Data generation parameters
    sequence_length = 30
    dt = 1/365
    K = 100
    S0 = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    data_generator = diffusion_data(sequence_length, dt, K, S0, idx_assets_to_hedge)
    
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
        batch_size = trial.suggest_categorical('batch_size', [1000, 2000, 5000, 10000])
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
        
        # Build data from diffusion generator
        train_data = data_generator.build_data()
        val_data_full, val_returns_full = data_generator.make_val_data()
        
        val_returns = val_returns_full
        
        # Convert to tensors
        train_returns = torch.tensor(train_data, dtype=torch.float32)
        val_returns_tensor = torch.tensor(val_returns, dtype=torch.float32)
        
        
        # Training setup
        T = dt * sequence_length
        loss_fn = loss_exp_OCE(K, T, 1.3, idx_asset_hedged, X_max=True).to(device)
        
        opt = torch.optim.Adam([
            {'params': network.parameters()}
        ], lr=learning_rate)
        
        # Helper function to reconstruct prices from log returns
        def prices_from_log_returns(r, s0=100.0):
            B, T, N = r.shape
            s0_vec = torch.full((B, 1, N), s0, device=r.device)
            logS = torch.cumsum(r, dim=1)          # (B, T, N)
            S = s0_vec * torch.exp(logS)           # (B, T, N)
            return torch.cat([s0_vec, S], dim=1)   # (B, T+1, N)
        
        # Training loop
        train_dataset = torch.utils.data.TensorDataset(train_returns)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation loader
        val_dataset = torch.utils.data.TensorDataset(val_returns_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        total_train_loss = 0.0
        iterations = 0
        
        for epoch in range(epoch_num):
            network.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                scaled_returns = batch[0].to(device)
                
                if scaled_returns.ndim == 2:
                    scaled_returns = scaled_returns.unsqueeze(-1)
                
                # Build prices from log-returns
                prices = prices_from_log_returns(scaled_returns, s0=S0)
                
                # Get holdings from network
                holding = network(scaled_returns)
                loss = loss_fn(holding, prices)
                
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
        
        avg_train_loss = total_train_loss / max(iterations, 1)
        
        # Validation
        network.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                scaled_returns = batch[0].to(device)
                
                if scaled_returns.ndim == 2:
                    scaled_returns = scaled_returns.unsqueeze(-1)
                
                # Build prices from log-returns
                prices = prices_from_log_returns(scaled_returns, s0=S0)
                
                # Get holdings from network
                holding = network(scaled_returns)
                loss = loss_fn(holding, prices)
                val_loss_total += loss.item()
            
            val_loss = val_loss_total / len(val_loader)
        
        print(f"Trial {trial.number}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return val_loss
    
    return objective


def run_optimization(stock_prices, asset_hedged='AAPL', assets_to_hedge=['AAPL', 'GOOGL'],
                    n_trials=50, n_trials_per_study=10, study_name='deep_hedging_rnn_diffusion'):
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
    
    print(f"Starting Optuna optimization for {asset_hedged} hedging with Diffusion")
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
    

def get_diffusion_study_name(asset_hedged, assets_to_hedge):
    """
    Generate study name for Diffusion model cross-validation.
    
    Args:
        asset_hedged (str): Name of the asset being hedged
        assets_to_hedge (list): List of assets used for hedging
    
    Returns:
        str: Study name in format 'deep_hedging_rnn_diffusion_ASSET_HEDGE1_HEDGE2_...'
    """
    study_name = f'deep_hedging_rnn_diffusion_{asset_hedged}'
    for ticker in assets_to_hedge:
        study_name += "_" + str(ticker)
    return study_name


def run_diffusion_crossval(asset_hedged, assets_to_hedge, stock_prices_path, n_trials=3, n_trials_per_study=3):
    """
    Run cross-validation for Diffusion deep hedging model.
    
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
    study_name = get_diffusion_study_name(asset_hedged, assets_to_hedge)
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
    run_diffusion_crossval('AAPL', ['AAPL', 'GOOGL'], stock_prices_path='stocks_close_prices_2008_2025.csv')
