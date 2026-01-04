import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from MarketData.Market_data_generator import *
import os
from Deephedging_JAX import DeepHedgingJAX
import math


def run_marketdata_deephedging(asset_hedged, stock_prices, assets_to_hedge,name_model, is_plot, plot_path):
    from Deephedging import DeepHedging
    

    assets = ['AAPL','GOOGL','MSFT','AMZN','BRK-B']
    idx_asset_hedged = assets.index(asset_hedged)
    idx_assets_to_hedge = []
    for asset in assets_to_hedge:
        idx_assets_to_hedge.append(assets.index(asset))

    # split the data : 80% training, 20% for testing and validation
    # split the 20% in testing (80%) and validation (20%)
    n_samples = int(0.7*stock_prices.shape[0])
    train_stock_prices = stock_prices.iloc[:n_samples,:]

    test_val_stock_prices = stock_prices.iloc[n_samples:,:]
    val_end = int(0.2*test_val_stock_prices.shape[0])
    val_stock_prices = test_val_stock_prices.iloc[:val_end,:]
    test_stock_prices = test_val_stock_prices.iloc[val_end:,:]

    train_stock_returns = np.log(train_stock_prices / train_stock_prices.shift(1)).dropna()
    # begin by initialising parameters for data generation
    sequence_length=30
    dt = 1/252
    #or dt = 1/365?
    K = 100
    S0 = 100 

    # Get parameters from Optuna cross-validation if it has been run, otherwise use defaults
    params = get_parameters(asset_hedged, assets_to_hedge, model_type='MarketData')
    network_params = params['network_params']
    training_params = params['training_params']
    
    # define training parameters
    # (learning_rate, batch_size, batch_num, epoch_num)
    training_parameters = (
        training_params['learning_rate'],
        training_params['batch_size'],
        training_params['batch_num'],
        training_params['epoch_num']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    # create the data generator object
    data_generator = market_data(sequence_length, dt, train_stock_returns, val_stock_prices, test_stock_prices, K,S0,idx_assets_to_hedge )
    # define the number of assets used for hedging
    number_assets = len(assets_to_hedge)
    # define the neural network architecture
    network = RNN_BN_simple(
        number_assets, 
        sequence_length, 
        device,
        hidden_dim_1=network_params['hidden_dim_1'], 
        hidden_dim_2=network_params['hidden_dim_2'],
        bn_eps=network_params['bn_eps'],
        bn_momentum=network_params['bn_momentum']
    ).to(device)

    # define the path where we will save the model 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)

    if name_model == 'RNN_BN_simple':
        name = "RNN_BN_simple"
        for asset in assets_to_hedge:
            name +=  f"_{asset}"
        deephedging = DeepHedging(data_generator, number_assets, idx_asset_hedged, network, device, training_parameters, name,is_plot, plot_path, save_path=save_dir)
        deephedging.get_data()
        deephedging.train_MarketData()
        deephedging.test()
    elif name_model == 'SigFormer':
        config = load_config()
        parameters = config['SigFormer']
        in_dim = len(assets_to_hedge)
        out_dim = len(assets_to_hedge)
        dh = DeepHedgingJAX(
            data_generator=data_generator,
            in_dim=in_dim,
            out_dim=out_dim,
            idx_asset=idx_asset_hedged,
            is_plot = is_plot,
            plot_path = plot_path, 
            model_dim=parameters['model_dim'],
            n_heads=parameters['n_heads'],
            d_ff=parameters['d_ff'],
            order=parameters['order'],
            n_blocks=parameters['n_blocks'],
            lr=float(parameters['learning_rate']),
            seed=parameters['seed'],
        )
        dh.get_data()
        dh.train(epochs=parameters['epochs'], batch_size=parameters['batch_size'])
        dh.test()

class RNN_BN_simple(nn.Module):
    """
    RNN with BatchNorm for Market Data (no regimes).
    - prices:    (batch, T, price_dim)
    """
    def __init__(self,
                 price_input_size,
                 sequence_length,
                 device,
                 hidden_dim_1=20,
                 hidden_dim_2=20,
                 bn_eps=1e-3,
                 bn_momentum=0.3
                 ):
        super().__init__()

        self.sequence_length = sequence_length
        self.device = device
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        # Total input: price features 
        self.input_size = price_input_size

        # ---- RNN with BN layers ----
        self.rnn = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(self.input_size, eps=bn_eps, momentum=bn_momentum),
                nn.Linear(self.input_size, hidden_dim_1),
                nn.BatchNorm1d(hidden_dim_1, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(),
                nn.Linear(hidden_dim_1, hidden_dim_2),
                nn.BatchNorm1d(hidden_dim_2, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(),
                nn.Linear(hidden_dim_2, price_input_size)     # output hedge for each asset
            )
            for _ in range(sequence_length)
        ])


    def forward(self, prices):
        """
        prices:  (batch, T, price_dim)
        """
        B, T, P = prices.shape
        #assert T == self.sequence_length
        x = prices
        # Transpose to (T, B)
        x = x.transpose(0, 1)

        outputs = []

        for t in range(T):
            xt = x[t]                    # shape (B, F)
            out_t = self.rnn[t](xt)      # predicted hedge at time t
            outputs.append(out_t.unsqueeze(0))

        # Back to (B, T, price_dim)
        return torch.cat(outputs, dim=0).transpose(0, 1)


def load_config():
    """
    Load configuration from config.yaml file.
    
    Returns:
        dict: Configuration dictionary from yaml file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config_path = os.path.join(parent_dir, 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}")
        return None


def get_parameters(asset_hedged, assets_to_hedge, model_type='MarketData'):
    """
    Load training and network parameters from Optuna cross-validation results if available,
    otherwise return default parameters from config.yaml.
    
    Args:
        asset_hedged (str): Name of the asset being hedged (e.g., 'AAPL', 'GOOGL')
        assets_to_hedge (list): List of assets used for hedging (e.g., ['AAPL', 'GOOGL'])
        model_type (str): Type of model ('MarketData', 'BS', 'Diffusion', etc.)
    
    Returns:
        dict: Dictionary containing network_params and training_params
    
    Raises:
        ValueError: If config.yaml is not found or doesn't contain required parameters
    """
    # Lazy import to avoid circular dependency
    from Cross_validation.optuna_hypparam_marketdata import get_marketdata_study_name
    
    # Load config.yaml
    config = load_config()
    
    # Check if config was loaded successfully
    if (config is None) or ('MarketData_RNN_Default' not in config):
        raise ValueError("Please make sure you have set up a default RNN parameter setting in config.yaml")
    
    
    
    # Extract default parameters from config.yaml
    marketdata_config = config['MarketData_RNN_Default']
    default_network_params = marketdata_config['default_network_params']
    default_training_params = marketdata_config['default_training_params']
    
    # Construct path to Optuna results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    optuna_dir = os.path.join(parent_dir, 'Cross_validation/Cross_validation_results')
    
    # Get study name using the cross-validation function
    study_name = get_marketdata_study_name(asset_hedged, assets_to_hedge)
    optuna_file = os.path.join(optuna_dir, f'{study_name}_best_params.csv')
    
    # Try to load Optuna results if the cross val code has been run before
    if os.path.exists(optuna_file):
        try:
            df = pd.read_csv(optuna_file)
            if len(df) > 0:
                row = df.iloc[0]
                
                # Extract network parameters
                network_params = {
                    'hidden_dim_1': int(row['hidden_dim_1']) if 'hidden_dim_1' in row else default_network_params['hidden_dim_1'],
                    'hidden_dim_2': int(row['hidden_dim_2']) if 'hidden_dim_2' in row else default_network_params['hidden_dim_2'],
                    'bn_eps': float(row['bn_eps']) if 'bn_eps' in row else default_network_params['bn_eps'],
                    'bn_momentum': float(row['bn_momentum']) if 'bn_momentum' in row else default_network_params['bn_momentum']
                }
                
                # Extract training parameters
                training_params = {
                    'learning_rate': float(row['learning_rate']) if 'learning_rate' in row else default_training_params['learning_rate'],
                    'batch_size': int(row['batch_size']) if 'batch_size' in row else default_training_params['batch_size'],
                    'batch_num': default_training_params['batch_num'],  
                    'epoch_num': default_training_params['epoch_num']
                }
                print("Running on optimal parameters from cross-validation")
                return {'network_params': network_params, 'training_params': training_params}
        except Exception as e:
            print(f"Warning: Could not load Optuna results from {optuna_file}: {e}")
            print("Using default parameters instead.")
    else:
        print(f"No Optuna results found at {optuna_file}. Using default parameters.")
    
    # Return defaults if file doesn't exist or couldn't be loaded
    # Ensure all parameters have correct types
    network_params = {
        'hidden_dim_1': int(default_network_params['hidden_dim_1']),
        'hidden_dim_2': int(default_network_params['hidden_dim_2']),
        'bn_eps': float(default_network_params['bn_eps']),
        'bn_momentum': float(default_network_params['bn_momentum'])
    }
    
    training_params = {
        'learning_rate': float(default_training_params['learning_rate']),
        'batch_size': int(default_training_params['batch_size']),
        'batch_num': int(default_training_params['batch_num']),
        'epoch_num': int(default_training_params['epoch_num'])
    }
    
    return {'network_params': network_params, 'training_params': training_params}


class loss_exp_OCE(nn.Module):
    def __init__(self,
                 Strike_price,
                    T,
                 lamb,
                 idx_asset,
                 X_max=False
                 ):
        super(loss_exp_OCE, self).__init__()
        self.K = Strike_price
        self.T = T
        self.lamb = lamb
        self.X_max = X_max
        self.idx_asset = idx_asset
    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
    def exp_utility(self, x):
        num_samples = x.shape[0]
        loss = (torch.logsumexp(x, dim=0) - math.log(num_samples)) / self.lamb
        return loss
    def compute_PnL(self, holding, price):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1,self.idx_asset])
        X = torch.sum(PnL,axis=1)-C_T
        return X

    def forward(self,
                holding, 
                price,
                mode='test'
               ):
        X = self.compute_PnL(holding, price)
        if self.X_max:
            X = torch.max(X, -torch.ones_like(X) * 10)
        loss = self.exp_utility(X)
        return loss

