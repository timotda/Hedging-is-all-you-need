import torch
import torch.nn as nn
import numpy as np
import yaml
from Diffusion.Diffusion_generator import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from Deephedging_JAX import DeepHedgingJAX
import math

def load_config():
    """
    Load configuration settings from config.yaml file.
    Returns:
        - dict: Configuration dictionary from yaml file
        - None : If the config file cannot be uploaded along with a Warning Message
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


def get_parameters(asset_hedged, assets_to_hedge, model_type='Diffusion'):
    """
    Load training and network parameters from Optuna cross-validation results if available,
    otherwise return default parameters from config.yaml.
    
    Args:
        asset_hedged (str): Name of the asset being hedged (e.g., 'AAPL', 'GOOGL')
        assets_to_hedge (list): List of assets used for hedging
        model_type (str): Diffusion by construction
    
    Returns:
        dict: Dictionary containing network_params and training_params
    
    Raises:
        ValueError: If config.yaml is not found or doesn't contain required parameters
    """
    # Load config.yaml
    config = load_config()
    
    # Check if config was loaded successfully
    if config is None:
        raise ValueError("Could not load config.yaml file. Please ensure it exists in the DeepHedging_clean directory.")
    
    # Check if Diffusion_RNN_Default section exists
    if 'Diffusion_RNN_Default' not in config:
        raise ValueError("config.yaml must contain 'Diffusion_RNN_Default' section with default parameters.")
    
    # Extract default parameters from config.yaml
    diffusion_config = config['Diffusion_RNN_Default']
    
    if 'default_network_params' not in diffusion_config:
        raise ValueError("config.yaml Diffusion_RNN_Default section must contain 'default_network_params'.")
    
    if 'default_training_params' not in diffusion_config:
        raise ValueError("config.yaml Diffusion_RNN_Default section must contain 'default_training_params'.")
    
    default_network_params = diffusion_config['default_network_params']
    default_training_params = diffusion_config['default_training_params']
    
    # Construct path to Optuna results using the same logic as in main.py
    from Cross_validation.optuna_hypparam_diffusion import get_diffusion_study_name
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    optuna_dir = os.path.join(parent_dir, 'Cross_validation/Cross_validation_results')
    
    # Get study name using the same function as main.py
    study_name = get_diffusion_study_name(asset_hedged, assets_to_hedge)
    optuna_file = os.path.join(optuna_dir, f'{study_name}_best_params.csv')
    
    # Try to load Optuna results
    if os.path.exists(optuna_file):
        try:
            df = pd.read_csv(optuna_file)
            if len(df) > 0:
                print(f"Loading optimized parameters from {optuna_file}")
                row = df.iloc[0]
                
                # Extract network parameters (no regime_embed_dim for Diffusion)
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
                    'batch_num': int(default_training_params['batch_num'])   ,  
                    'epoch_num': int(default_training_params['epoch_num'])   
                }
                
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

def run_diffusion_deephedging(asset_hedged,assets_to_hedge,name_model,is_plot, plot_path):
    """
    Run deep hedging on diffusion-simulated paths.
    
    Args:
        asset_hedged: Name of the asset to hedge (e.g., 'AAPL')
        simulated_paths: Pre-simulated paths (num_paths, T, num_assets) - 3D array
        assets_to_hedge: List of asset names to use for hedging
    """
    from Deephedging import DeepHedging


    # Define available assets matching the order in simulated_paths
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'BRK-B']
    
    # Get indices for the asset to hedge and assets used for hedging
    idx_asset_hedged = assets.index(asset_hedged)
    idx_assets_to_hedge = [assets.index(asset) for asset in assets_to_hedge]
    
    # Initialize parameters for data generation
    sequence_length = 30
    dt = 1/365
    K = 100
    S0 = 100 

    # Get parameters from Optuna cross-validation if available, otherwise use defaults
    params = get_parameters(asset_hedged, assets_to_hedge, model_type='Diffusion')
    network_params = params['network_params']
    training_params = params['training_params']

    # Define training parameters
    # (learning_rate, batch_size, batch_num, epoch_num)
    training_parameters = (
        training_params['learning_rate'],
        training_params['batch_size'],
        training_params['batch_num'],
        training_params['epoch_num']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    # Create the data generator object with simulated paths
    data_generator = diffusion_data(
        sequence_length, 
        dt,  
        K, 
        S0, 
        idx_assets_to_hedge
    )
    
    # Define the number of assets used for hedging
    number_assets = len(assets_to_hedge)
    
    # Define the neural network architecture with optimized parameters
    network = RNN_BN_simple(
        number_assets, 
        sequence_length, 
        device,
        hidden_dim_1=network_params['hidden_dim_1'],
        hidden_dim_2=network_params['hidden_dim_2'],
        bn_eps=network_params['bn_eps'],
        bn_momentum=network_params['bn_momentum']
    ).to(device)
    

    # Define the path where we will save the model 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    if name_model == 'RNN_BN_simple':
        name = "RNN_BN_simple"
        name = "RNN_BN_simple"
        for asset in assets_to_hedge:
            name +=  f"_{asset}"
        deephedging = DeepHedging(
            data_generator, 
            number_assets,
            idx_asset_hedged, 
            network, 
            device, 
            training_parameters,
            name, 
            is_plot, 
            plot_path,
            save_path=save_dir
        )
        deephedging.get_data_Diffusion()
        deephedging.train_Diffusion()
        deephedging.test()
    elif name_model == 'SigFormer':
        in_dim = len(assets_to_hedge)
        out_dim = len(assets_to_hedge)

        config = load_config()
        parameters = config['SigFormer']

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
        dh.get_data_Diffusion()
        dh.train(epochs=parameters['epochs'], batch_size=parameters['batch_size'])
        dh.test()


def merge_diffusion_data(tickers):
    """
    this function merges together the prices (pre)generated from the diffusion given a list of tickers into 
    a signle multi-dimensional array suitable for training multi-asset hedging models
    Args : 
        - tickers (list) : stock ticker list 
    Returns : 
        - simulated_paths (np.array) : shape == (num_paths, T, num_assets) : contains synchronized price return paths for all assets in tickers
    """
    assets_data = {}
    try:
        for ticker in tickers:
            path = "Data/" + str(ticker) + "_returns.npy"
            print(path)
            assets_data[ticker] = np.load(path)
    except:
        print("\n WARNING: You have not pre-generated the diffusion data for the given tickers, refer to the Read-Me to generate the data, then try again. \n Or verify ticker names.")


    arrays = [assets_data[ticker] for ticker in tickers]

    # Stack into shape (num_paths, T, num_assets)
    simulated_paths = np.stack(arrays, axis=-1)

    if simulated_paths.ndim == 4:
        simulated_paths = simulated_paths.squeeze(axis=2)

    return simulated_paths

class RNN_BN_simple(nn.Module):
    """
    RNN with BatchNorm that accepts price inputs:
    - prices: (batch, T, price_dim)

    The model processes price paths to determine optimal hedging strategy.
    """
    def __init__(self,
                 price_input_size,   # number of assets
                 sequence_length,
                 device,
                 hidden_dim_1=20,    # first hidden layer size
                 hidden_dim_2=20,    # second hidden layer size
                 bn_eps=1e-3,        # BatchNorm epsilon
                 bn_momentum=0.3     # BatchNorm momentum
                 ):
        super().__init__()

        self.sequence_length = sequence_length
        self.device = device
        self.input_size = price_input_size

        # ---- RNN with BN layers ----
        self.rnn = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(self.input_size),
                nn.Linear(self.input_size, hidden_dim_1),
                nn.BatchNorm1d(hidden_dim_1),
                nn.ReLU(),
                nn.Linear(hidden_dim_1, hidden_dim_2),
                nn.BatchNorm1d(hidden_dim_2),
                nn.ReLU(),
                nn.Linear(hidden_dim_2, price_input_size)     # output hedge for each asset
            )
            for _ in range(sequence_length)
        ])

        # BatchNorm stability tuning
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eps = bn_eps
                module.momentum = bn_momentum

    def forward(self, prices):
        """
        Function defines how input data flows through the network to produce hedging decisions. 
        Args : 
            - prices (torch.tensor):  (batch size, T, price_dim) input tensor 
        Returns : 
            - holdings (torch.tensor) : (batch size, T, price_dim) : Hedging positions for each path, timestep and asset
        """
        B, T, P = prices.shape
        x = prices
        # Transpose to (T, B, P)
        x = x.transpose(0, 1)

        outputs = []

        for t in range(T):
            xt = x[t]                    # shape (B, P)
            out_t = self.rnn[t](xt)      # predicted hedge at time t
            outputs.append(out_t.unsqueeze(0))

        # Back to (B, T, price_dim)
        return torch.cat(outputs, dim=0).transpose(0, 1)


class loss_exp_OCE(nn.Module):
    """
    Exponential utility loss function for deep hedging.
    """
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
        """
        Calculates the terminal payoff of the option at time T = 30 (timesteps 0,...,29)
        C_T = payoff at expiration = max(S_T - K,0)

        Args : 
            - final price (torch.tensor) : shape == (nb of paths, 1,1) final price tensor for one asset
        
        Returns : 
            - terminal payoff (torch.tensor) : shape == (nb of paths, 1,1) terminal payoff tensor for one asset
        """
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
    def exp_utility(self, x):
        """
        Computes the exponential utility-based risk measure for holdings outcomes
        Args : 
            - x (torch.tensor) : shape == (num_samples,) represents net profit/loss values across different scenarios
        
        Returns : 
            - loss (float) : value representing the expected loss 
        """

        num_samples = x.shape[0]
        loss = (torch.logsumexp(x, dim=0) - math.log(num_samples)) / self.lamb
        return loss

    def forward(self,holding,price):
        """
        Computes the delta, the change in price of the asset, between each timestep, 
        Uses this delta to find PnL and then fin the terminal payoff of the option using the last
        price of the sequence (at maturity). Then computes the PnL and feeds it to the loss function
        Args : 
            - holding (torch.Tensor) : shape == (batch_size,T, nb_assets) : hedge positions (nb units of each asset held at each timestep)
            - price (torch.Tensor) : shape == (batch_size, T, nb_assets) : asset price paths
        Returns : 
            - loss (float) : Average batch's loss
        """

        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1, self.idx_asset])
        X = torch.sum(PnL, axis=1) - C_T
        
        if self.X_max:
            X = torch.max(X, -torch.ones_like(X) * 10)

        loss = self.exp_utility(X)
        return loss
