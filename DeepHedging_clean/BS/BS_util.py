import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from BS.BS_generator import *
import os
from Deephedging_JAX import DeepHedgingJAX
import math

def run_BS_deephedging(asset_hedged,stock_prices,assets_to_hedge,name_model,is_plot, plot_path):
    """
    Entry point for training deephedging models (either RNN or SigFormer) on Black-Scholes dynamics
    simulated data. Responsible of the whole pipeline, from data generation to training and testing

    Args : 
        - asset_hedged (str) : Underlying asset 
        - stock_prices (pd.DataFrame) : Historical price DataFrame for all assets
        - assets_to_hedge (list) : list of ticker symbols used as hedging instruments
        - name_model (str) : Model architecture (either 'SigFormer' or 'RNN_BN_simple')
        - is_plot (bool) : Boolean flag on Whether we want to plot some sampled hedging paths along with price paths
        - plot_path (str) : Path for saving plots/results
    """
    from Deephedging import DeepHedging

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assets = ['AAPL','GOOGL','MSFT','AMZN','BRK-B']
    idx_asset_hedged = assets.index(asset_hedged)
    idx_assets_to_hedge = []
    for asset in assets_to_hedge:
        idx_assets_to_hedge.append(assets.index(asset))
    train_stock_prices = stock_prices.iloc[:int(0.7*stock_prices.shape[0]),:]
    test_stock_prices =  stock_prices.iloc[int(0.7*stock_prices.shape[0]):,:]
    #train_stock_returns = np.log(train_stock_prices / train_stock_prices.shift(1)).dropna()
    # begin by initialising parameters for data generation
    sequence_length=30
    dt = 1/252
    K = 100
    S0 = 100 
    train_size = 1000

    # Get parameters from Optuna cross-validation if it has been run, otherwise use defaults
    params = get_parameters(asset_hedged, assets_to_hedge, model_type='BS')
    network_params = params['network_params']
    training_params = params['training_params']
    
    # define training parameters
    # tuple : (learning_rate, batch_size, batch_num, epoch_num)
    training_parameters = (
        training_params['learning_rate'],
        training_params['batch_size'],
        training_params['batch_num'],
        training_params['epoch_num']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    # create the data generator object
    current_dir = os.path.dirname(os.path.abspath(__file__))
    regime_model_dir = os.path.join(current_dir, "regime_models")
    os.makedirs(regime_model_dir, exist_ok=True)
    regime_model_path = os.path.join(regime_model_dir, f"gmm_{asset_hedged}.joblib")

    data_generator = Black_Scholes(sequence_length, dt, train_stock_prices, test_stock_prices, K,S0, train_size, idx_asset_hedged, idx_assets_to_hedge, regime_model_path=regime_model_path)
    # define the number of assets used for hedging
    number_assets = len(assets_to_hedge)
    # define the neural network architecture - BS uses 1 input (log price)
    network = RNN_BN_simple(
        number_assets, 
        data_generator.n_regimes, 
        sequence_length, 
        device,
        regime_embed_dim=network_params['regime_embed_dim'],
        hidden_dim_1=network_params['hidden_dim_1'], 
        hidden_dim_2=network_params['hidden_dim_2'],
        bn_eps=network_params['bn_eps'],
        bn_momentum=network_params['bn_momentum']
    ).to(device)

    # define the path where we will save the model 
    save_dir = os.path.join(current_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    
    if name_model == "RNN_BN_simple":
        name = "RNN_BN_simple"
        name = "RNN_BN_simple"
        for asset in assets_to_hedge:
            name +=  f"_{asset}"
        deephedging = DeepHedging(data_generator, number_assets, idx_asset_hedged, network, device, training_parameters, name, is_plot, plot_path, save_path=save_dir)
        deephedging.get_data_BS(idx_assets_to_hedge)
        deephedging.train_BS()
        deephedging.test()
    elif name_model == "SigFormer":
        # price returns + one-hot regimes
        in_dim = len(assets_to_hedge) + data_generator.n_regimes
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
            n_regimes=data_generator.n_regimes,
        )
        dh.get_data_BS(idx_assets_to_hedge)
        dh.train(epochs=parameters['epochs'], batch_size=parameters['batch_size'])
        dh.test()



def load_config():
    """
    Load configuration from config.yaml file.
    
    Returns:
        - dict: Configuration dictionary from yaml file
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


def get_parameters(asset_hedged, assets_to_hedge, model_type='BS'):
    """
    Load training and network parameters from Optuna cross-validation results if available,
    otherwise return default parameters from config.yaml.
    
    Args:
        - asset_hedged (str): Name of the asset being hedged (e.g., 'AAPL', 'GOOGL')
        - assets_to_hedge (list): List of assets used for hedging (e.g., ['AAPL', 'GOOGL'])
        - model_type (str): Type of model ('BS', 'Diffusion', etc.)
    
    Returns:
        - dict: Dictionary containing network_params and training_params
    
    Raises:
        - ValueError: If config.yaml is not found or doesn't contain required parameters
    """
    # Lazy import to avoid circular dependency
    from Cross_validation.optuna_hypparam_BS import get_BS_study_name
    
    # Load config.yaml
    config = load_config()
    
    # Check if config was loaded successfully
    if config is None:
        raise ValueError("Could not load config.yaml file. Please ensure it exists in the DeepHedging_clean directory.")
    
    # Check if BS_RNN_Default section exists
    if 'BS_RNN_Default' not in config:
        raise ValueError("config.yaml must contain 'BS_RNN_Default' section with default parameters.")
    
    # Extract default parameters from config.yaml
    bs_config = config['BS_RNN_Default']
    
    if 'default_network_params' not in bs_config:
        raise ValueError("config.yaml BS_RNN_Default section must contain 'default_network_params'.")
    
    if 'default_training_params' not in bs_config:
        raise ValueError("config.yaml BS_RNN_Default section must contain 'default_training_params'.")
    
    default_network_params = bs_config['default_network_params']
    default_training_params = bs_config['default_training_params']
    
    # Construct path to Optuna results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    optuna_dir = os.path.join(parent_dir, 'Cross_validation/Cross_validation_results')
    
    # Get study name using the cross-validation function
    study_name = get_BS_study_name(asset_hedged, assets_to_hedge)
    optuna_file = os.path.join(optuna_dir, f'{study_name}_best_params.csv')
    
    # Try to load Optuna results if the cross val code has been run before
    if os.path.exists(optuna_file):
        try:
            df = pd.read_csv(optuna_file)
            if len(df) > 0:
                row = df.iloc[0]
                
                # Extract network parameters
                network_params = {
                    'regime_embed_dim': int(row['regime_embed_dim']) if 'regime_embed_dim' in row else default_network_params['regime_embed_dim'],
                    'hidden_dim_1': int(row['hidden_dim_1']) if 'hidden_dim_1' in row else default_network_params['hidden_dim_1'],
                    'hidden_dim_2': int(row['hidden_dim_2']) if 'hidden_dim_2' in row else default_network_params['hidden_dim_2'],
                    'bn_eps': float(row['bn_eps']) if 'bn_eps' in row else default_network_params['bn_eps'],
                    'bn_momentum': float(row['bn_momentum']) if 'bn_momentum' in row else default_network_params['bn_momentum']
                }
                
                # Extract training parameters
                training_params = {
                    'learning_rate': float(row['learning_rate']) if 'learning_rate' in row else default_training_params['learning_rate'],
                    'batch_size': int(row['batch_size']) if 'batch_size' in row else default_training_params['batch_size'],
                    'batch_num': int(default_training_params['batch_num']),  
                    'epoch_num': int(default_training_params['epoch_num'])   
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
        'regime_embed_dim': int(default_network_params['regime_embed_dim']),
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


class RNN_BN_simple(nn.Module):
    """
    Class responsible of implementing the Recurrent Neural Network with batch normalization for 
    learning dynamic hedging strategies. 
    It accepts two inputs that serve as features : 
        - prices : shape = (batch_size, T, price_dim)
        - regimes : shape = (batch_size,T) --> integer regime labels OR shape == (batch_size, T,R) where the regime features are one-hot encoded
    
    The model uses regimes ONLY to choose its hedge.

    Despite its name 'simple', it is a time-distributed feedforward network, 
    where each timestep has its own network allowing the model to learn timestep-specific
    hedging behaviors. 
    """
    def __init__(self,
                 price_input_size,   # typically 1
                 n_regimes,          # number of discrete regimes
                 sequence_length,
                 device,
                 regime_embed_dim=4,  # embedding dimension
                 hidden_dim_1=20,     # first hidden layer size
                 hidden_dim_2=20,     # second hidden layer size
                 bn_eps=1e-3,         # BatchNorm epsilon
                 bn_momentum=0.3      # BatchNorm momentum
                 ):
        super().__init__()

        self.sequence_length = sequence_length
        self.device = device

        # ---- Regime encoder (embedding) ----
        self.embedding = nn.Embedding(n_regimes, regime_embed_dim)

        # Total input: price features + regime embedding
        self.input_size = price_input_size + regime_embed_dim

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


    def forward(self, prices, regimes):
        """
        Function defines how input data flows through the network to produce hedging decisions. 
        It combines price information along with market regime context to output hedge positions
        for each timestep. 
        Args : 
            - prices (torch.tensor):  (batch size, T, price_dim) input tensor 
            - regimes (int) : (batch size, T) integer labels 
        
        Returns : 
            - holdings (torch.tensor) : (batch size, T, price_dim) : Hedging positions for each path, timestep and asset
        """
        B, T, P = prices.shape
        #assert T == self.sequence_length

        # Regime embeddings
        regime_emb = self.embedding(regimes)  # (B, T, embed_dim)

        # Combine price + regime feature
        x = torch.cat([prices, regime_emb], dim=-1)  # (B, T, P+embed_dim)

        # Transpose to (T, B, F)
        x = x.transpose(0, 1)

        outputs = []

        for t in range(self.sequence_length):
            xt = x[t]                    # shape (B, F)
            out_t = self.rnn[t](xt)      # predicted hedge at time t
            outputs.append(out_t.unsqueeze(0))

        # Back to (B, T, price_dim)
        return torch.cat(outputs, dim=0).transpose(0, 1) 

class loss_exp_OCE(nn.Module):
    """
    Class responsible for the exponential certainty equivalent (OCE) loss function, which is the 
    objective function used to train deep hedging models. It measures the risk-adjusted performance 
    of a hedging strategy. 

    It accounts for one's risk preferences (through the lambda parameter) through an exponential utility function 
    """

    def __init__(self,
                 Strike_price,
                    sigma,
                    T,
                 lamb,
                 idx_asset,
                 X_max=False
                 ):
        super(loss_exp_OCE, self).__init__()
        self.K = Strike_price
        self.vol = sigma
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
        Applies the exponential utility function to measure risk-adjusted value : U(x) = 1/lambda * log(E[exp(-lambda  x)])
        Exponential utility form penalizes large losses more than it rewards large gains
        Mean across all paths gives the expectation 
        
        Args : 
            - x (torch.tensor) : shape == (30,1) PnL of the whole path (30 days) clipped by -10 to prevent the loss going to infty 
        
        Returns : 
            - loss (float) : exponential utility loss
        """

        num_samples = x.shape[0]
        loss = (torch.logsumexp(x, dim=0) - math.log(num_samples)) / self.lamb
        return loss

    def compute_PnL(self,holding,price):
        """
        Computes the overall PnL for each price path and averages it over all paths. The input 
        is the holdings (30 days vector) for each path and each asset (3rd dimension). It ouputs
        the average PnL per timestep across all paths. 

        The PnL is defined as the holdings (delta) from t-1 to t x changes in price from (t-1) to t summed
        over t's - terminal payoff (we pay C_T to the holder of the call option)

        Args : 
            - holding (torch.tensor) : shape == (batch_size, T, nb_assets used to hedge) 
            - price (torch.tensor) : shape is the same as holding
        
        Returns : 
            - X (torch.tensor) : shape == (batch_size, 1, nb_assets) : PnL for each path
        """

        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1,self.idx_asset])
        X = torch.sum(PnL,axis=1)-C_T
        return X
    
    def forward(self,holding, price):
        """
        This is the forward methd called during both training and testing to evaluate
        hedging performance. 
        Central function putting together the above functions to calculate all the
        metrics needed for the loss computations

        Args : 
            - holding (torch.tensor) : holdings tensor for the whole batch and all assets 
            - price (torch.tensor) : price tensor for the whole batch and all assets 
        Returns : 
            - loss (float) : batch's loss 

        """

        X = self.compute_PnL(holding,price)
        if self.X_max:
            X = torch.max(X, -torch.ones_like(X) * 10)
        loss = self.exp_utility(X)
        return loss
