import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import math
import os
from MarketData.Market_data_generator import *
from MarketData import Market_data_util
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from plots.plot import plot_holdings_hedge,_export_holdings_to_csv

class DeltaHedge:
    """
    Implements a classical Black-Scholes-Merton delta hedging strategy that serves as a benchmark for Deep Hedging models
    Initialization : 
        - asset_hedged (str) : Ticker symbol of the underlying asset ('AAPL' for example)
        - stock_prices (pd.DataFrame) : Historical stock prices, previously downloaded
        - assets_to_hedge (str) : Ticker symbol of the asset used to hedge the option ('AAPL' for example, has to be the same asset as the underlying)

    Pipeline : 
        1. Data preparation : split historical stock price into a training and test set. The training set is used to estimate the Black-Scholes parameters
        which are mu (historical drift) and sigma (historical volatility). The test set is used to calculate the optimal hedging. 

        2. Hedging strategy : Find the optimal holdings in the underlying asset at each timestep. This depends on the spot price (the current stock price),
        the volatility (sigma), the time remaining before the option expires (time to maturity), the risk-free rate and the strike K.

        3. PnL calculation : Find the PnL for each path and aggregates it to output the overall average PnL. It serves as a baseline for the other models 
    """

    def __init__(self, asset_hedged, stock_prices, assets_to_hedge, name_model, is_plot, plot_path):
        self.asset_hedged = asset_hedged
        self.stock_prices = stock_prices
        self.assets_to_hedge = assets_to_hedge
        self.K = 100
        self.S0 = 100
        self.dt = 1/252
        self.sequence_length = 31
        self.T = self.dt*self.sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name_model = name_model
        self.is_plot = is_plot
        self.plot_path = plot_path


        
    def data_delta_hedge(self):
        """
        Initializes and prepares the market data environment for the Delta hedging strategy. 
            - Selects the desired assets for hedging and splits the data into raw training and raw testing sets (fed to the data generator).
            - Creates the data generator object

        Returns : None
        """

        assets = ['AAPL','GOOGL','MSFT','AMZN','BRK-B']
        self.idx_asset_hedged = assets.index(self.asset_hedged)
        self.idx_assets_to_hedge = []
        for asset in self.assets_to_hedge:
            self.idx_assets_to_hedge.append(assets.index(asset))
        
        n_samples = int(0.7*self.stock_prices.shape[0])
        self.train_stock_prices = self.stock_prices.iloc[:n_samples,:]
        test_val_stock_prices = self.stock_prices.iloc[n_samples:,:]

        val_end = int(0.2*test_val_stock_prices.shape[0])
        self.val_stock_prices = test_val_stock_prices.iloc[:val_end,:]
        self.test_stock_prices = test_val_stock_prices.iloc[val_end:,:]

        self.train_stock_returns = np.log(self.train_stock_prices / self.train_stock_prices.shift(1)).dropna()
        
        train_size = 10000
        
        print(f"running on {self.device}")

        # create the data generator object
        self.data_generator = market_data(self.sequence_length, self.dt, self.train_stock_returns,self.val_stock_prices ,self.test_stock_prices, self.K, self.S0, self.idx_assets_to_hedge)
        self.number_assets = len(self.assets_to_hedge)


    def get_data_BS(self):
        """
        Creates the training and testing set by calling the function 'build_data()' from the previously created data generator object

        Returns : None
        """

        print("Generating data...")
        self.prices_train_data, self.test_data, _= self.data_generator.build_data()


    def black_scholes_delta(self, S, T, r, sigma):
        """
        Function responsible of calculating the Black-Scholes delta for a European Call option
        Calculate Black-Scholes delta for a European option, which represents the rate of change
        of the option price with respect to the underlying asset price 
        Args : 
            - S (float): Current stock price
            - T (int) : Time to maturity (in years)
            - r (float) : Risk-free rate
            - sigma (float): Volatility (daily)
        Returns : 
            - delta (float) : the delta of the option ; how much of the underlying to invest to mitigate 
            risks associated with fluctuations in the stock price 
        """
        if T <= 0:
            return 1.0 
        
        d1 = (np.log(S / self.K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1)
        return delta

    def hedge_all_paths(self, test_data, r, sigma_daily):
        """
        Apply delta hedging to all stock price paths.

        It iterates through every "simulated" (resampled) price paths which contains a 30 days stock path.
        It then iterates through the timesteps of each paths to construct the history of hedging positions. 

        In order to effectively compare with the other algorithms, this part must be done on test data.
        
        Args:
            test_data (np.array) : Array of shape (num_paths, num_steps, 1) containing the resamplec historical stock prices path
            r (int) : risk-free rate
            sigma_daily (int) : Daily volatility
        
        Returns:
            holdings (np.array): Array of shape (num_paths, num_steps) with delta hedge positions
        """
        
        num_paths, num_steps, _ = test_data.shape

        holdings = np.zeros((num_paths, num_steps-1))
        
        for path_idx in range(num_paths):
            stock_prices_path = test_data[path_idx, :, 0]  # Shape: (30,)
            
            for step in range(num_steps-1):
                S_current = stock_prices_path[step]
                # Remaining time to maturity decreases as days pass
                days_remaining = (num_steps) - step
                T = days_remaining / 252
                
                # Calculate delta for this step
                delta = self.black_scholes_delta(S_current, T, r, sigma_daily)
                holdings[path_idx, step] = delta
        
        return holdings





def run_delta_hedge(asset_hedged, stock_prices, assets_to_hedge,name_model,is_plot,plot_path):
    """
    Function responsible for the classical Black-Scholes delta-hedging. 
    Responsible of the complete pipeline for the classical BS hedging strategy. 

    Serves as benchmark for comparing against deep hedging strategies. 

    Args : 
        - asset_hedged (str) : ticker symbok of the underlying asset ('AAPL' for ex)
        - stock_prices (pd.DataFrame) : Historical stock prices dataset
        - name_model (str) : Model identifier for naming outputs ('delta_hedge' by construction)
        - is_plot (bool) : Whether to generate plots and export them
        - plot_path (str) : Path for saving the plots if is_plot == True
    """
    print(type(stock_prices))
    #Create and run delta hedge
    delta_hedge = DeltaHedge(asset_hedged, stock_prices, assets_to_hedge, name_model, is_plot, plot_path)
    delta_hedge.data_delta_hedge()
    delta_hedge.get_data_BS()

    # Calculate mu and sigma from log returns 
    log_returns = delta_hedge.train_stock_returns[asset_hedged].values
    mu_daily = log_returns.mean()  # Daily drift
    sigma_daily = log_returns.std()  # Daily volatility


    r = mu_daily  # Use daily drift as risk-free rate for consistency


    holdings = delta_hedge.hedge_all_paths(delta_hedge.test_data, r, sigma_daily)
    holdings = holdings[:, :, np.newaxis]
    
    loss_fn = Market_data_util.loss_exp_OCE(
                delta_hedge.K, 
                delta_hedge.T, 
                1.3, 
                delta_hedge.idx_assets_to_hedge,
                X_max=True, 
            ).to(delta_hedge.device)
    
    holdings = torch.from_numpy(holdings).float().to(delta_hedge.device)
    test_data_tensor = torch.from_numpy(delta_hedge.test_data).float().to(delta_hedge.device)
    PnL = loss_fn.compute_PnL(holdings,test_data_tensor)

    print(f'Delta Hedge PnL : {PnL.mean()}')
    if delta_hedge.is_plot : 
        plot_holdings_hedge(holdings, prices = delta_hedge.test_data, name = name_model, plot_path= delta_hedge.plot_path, idx_assets= delta_hedge.idx_assets_to_hedge)


