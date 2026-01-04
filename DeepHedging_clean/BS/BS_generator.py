import torch
from BS.BS_util import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wrds
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import matplotlib.patches as mpatches
import os
    

class Black_Scholes: 
    """
    This class is the data generator for deep hedging models when the training data
    generating process follows the Black-Scholes dynamics with regime-switching.
    Regimes are detected from a US portfolio representing the overall US economy. 

    Instead of using the classical BS framework with constant parameters,
    this generator allows for : 
        1) Detecting market regimes using Gaussian Mixture Models (GMM)
        2) Estime regime-specific covariance matrices
        3) Simulate paths that switch between regimes
        4) Provide regime labels as additional features to the hedging network
    """

    def __init__(self, sequence_length, dt,train_stock_prices,test_stock_prices,K,S0,nmb_of_paths,idx_asset, idx_assets_to_hedge, regime_model_path=None):
        """
        Initializes the variables needed throughout the class
        """
        train_stock_returns = np.log(train_stock_prices / train_stock_prices.shift(1)).dropna()
        self.sequence_length = sequence_length
        self.dt = dt 
        self.mu = np.mean(train_stock_returns.values,axis=0)
        self.regime_model_path = regime_model_path
        self.cov_matrix = self.run_market_regime_clustering(train_stock_returns, regime_model_path=regime_model_path)
        self.sigma = [np.sqrt(cov[idx_asset,idx_asset]) for _,cov in self.cov_matrix.items()]
        self.T = self.sequence_length*self.dt
        self.K = K
        self.S0 = S0
        self.nmb_of_paths = nmb_of_paths
        self.df_stocks_test = test_stock_prices
        self.df_stocks_train = train_stock_prices
        self.idx_assets_to_hedge = idx_assets_to_hedge

    
    def build_data(self,idx_assets_to_hedge): 
        """
        Function responsible of generating data
        Args : 
            - idx_assets_to_hedge (list) : idx of the assets used to hedge the option
        
        Returns : 
            - prices (torch.tensor) : stock prices for all regimes and all paths and for the assets used to hedge (in 3rd dim)
            - regimes (torch.tensor): regime labels indicating which volatility regime each path belongs to 
        """
        self.price_train = self.generate(self.nmb_of_paths)
        prices,regimes = self.prepare_training_batch(self.price_train,device = 'cpu')
        self.price_train = prices[:,:,idx_assets_to_hedge]
        self.regimes = regimes
        return prices[:,:,idx_assets_to_hedge],regimes
    
    def run_market_regime_clustering(self,train_stock_returns,start = '2008-09-26', end= '2025-01-01' ,download = False ,path_mkt=None , path_stocks = None, model = True, plot = False, regime_model_path=None):
        """
        Function whose purpose is to detect distinct market regimes using GMM and estimate
        regime-specific covariance matrices for BS simulation. Methods which enables regime-switching
        dynamics in synthetic price generation
        Args : 
            - train_stock_returns (pd.DataFrame) : shape == (n_days,n_assets) log-returns of stocks during training period
            - start (str, optional) : Start date for market data in 'YYYY-MM-DD' format, default = '2008-09-26'
            - end (str, optional) : End date for market data in 'YYYY-MM-DD' format, default = '2025-01-01'
            - download (bool, optional) : If true, downloads market data from WRDS database (WRDS key needed)
            - path_mkt (str, optional)  : Path to value-weighted market returns CSV file. Default : 'Data/value_weighted_returns.csv' 
            - path_stocks (str, optional)  : Path to stock prices CSV
            - model (bool, optional) : default True
            - plot (bool,optional) : If true, generates scatter plot of market returns colored by regime
            - regime_model_path (str,optional) : Path save/load trained GMM model and scaler. If file exists, loads pre-trained model, If None or file doesnt exist, trains new model
        Returns : 
            - cov_matrix (dict) : dictionary mapping regime ID to covariance matrix for this specific regime (structure : (0 : sigma_0, 1 : sigma_1,..., k : sigma_k))
        """

        if path_mkt is None:
            path_mkt = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "value_weighted_returns.csv")
        if path_stocks is None:
            path_stocks = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "stocks_close_prices_2008_2021.csv")
        
        df_market = self.get_data_vol_clust(start, end,download,path_mkt)
        self.df_market_train = df_market.iloc[:int(train_stock_returns.shape[0])]
        self.df_market_test = df_market.iloc[int(train_stock_returns.shape[0]):]
        loaded = None
        if regime_model_path and os.path.exists(regime_model_path):
            try:
                loaded = joblib.load(regime_model_path)
                print(f"Loaded saved regime model from {regime_model_path}")
            except Exception as e:
                print(f"Warning: could not load regime model at {regime_model_path}: {e}")

        if loaded:
            scaler = loaded.get("scaler", None)
            model_obj = loaded.get("model", None)
            self.n_regimes = loaded.get("n_regimes", getattr(model_obj, "n_components", None))
            X, features, scaler = self.make_features_vol_clust(self.df_market_train,train_stock_returns, scaler=scaler)
            pred = pd.Series(model_obj.predict(X), index=features.index)
            model = model_obj
        else:
            X, features, scaler = self.make_features_vol_clust(self.df_market_train,train_stock_returns)
            pred,model = self.GMM(X,features,plot)

            # derive number of regimes from fitted model (best_estimator_) or from predictions
            self.n_regimes = getattr(model, "n_components", len(pd.Series(pred).unique()))

            if regime_model_path:
                to_save = {"model": model, "scaler": scaler, "n_regimes": self.n_regimes}
                os.makedirs(os.path.dirname(regime_model_path), exist_ok=True)
                joblib.dump(to_save, regime_model_path)
                print(f"Saved regime model to {regime_model_path}")

        cov_matrix = self.parameters_per_regime(pred,train_stock_returns)

        self.regime_model = model
        self.scaler = scaler
        self.features_train = features
        self.pred_train = pred
        return cov_matrix

    def get_data_vol_clust(self,start,end,download ,path_mkt):
        """
        Loads or download CRSP value-weighted market return data regime detection clustering
        This market data provides the signals for identifiying regimes (e.g calm vs crisis periods classified in multiple levels)

        Args : 
            - start (str) : start date for data fetching in 'YYYY-MM-DD' format
            - end (str) : end date for data fetching in 'YYYY-MM-DD' format
            - download (bool) : If True : Downloads fresh data from WRDS database, if False : Read from local CSV file located in path_mkt
            - path_mkt (str) : File path to CSV file containing market returns
        Returns : 
            - df_market (pd.DataFrame) : Market returns dataframe with the Date as Index 

        """
        if download:
            db = wrds.Connection()
            query = f"""
                SELECT date, vwretd
                FROM crsp.dsi
                WHERE date BETWEEN '{start}' AND '{end}'
                ORDER BY date;
            """
            df_market = db.raw_sql(query)
            df_market['date'] = pd.to_datetime(df_market['date'])
            df_market = df_market.set_index('date')
            df_market.to_csv(path_mkt, index=True)

        df_market = pd.read_csv(path_mkt)
        df_market = df_market.set_index('date')
        return df_market

    def make_features_vol_clust(self,df_market,ret_stocks,scaler=None):
        """
        Engineers four key features from market and stock returns data from regime detection clustering. 
        Features capture different dimensions of market behavior : direction, volatility, dispersion and 
        correlation structure. 
            - Feature 1 : Daily market returns data 
            - Feature 2 : 30 days rolling window volatility
            - Feature 3 : Cross-sectional volatility : Measures dispersion of returns across stocks on each day
            - Feature 4  : Average 60 days rolling window correlation 

        Args : 
            - df_market (pd.DataFrame) : Market-level returns data
            - ret_stocks (pd.DataFrame) : Individual stock log-returns
            - scaler (StandardScaler, optional) : Pre-fitted sklearn StandardScaler
        Returns : tuple(X,featurs,scaler)
            - X (np.ndarray) : standardized feature matrix ready for GMM clustering
            - features (pd.DataFrame) : original (unstandardized) features with DateTimeIndex
            - scaler (StandardScaler) : Fitted StandardScaler object
        """

        features = pd.DataFrame()
        features['market_ret'] = df_market['vwretd']
        ## 30 days rolling window vol
        features['vol_30d'] = df_market['vwretd'].rolling(30).std()
        ## Cross sectionnal volatily
        features['XS_vol'] = ret_stocks.std(axis=1)
        ## mean corr over last 60 days
        features['avg_corr_60d'] = ret_stocks.rolling(60).corr().groupby(level=0).apply(lambda x: x.values[np.triu_indices_from(x, k=1)].mean())
        features.dropna(inplace=True)
        
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(features)
        else:
            X = scaler.transform(features)
        return X,features,scaler


    def build_colormap(self,n_states, cmap_name="tab10"):
        """
        Automatically build a colormap for n_states using a matplotlib colormap.
        Returns a dict: {0: color0, 1: color1, ...}
        """
        cmap = plt.get_cmap(cmap_name)
        colors = {state: cmap(state % cmap.N) for state in range(n_states)}
        return colors


    def GMM(self,X,features, plot = True):
        """
        Fits a Gaussian Mixture Model to detect optimal number market regimes using 
        GridSearch over model configurations (n_components from 1 to 6 and covariances type).
        Uses Bayesian Information Criterion to balance model fit quality with complexity, preventing overfitting

        Args : 
            - X (np.ndarray) : shape == (n_days, 4) standardized feature matrix containing [market_ret, vol_30d, XS_vol, avg_corr_60d]
            - features (pd.DataFrame) : Original unstandardized features
            - plot (bool,optional) : Whether to generate visualization of detected regimes 
        
        Returns : tuple : (predictions, best_model) : 
            - predictions (pd.Series) : Regime assignments for each day
            - best_model (GaussianMixture) : Fitted sklearn GaussianMixture with optimal parameters
        """
        n_components = np.arange(1,7)
        param_grid = { 
                    'n_components' : n_components,
                    'covariance_type' :['spherical','tied','diag','full']}
        
        def bic_scorer(fitted_gmm, X):
            return -fitted_gmm.bic(X)  # return -bic as higher is better for GridSearchCV

        model_GMM  = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=bic_scorer)
        model_GMM.fit(X)
        best_model = model_GMM.best_estimator_
        print('number of states =', model_GMM.best_params_['n_components'])
        predictions = best_model.predict(X)
        if plot :
            color_map = self.build_colormap(model_GMM.best_params_['n_components'])
            features.index = pd.to_datetime(features.index)
            features['state_GMM'] = predictions
            colors = features['state_GMM'].map(color_map)
            plt.figure(figsize=(12,6))

            plt.scatter(
                features.index,
                features['market_ret'],
                c=colors,
                s=10
            )
            legend_handles = [
                mpatches.Patch(color=color, label=f"State {int(state)}")
                for state, color in color_map.items()
            ]

            plt.legend(handles=legend_handles, title="GMM States")

            plt.title("Returns by GMM State")
            plt.xlabel("Time")
            plt.ylabel("Returns")
            plt.grid(True)
            
            plt.show()
        return pd.Series(predictions,index=features.index),best_model

    def parameters_per_regime(self,predictions, market_data):
        """
        Function that computes regime-specific covariance matrices by grouping stock returns according
        to their assigned market regime. These covariance matrices capture the disctinct 
        correlation and volatility structures characteristic of each market state. 
        Args : 
            - predictions : Regime assignments for each day 
            - market_data (pd.DataFrame) : Stock log returns used to compute covariances
        Returns : 
            - cov_matrix_regime (dict) : Dictionary mapping regime ID to covariance matrix
        """

        market_data['regimes'] = predictions
        market_data.dropna(inplace=True)
        cov_matrix_regime = {}
        for k in np.unique(predictions):
            cov_matrix_regime[k]  = market_data.iloc[:,:-1][market_data['regimes'] == k].cov().to_numpy()
        return cov_matrix_regime

    def generate(self,M):
        sigma_chol = {}
        N = self.cov_matrix[0].shape[0]
        T = self.sequence_length +1 # time window plus un pour le payoff
        samples = {}
        regimes = np.arange(len(self.cov_matrix))
        dt = 1/252
        for i in range(len(self.cov_matrix)):
            sigma_chol[i] = np.linalg.cholesky(self.cov_matrix[i])
            S = np.zeros((T, N, M))
            S[0] = 100*np.ones((N,M))
            for t in range(1,T):
                z = np.random.standard_normal((N, M))
                S[t] = S[t-1] * np.exp((self.mu.reshape(-1,1)-1/2 *np.diag(self.cov_matrix[i]).reshape(-1,1))*dt + sigma_chol[i] @ z) 
            samples[regimes[i]] = S
        return samples
    
    def prepare_training_batch(self,samples, device="cpu"):
        """
        Transforms regime-separated synthetic price paths into Pytorch tensors suitable for NN training. 
        Combines paths from all regimes into a single unified training dataset with corresponding regime labels. 

        Args : 
            - samples (dict) : Dictionary of price paths keyed by regime ID : {regime_id : array(T,N,M)}
            - device (str, optional) : PyTorch device for tensor placement

        Returns:
            - prices (torch.Tensor)  : shape == (batch_size, T, N) : batch price paths tensor
            - regimes (torch.Tensor) : shape == (batch_size, T) : regime labels for each path and timestep
        """
        
        price_list = []
        regime_list = []

        for regime, S in samples.items():
            T, N, M = S.shape

            # Convert to (M, T, N)
            S_paths = np.transpose(S, (2, 0, 1))   # (M paths, T timesteps, N assets)

            # Regime label repeated for every path and timestep
            r = np.full((M, T), regime)

            price_list.append(S_paths)
            regime_list.append(r)

        # Stack all regimes
        prices = np.concatenate(price_list, axis=0)     # (batch, T, N)
        regimes = np.concatenate(regime_list, axis=0)   # (batch, T)

        
        prices  = torch.tensor(prices, dtype=torch.float32).to(device)
        regimes = torch.tensor(regimes, dtype=torch.long).to(device)
        return prices, regimes

    def make_test_data(self, idx_assets_to_hedge, device="cpu"):
        """
        Build a test dataset from REAL historical market data.
        It takes the last 80% of the real stock prices and computes log returns. 
        Then perform regime detection using the trained GMM model and scaler from training. Compute
        the same 4 features and predicts which market regime each test period belongs to
        Creates overlapping sequences of lenght T from the historcla data, each sequence is normalized
        such that it starts at S0
        Args : 
            - df_market_test (pd.DataFrame): DataFrame with 'vwretd' 
            - df_stocks_test (pd.DataFrame) : DataFrame with close prices for the same stocks as training
            - idx_asset (list): which column of df_stocks_test we hedge on
        Returns:
            - prices_test (torch.tensor)  : shape == (batch, T, N)  : contains normalized price paths for the selecte assets
            - regimes_test (torch.tensor) :  shape == (batch, T) : contains regime labels for each timestep of each path
        """
        self.df_stocks_test = self.df_stocks_test.iloc[int(0.2*self.df_stocks_test.shape[0]):, :]
        ret_stocks_test = np.log(self.df_stocks_test / self.df_stocks_test.shift(1)).dropna()

        
        X_test, features_test, _ = self.make_features_vol_clust(
            self.df_market_test, ret_stocks_test, scaler=self.scaler
        )

        
        regimes_series = pd.Series(
            self.regime_model.predict(X_test),
            index=features_test.index
        )
       
        asset_prices = self.df_stocks_test

        asset_prices.index = pd.to_datetime(asset_prices.index)
        regimes_series.index = pd.to_datetime(regimes_series.index)
        features_test.index = pd.to_datetime(features_test.index)

        
        common_index = asset_prices.index.intersection(regimes_series.index)
        asset_prices = asset_prices.loc[common_index]
        regimes_series = regimes_series.loc[common_index]

        T = self.sequence_length + 1
        price_paths = []
        regime_paths = []

        for start in range(0, len(asset_prices) - T + 1):
            end = start + T

            # price path
            S_window = asset_prices.iloc[start:end].values  
            # normalize path to start at 1 (matches BS generator)
            S_norm = S_window / S_window[0] * self.S0
            price_paths.append(S_norm)  #

            # regime path
            r_window = regimes_series.iloc[start:end].values  
            regime_paths.append(r_window)

        prices_test = torch.tensor(
            np.stack(price_paths, axis=0), dtype=torch.float32
        ).to(device)   # (batch, T, N)

        regimes_test = torch.tensor(
            np.stack(regime_paths, axis=0), dtype=torch.long
        ).to(device)   # (batch, T)
        return prices_test[:,:,idx_assets_to_hedge], regimes_test

    def make_val_data(self, idx_assets_to_hedge, device="cpu"):
        """
        Identic function to make_test_data but performed on the validation set. 
        """
        val_prices = self.df_stocks_test.iloc[:int(0.2*self.df_stocks_test.shape[0]), :]
        
        ret_stocks_val = np.log(val_prices / val_prices.shift(1)).dropna()

        
        X_test, features_test, _ = self.make_features_vol_clust(
            self.df_market_test, ret_stocks_val, scaler=self.scaler
        )

        
        regimes_series = pd.Series(
            self.regime_model.predict(X_test),
            index=features_test.index
        )
        
        asset_prices = self.df_stocks_test

        asset_prices.index = pd.to_datetime(asset_prices.index)
        regimes_series.index = pd.to_datetime(regimes_series.index)
        features_test.index = pd.to_datetime(features_test.index)

        
        common_index = asset_prices.index.intersection(regimes_series.index)
        asset_prices = asset_prices.loc[common_index]
        regimes_series = regimes_series.loc[common_index]

        T = self.sequence_length + 1
        price_paths = []
        regime_paths = []

        for start in range(0, len(asset_prices) - T + 1):
            end = start + T

            # price path
            S_window = asset_prices.iloc[start:end].values  
            # normalize path to start at 1 (matches BS generator)
            S_norm = S_window / S_window[0] * self.S0
            price_paths.append(S_norm)  #

            # regime path
            r_window = regimes_series.iloc[start:end].values  
            regime_paths.append(r_window)

        prices_val = torch.tensor(
            np.stack(price_paths, axis=0), dtype=torch.float32
        ).to(device)   # (batch, T, N)

        regimes_val = torch.tensor(
            np.stack(regime_paths, axis=0), dtype=torch.long
        ).to(device)   # (batch, T)

        return prices_val[:,:,idx_assets_to_hedge], regimes_val

    def plot_data(self,step):
        """
        Function to visualize a subset of generated stock price paths to inspect the quality and characteristics of the 
        simulated data. 
        Args : 
            - step (int) : sampling interval --> plots every step-th path to help visualization
        
        Returns : 
            - None
        """
        plt.figure(figsize=(12,6))
        for i in range(0,self.paths_np.shape[0],step) : 
            plt.plot(self.paths_np[i], alpha = 0.6) 
        plt.title(f'Sampled stock price paths (every {step}th)')
        plt.xlabel('Timestep')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
    
    def save_data(self):
        """
        Saves the train,test and validation set for later use without regenerating
        """

        torch.save(self.price_train, f'{self.path}/BS_train.pt') 
        torch.save(self.price_test, f'{self.path}/BS_test.pt')
        torch.save(self.price_val, f'{self.path}/BS_val.pt')
