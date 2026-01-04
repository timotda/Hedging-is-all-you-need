from BS.BS_generator import *

import torch
import torch.nn as nn
import argparse
import time
import numpy as np
from BS import BS_util
from MarketData import Market_data_util
from plots.plot import _export_holdings_to_csv, plot_train_vs_val_loss


def lag_returns(returns):
        """
        Function that shifts returns by one timestep to create lagged returns. 
        It is used in order to prevent look ahead bias when training. 
        Indeed, when the network takes decisions, it should not be able
        to see next period returns to make any decision. 

        Args : 
            - returns (torch.tensor) : returns for all paths
        
        Returns : 
            - torch.tensor : returns shifted by for all paths. The first line has been replaced by 0's
        """
        # returns: (B, T-1, N) where returns[:,t]=log(S_{t+1}/S_t)
        zero = torch.zeros_like(returns[:, :1, :])
        return torch.cat([zero, returns[:, :-1, :]], dim=1)  # (B, T-1, N)


class DeepHedging:
    """
    Object Deephedging is responsible of the whole pipeline for the Neural Network. It wraps
    together all the required functions and generators for generating data, training the network
    and testing. 
    """

    def __init__(self, data_generator, number_assets,idx_asset, network, device, training_parameters, name, is_plot,plot_path, save_path=None):

        """
        Args : 
            - data_generator (Object) : object of how to create paths (BS, MarketData, Diffusion)
            - number_assets (int) : the number of assets we are going to hedgethe option with
            - idx_asset (int) : the index of the asset the option is based on
            - network (Object) : The network responsible for training and testing 
            - device (torch.device) : The device where the Pytorch operations are executed 
            - training_parameters (tuple) : a tuple containing (learning_rate, batch_size, batch_num, epoch_num)
            - name (str) : the name of the network used (RNN_BN_simple)
            - is_plot (Bool) : Bool to express whether we want to plot the hedging holdings
            - plot_path (str): location of the path where we want to save the plots 
        """
        self.data_generator = data_generator
        self.number_assets = number_assets
        self.network = network
        self.save_path = save_path
        self.device = device
        self.learning_rate, self.batch_size, self.batch_num, self.epoch_num = training_parameters
        self.name = name
        self.idx_asset = idx_asset
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.test_returns = None
        self.is_plot = is_plot
        self.plot_path = plot_path
        
    
    def get_data(self):
        """
        Builds train/test/val datasets using the data generator's build_data() function. 
        """
        print("Build data...")
        self.train_data, self.test_data, self.val_data = self.data_generator.build_data()
        print("Data built.")

    def get_data_BS(self,idx_assets_to_hedge):
        print("Build data...")

        self.prices_train_data,self.regimes_train_data = self.data_generator.build_data(idx_assets_to_hedge)
        self.prices_val_data, self.regimes_val_data = self.data_generator.make_val_data(idx_assets_to_hedge)
        test_prices, test_regimes = self.data_generator.make_test_data(idx_assets_to_hedge)
        
        # Create validation data: take first 20% of testing data
        """
        val_split = int(0.2 * test_prices.shape[0])
        self.prices_val_data = test_prices[:val_split]
        self.regimes_val_data = test_regimes[:val_split]
        self.test_data = (test_prices[val_split:], test_regimes[val_split:])
        """
       
        self.test_data = (test_prices, test_regimes)
        
        print("Data built.")

    def get_data_Diffusion(self):
        """
        Builds train/test/val datasets specifically for the Diffusion data generating process
        """
        print("Build data...")
        self.train_data = self.data_generator.build_data()
        test_data_full, test_returns_full = self.data_generator.make_test_data()
        
        # Create validation data: take first 20% of test data
        val_split = int(0.2 * test_data_full.shape[0])
        self.val_data = test_data_full[:val_split]
        self.val_returns = test_returns_full[:val_split]
        self.test_data = test_data_full[val_split:]
        self.test_returns = test_returns_full[val_split:]
        
        print("Data built.")

    
    def train_BS(self):
        """
        Trains the network on Black-Scholes data with regime switching. 
        Utilizes exponential utility loss, Adam optimizer, validates each epoch 
        and plots training vs val curves if specified 
        """

        # these parameters should be given by the user !
        T = self.data_generator.dt * self.data_generator.sequence_length
        K = self.data_generator.K
        S0 = self.data_generator.S0
        sigma = self.data_generator.sigma
        


        price_train = self.prices_train_data
        regime_train = self.regimes_train_data
        price_val = self.prices_val_data
        regime_val = self.regimes_val_data
        N = price_train.shape[0]
        
        print("Starting training...")
        print(f"Training samples: {price_train.shape[0]}, Validation samples: {price_val.shape[0]}")
        
        # Prepare validation data loader
        val_dataset = torch.utils.data.TensorDataset(price_val, regime_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # validation results and training results are kept in this array for plotting
        train_val_results = []
        
        # Calculate number of partitions based on available training data
        num_partitions = min(int(1e5/N), len(price_train) // N)
        for part in range(0, num_partitions):
            index1 = part*N
            index2 = (part+1)*N
            # Prepare partitioned training data and loader
            price_batch  = price_train[index1:index2]
            regime_batch = regime_train[index1:index2]
            train_data = torch.utils.data.TensorDataset(price_batch, regime_batch)
            print(f"BS partition {part}/{num_partitions}, samples: {len(price_batch)}")
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

            # Initialize network and loss function
            network = self.network
            # Use BS exponential utility loss
            loss_fn = BS_util.loss_exp_OCE(K, sigma, T,1.3,self.idx_asset,X_max=True).to(self.device)

            # Setup Trainable parameters and optimizer

    
            opt = torch.optim.Adam([
                {'params': network.parameters()},  # Model parameters
            ], lr=self.learning_rate)

            best_loss = float('inf')
            # Training loop
            for i in range(self.epoch_num):
                time1 = time.time()
                network.train()
                train_result = self.epoch_loader_BS(train_loader, network, loss_fn, opt)
                
                # Validation evaluation
                network.eval()
                with torch.no_grad():
                    val_result = self.epoch_loader_BS(val_loader, network, loss_fn, opt=None)
                
                train_val_results.append((train_result, val_result))
                time2 = time.time()
                print(f"epoch {i}, train loss: {train_result:.6f}, val loss: {val_result:.6f}, time: {time2-time1:.2f}s")
 

        # Save trained model
        network.to('cpu')
        network.device = 'cpu'
        print("Training completed.")
        if self.save_path:
            print("Saving model...")
            torch.save(network.state_dict(), f"{self.save_path}/{self.name}.pt")
        if self.is_plot:
            plot_train_vs_val_loss(train_val_results)

    def train_MarketData(self):
        """
        Trains the network on real market data.
        Utilizes the exponential utility loss (with lambda = risk aversion = 1.3), Adam optimizer, a learning rate scheduler function of the epoch, 
        validates each epoch, saves model if specified and plots training vs val loss if specified.
        """
        # these parameters should be given by the user !
        T = self.data_generator.dt * self.data_generator.sequence_length
        K = self.data_generator.K
        S0 = self.data_generator.S0
        
    
        # Define the name for saving the model
        name = self.name

        price_train = self.train_data
        price_val = self.val_data
        N = price_train.shape[0]

        print("Starting training...")
        print(f"Training samples: {price_train.shape[0]}, Validation samples: {price_val.shape[0]}")
        
        # Prepare validation data loader
        tensor_val_data = torch.tensor(price_val, dtype=torch.float32)
        val_dataset = torch.utils.data.TensorDataset(tensor_val_data)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # validation results and training results are kept in this array for plotting
        train_val_results = []
        # train_data is now a numpy array (num_windows, T, num_assets) from rolling windows
        # Calculate number of partitions based on available training data
        num_partitions = 0 # min(int(1e5/N), len(price_train) // N)
        for part in range(0, num_partitions+1):
            index1 = part*N
            index2 = (part+1)*N
            # Prepare partitioned training data and loader
            eps  = 1e-6
            price_batch  =price_train[index1:index2] # np.log((price_train[index1:index2][:, 1:, :] + eps) / (price_train[index1:index2][:, :-1, :] + eps))
            price_batch_returns = np.log((price_train[index1:index2][:, 1:, :] + eps) / (price_train[index1:index2][:, :-1, :] + eps))
            # Convert numpy array to tensor
            tensor_data = torch.tensor(price_batch, dtype=torch.float32)
            train_data = torch.utils.data.TensorDataset(tensor_data)
            print(f"Market data partition {part}/{num_partitions}, samples: {len(price_batch)}")
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

            # Initialize network and loss function
            network = self.network
            loss_fn = Market_data_util.loss_exp_OCE(K, T, 1.3, self.idx_asset,X_max=True).to(self.device)

    
            opt = torch.optim.Adam([
                {'params': network.parameters()},  # Model parameters 
            ], lr=self.learning_rate)
            # Learning rate scheduler
            LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.alpha_learning_rate)

            best_loss = float('inf')
            # Training loop
            for i in range(self.epoch_num):
                time1 = time.time()
                network.train()
                train_result = self.epoch_loader_MarketData(train_loader, network, loss_fn, opt)
                
                # Validation evaluation
                network.eval()
                with torch.no_grad():
                    val_result = self.epoch_loader_MarketData(val_loader, network, loss_fn, opt=None)
                
                train_val_results.append((train_result, val_result))
                time2 = time.time()
                print(f"epoch {i}, train loss: {train_result:.6f}, val loss: {val_result:.6f}, time: {time2-time1:.2f}s")
                # Step the learning rate scheduler
                LR_scheduler.step()

        # Save trained model
        network.to('cpu')
        network.device = 'cpu'
        print("Training completed.")
        if self.save_path:
            print("Saving model...")
            torch.save(network.state_dict(), f"{self.save_path}/{self.name}.pt")
        if self.is_plot:
            plot_train_vs_val_loss(train_val_results)
    
    def train_Diffusion(self):
        """
        Function responsible for training the network when the data generating process is set as Diffusion
        Utilizes the exponential utility loss, uses the learning rate scheduler function of the epochs nb,
        validates the epochs, saves the model if specifies and plots the train vs val loss if specified.
        """
        # these parameters should be given by the user !
        T = self.data_generator.dt * self.data_generator.sequence_length
        K = self.data_generator.K
        S0 = self.data_generator.S0
        # Define the name for saving the model

        scaled_returns_train = self.train_data
        scaled_returns_val = self.val_returns

        N = scaled_returns_train.shape[0]
        print("Starting training...")
        print(f"Training samples: {scaled_returns_train.shape[0]}, Validation samples: {scaled_returns_val.shape[0]}")
        
        # Prepare validation data loader
        tensor_val_data = torch.tensor(scaled_returns_val, dtype=torch.float32)
        val_dataset = torch.utils.data.TensorDataset(tensor_val_data)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # validation results and training results are kept in this array for plotting
        train_val_results = []
        
        # train_data is now a numpy array (num_windows, T, num_assets) from rolling windows
        # Calculate number of partitions based on available training data
        num_partitions = min(int(1e5/N), len(scaled_returns_train) // N)
        for part in range(0, num_partitions+1):
            index1 = part*int(N/2)
            index2 = (part+1)*int(N/2)
            # Prepare partitioned training data and loader
            eps  = 1e-6
            returns_batch =scaled_returns_train[index1:index2] # np.log((price_train[index1:index2][:, 1:, :] + eps) / (price_train[index1:index2][:, :-1, :] + eps))
            # Convert numpy array to tensor
            tensor_data = torch.tensor(returns_batch, dtype=torch.float32)
            train_data = torch.utils.data.TensorDataset(tensor_data)
            print(f"Diffusion data partition {part}/{num_partitions}, samples: {len(returns_batch)}")
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

            # Initialize network and loss function
            network = self.network
            loss_fn = Market_data_util.loss_exp_OCE(K, T, 1.3, self.idx_asset,X_max=True).to(self.device)

    
            opt = torch.optim.Adam([
                {'params': network.parameters()},  # Model parameters
            ], lr=self.learning_rate)
            # Learning rate scheduler
            LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.alpha_learning_rate)

            best_loss = float('inf')
            # Training loop
            for i in range(self.epoch_num):
                time1 = time.time()
                network.train()
                train_result = self.epoch_loader_Diffusion(train_loader, network, loss_fn, opt)
                
                # Validation evaluation
                network.eval()
                with torch.no_grad():
                    val_result = self.epoch_loader_Diffusion(val_loader, network, loss_fn, opt=None)
                
                train_val_results.append((train_result, val_result))
                time2 = time.time()
                print(f"epoch {i}, train loss: {train_result:.6f}, val loss: {val_result:.6f}, time: {time2-time1:.2f}s")
                # Step the learning rate scheduler
                LR_scheduler.step()

        # Save trained model
        network.to('cpu')
        network.device = 'cpu'
        print("Training completed.")
        if self.save_path:
            print("Saving model...")
            torch.save(network.state_dict(), f"{self.save_path}/{self.name}.pt")
        if self.is_plot:
            plot_train_vs_val_loss(train_val_results)




    def test(self):
        """
        Evaluates trained model on test data. Testing data is real financial market data. 
        First loads Auto-detects the data type (BS/market data/Diffusion), computes losses and mean PnL 
        and plots holdings if specified.
        """

        print("testing...")
        # Detect dataset type
        is_diffusion_data = self.test_returns is not None
        is_bs_data = isinstance(self.test_data, tuple) and len(self.test_data) == 2
        is_market_data = not is_diffusion_data and not is_bs_data
        if is_market_data:
            # Market data case: test_data is numpy array (batch, T, num_assets)
            loss_fn = Market_data_util.loss_exp_OCE(
                self.data_generator.K, 
                self.data_generator.T, 
                1.3, 
                self.idx_asset,
                X_max=True
            ).to(self.device)
            
            def performance(network, price):
                # Market data preprocessing: (B, T, N) -> (B, T-1, N) 
                eps = 1e-6

                if len(price.shape) == 2:
                    # Single asset case: (B, T) → (B, T-1, 1)
                    returns = torch.log((price[:, 1:] + eps) / (price[:, :-1] + eps)).unsqueeze(-1)
                else:
                    # Multi-asset case: (B, T, N) → (B, T-1, N)
                    returns = torch.log((price[:, 1:, :] + eps) / (price[:, :-1, :] + eps))
                holding = network(lag_returns(returns))

                name = self.data_generator.__class__.__name__.lower()
                print(name)
                if self.is_plot : 
                    _export_holdings_to_csv(holding,price,name = f'{name} RNN', plot_path = self.plot_path, idx_assets = self.data_generator.idx_assets_to_hedge)
                loss = loss_fn(holding,price)
                PnL = loss_fn.compute_PnL(holding,price)
                return loss.item(),PnL.mean()

            # Convert numpy array to tensor
            price = torch.tensor(self.test_data, dtype=torch.float32).to(self.device)
        elif is_diffusion_data:
            # Diffusion data case: test_data is torch tensor (batch, T, num_assets)
            loss_fn = Market_data_util.loss_exp_OCE(
                self.data_generator.K, 
                self.data_generator.T, 
                1.3, 
                self.idx_asset,
                X_max=True
            ).to(self.device)
            
            def performance(network, price):
                # Market data preprocessing: (B, T, N) -> (B, T-1, N) 
                returns = torch.tensor(self.test_returns, dtype=torch.float32).to(self.device)
                x = lag_returns(returns)
                holding = self.network(x)
                loss = loss_fn(holding,price)
                name = self.data_generator.__class__.__name__.lower()
                if self.is_plot : 
                    _export_holdings_to_csv(holding,price,name = f'{name} RNN', plot_path = self.plot_path, idx_assets = self.data_generator.idx_assets_to_hedge)

                PnL = loss_fn.compute_PnL(holding,price)
                return loss.item(),PnL.mean()

            # Convert numpy array to tensor
            price = torch.tensor(self.test_data, dtype=torch.float32).to(self.device)

     
        else : 
            loss_fn = BS_util.loss_exp_OCE(self.data_generator.K, self.data_generator.sigma, self.data_generator.T,1.3,self.idx_asset,X_max=True).to(self.device)
            # load the test data (now a tuple)
            price, regimes = self.test_data
            price   = price.to(self.device)
            regimes = regimes.to(self.device)
            def performance(network, price, regimes):
                # same preprocessing as in training
                eps = 1e-6
                returns = torch.log((price[:, 1:, :] + eps) / (price[:, :-1, :] + eps))         # (B, T-1, N)
                holding = network(lag_returns(returns), regimes[:, :-1])  # network expects price + regimes
                loss = loss_fn(holding, price)
                name_data_gen = self.data_generator.__class__.__name__.lower()
       
                if self.is_plot : 
                    _export_holdings_to_csv(holding,price,name = f'{name_data_gen} RNN', plot_path = self.plot_path, idx_assets = self.data_generator.idx_assets_to_hedge)

                # PnL: hedging gains minus option payoff
                dS = price[:, 1:, :] - price[:, :-1, :]
                pnl = (holding * dS[:, :, :holding.shape[-1]]).sum(dim=(1, 2))
                ST = price[:, -1, self.idx_asset]
                payoff = torch.maximum(ST - self.data_generator.K, torch.tensor(0.0, device=price.device))
                raw_pnl = pnl - payoff
                return loss.item(), raw_pnl.mean().item()

            
            
            # load the network 
            self.network.eval()
            self.network.load_state_dict(torch.load(f"{self.save_path}/{self.name}.pt"))
            
        # Call performance based on data type
        if is_market_data:
            print(f"The performance of the model is: {performance(self.network, price)}")
        elif is_diffusion_data:
            print(f"The performance of the model is: {performance(self.network, price)}")
        else:
            loss_val, pnl_val = performance(self.network, price, regimes)
            print(f"BS test loss: {loss_val}, mean PnL: {pnl_val}")





    def epoch_loader_BS(self, loader, network, loss_fn, opt=None):
        """
        Runs one epoch of training or evaluation for training.

        Args:
            loader: DataLoader providing batches of price data
            network: Neural network model.
            loss_fn: Loss function.
            opt: Optimizer (if training, None for evaluation).

        Returns:
            Average loss for the epoch.
        """
        total_loss=0.
        for batch in loader:
            # BS : only price
            price, regimes = batch   

            price   = price.to(self.device)
            regimes = regimes.to(self.device)
            eps = 1e-6
            # Price feature
            returns = torch.log((price[:, 1:, :] + eps) / (price[:, :-1, :] + eps))
            x = lag_returns(returns)
            holding = network(x, regimes[:, :-1])
            loss = loss_fn(holding, price)

            # Loss receives ONLY prices 
            loss = loss_fn(holding, price)
                
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item()
        total_loss /= len(loader)
        return total_loss

    def epoch_loader_MarketData(self, loader, network, loss_fn, opt=None):
        """
        Runs one epoch of training or evaluation for training.

        Args:
            loader: DataLoader providing batches of price data.
            network: Neural network model.
            loss_fn: Loss function.
            opt: Optimizer (if training, None for evaluation).

        Returns:
            Average loss for the epoch.
        """
        total_loss=0.
        for batch in loader:
            
            # Market data: only price (B, T, N) where N is number of assets
            price = batch   
            price = price[0].to(self.device)

            # Price feature: log prices (B, T-1, N)
            # If single asset, shape is (B, T, 1) already from data
            # If multi-asset, shape is (B, T, N)
            eps = 1e-6

            if len(price.shape) == 2:
                # Single asset case: (B, T) → (B, T-1, 1)
                returns = torch.log((price[:, 1:] + eps) / (price[:, :-1] + eps)).unsqueeze(-1)
            else:
                # Multi-asset case: (B, T, N) → (B, T-1, N)
                returns = torch.log((price[:, 1:, :] + eps) / (price[:, :-1, :] + eps))
                
            # Forward pass
            x = lag_returns(returns)
            holding = network(x)

            # Loss receives ONLY prices (including final price for payoff calculation)
            loss = loss_fn(holding, price)
                
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item()
        total_loss /= len(loader)
        return total_loss

    def epoch_loader_Diffusion(self, loader, network, loss_fn, opt=None):
        """
        Runs one epoch for Diffusion data. Reconstructs prices from log-returns, handles single/multi-asset cases,
        optimizes network if training (forward pass, backprop).
        Args : 
            - loader (object) : Data loader providing mini-batches 
            - Network (object) : the network responsible for training 
            - loss_fn (function) : the loss function used to optimize the network
            - opt (torch.object) : PyTorch optimizer responsible for updating the RNN weights 
        Returns : 
             - Av. loss per batch (float) : Average loss per batch for the entire epoch
        """

        total_loss = 0.

        def prices_from_log_returns(r, s0=100.0):
            B, T, N = r.shape
            s0_vec = torch.full((B, 1, N), s0, device=r.device)
            logS = torch.cumsum(r, dim=1)          # (B, T, N)
            S = s0_vec * torch.exp(logS)           # (B, T, N)
            return torch.cat([s0_vec, S], dim=1)   # (B, T+1, N)
        '''
        # mu/sigma stored in data_generator.build_data()
        mu = torch.tensor(self.data_generator.mu, dtype=torch.float32, device=self.device).view(1, 1, -1)
        sigma = torch.tensor(self.data_generator.sigma, dtype=torch.float32, device=self.device).view(1, 1, -1)
        '''
        for (scaled_returns,) in loader:
            x = scaled_returns.to(self.device)
            if x.ndim == 2:
                x = x.unsqueeze(-1)  # (B, T, 1)

            # build prices from *unscaled* log-returns
            r = x #* sigma + mu                   # (B, T, N)
            prices = prices_from_log_returns(r, s0=self.data_generator.S0)  # (B, T+1, N)
            # causal hedge: Δ_t uses info up to t-1
            holding = network(x)             # (B, T, N)

            loss = loss_fn(holding, prices)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss += loss.item()

        return total_loss / len(loader)


    # Learning rate schedule function
    def alpha_learning_rate(self, epoch):
        """
        Learning rate schedule that reduces the learning rate at specific epochs during training. 
        It controls the size of the steps the optimizer makes when training. 
        Args : 
            - epoch (int) : the epoch nb 
        
        Returns : 
            - Learning rate (int) : the learning rate to apply, function of the nb of epochs already done
        """
        
        if epoch<100:
            return 1
        elif epoch<200:
            return 0.1
        elif epoch<250:
            return 0.01
        else:
            return 0.001
