# Deep Hedging Framework

A comprehensive deep learning framework for options hedging that combines neural networks with financial derivatives pricing. This project implements and compares multiple hedging strategies including traditional delta hedging and advanced deep reinforcement learning approaches.

## Overview

This project implements a **Deep Hedging** framework for options portfolio management, providing automated hedging strategies that learn optimal policies from market data. The framework supports multiple data generation modes, neural network architectures, and includes hyperparameter optimization capabilities.


## Project Structure

```
deep-hedging/
├── DeepHedging_clean/          # Main implementation
│   ├── config.yaml             # Configuration file
│   ├── main.py                 # Entry point
│   ├── Deephedging.py          # Deep hedging class associated with the RNN
│   ├── Deephedging_JAX.py      # Deep hedging class associated with the SigFormer
│   ├── requirements.txt        # Python dependencies
│   ├── test_all_models.py      # Model testing suite for RNN
│   ├── test_all_models_sigformer.py  # Model testing suite for SigFormer
│   ├── interactive_training.ipynb    # Interactive notebook
│   │
│   ├── BS/                     # Black-Scholes implementation
│   │   ├── BS_generator.py     # BS path generation
│   │   ├── BS_util.py          # Utilities and training
│   │   └── trained_models/     # Pre-trained RNN models for BS
│   │
│   ├── MarketData/             # Market data processing
│   │   ├── Market_data_generator.py
│   │   ├── Market_data_util.py
│   │   └── trained_models/     # Pre-trained RNN models for market data
│   │
│   ├── Diffusion/              # Diffusion model implementation
│   │   ├── Diffusion_generator.py
│   │   ├── Diffusion_util.py
│   │   └── trained_models/     # Pre-trained RNN models for diffusion
│   │
│   ├── Delta_hedge/            # Classical delta hedging
│   │   └── delta_hedge.py      # BSM delta hedge implementation
│   │
│   ├── SigFormer/              # Signature-based transformer
│   │   ├── model.py            # SigFormer architecture
│   │   ├── layer.py            # Custom layers
│   │   └── utils.py
│   │
│   ├── Cross_validation/       # Hyperparameter optimization
│   │   ├── optuna_hypparam_BS.py
│   │   ├── optuna_hypparam_diffusion.py
│   │   └── optuna_hypparam_marketdata.py
│   │
│   ├── DataLoader/             # Data loading utilities
│   │   └── DataLoader.py
│   │
│   ├── Data/                   # Historical stock data
│   │   └── value_weighted_returns.csv
│   │
│   ├── plots/                  # Plotting utilities
│   │   ├── compare_aapl_distributions.py
│   │   ├── compare_all_distributions.py
│   │   ├── plot.py
│   │   ├── plot_distrib_hedging_path.ipynb
│   │   └── policy_map.ipynb
│   │
│   ├── SigDiffusion_Generation/  # Signature diffusion path generation
│   │   ├── compute_signatures.py
│   │   ├── data_loading_utils.py
│   │   ├── invert_signatures.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── ode_lib.py
│   │   ├── requirements.txt
│   │   ├── sample.py
│   │   ├── train.py
│   │   ├── training_utils.py
│   │   ├── config/
│   │   │   └── stock_returns.yaml
│   │   ├── data/
│   │   │   ├── create_npy.ipynb
│   │   │   └── generated_paths/  # Generated stock return paths
│   │   ├── evaluation/
│   │   │   ├── discriminative_metrics.py
│   │   │   ├── evaluation.ipynb
│   │   │   ├── metric_utils.py
│   │   │   └── predictive_metrics.py
│   │   └── signature_inversion_utils/
│   │       ├── fourier_inversion.py
│   │       └── free_lie_algebra.py
│   │
│   └── trained_models/         # Pre-trained SigFormer models
```

## Getting Started

### Prerequisites

All libraries used can be found in `requirements.txt`. Please ensure all until section SigDiffusion are in your environment before running.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deep-hedging
```

2. Install dependencies:
```bash
pip install torch numpy pandas scipy matplotlib pyyaml scikit-learn
pip install optuna joblib  # For hyperparameter optimization
pip install jax jaxlib equinox  # For JAX/SigFormer implementation
```

3. Navigate to the main directory:
```bash
cd DeepHedging_clean
```

## 📖 Usage

### Basic Configuration

Edit the `config.yaml` file to set your hedging parameters:

```yaml
seed: 42

Hedging:
  underlying: 'AAPL'              # Choose 1 underlying asset
  hedge_assets: ['AAPL', 'GOOGL'] # Choose 1 or more assets used for hedging
  model: RNN_BN_simple            # Choose the model: RNN_BN_simple or SigFormer
  data_mode: market_data          # Choose the dataset: market_data, delta_hedge, diffusion, bs_deephedging
  cross_validation: False         # Set to True to run cross-validation

dataset:
  stock_prices_path: "Data/stocks_close_prices_2008_2025.csv" # path to pre-dowloaded stock prices 
  seq_len: 30 # Size of Windows
  dim: 1

training:
  batch_size: 128

plotting : 
  is_plot : False # Set to true to see all plots
  plot_path : "plots" # folder where saved plots will end up

... # see config.yaml to set up default parameters for the RNN

```

### Running the Framework
**Before anything do these steps**:
1. Download the stock prices for the assets you need. (see Data)
2. configure the config.yaml file

**Runing one method at a time:**
```bash
python main.py --config config.yaml
```
This will run the training and testing for your chosen configuration.

**Running all  RNN methods a once: (takes longer)**
```bash
python test_all_models.py #RNN
python test_all_models_sigformer.py #SigFormer
```
you can still select what models you would liek to run at the begginign of the file: 
 ```bash
 # ---- Setup here ! ----
HEDGE_ASSETS_LIST = [
    ["AAPL"],
    ["AAPL", "MSFT"],
    ["AAPL", "GOOGL", "MSFT"]
    # add more if wanted...
]

# remove what you don't want to run
DATA_MODES = ["market_data", "diffusion", "bs_deephedging", "delta_hedge"]

```

### Only testing (skip training)
In all `_util.py` files, comment:

```bash
  # Example for diffusion
  
  # Comment out this line
  deephedging.train_Diffusion()

  # Comment out this line
  dh.train(epochs=parameters['epochs'], batch_size=parameters['batch_size'])
```

### Data Modes

The framework supports four data generation modes:

1. **`market_data`**: Uses real historical stock prices
2. **`bs_deephedging`**: Black-Scholes simulated paths with regime switching
3. **`diffusion`**: Diffusion model-based price generation
4. **`delta_hedge`**: Classical Black-Scholes delta hedging (benchmark)

### Model Architectures

- **`RNN_BN_simple`**: Recurrent Neural Network with batch normalization
- **`SigFormer`**: Signature-based transformer for path-dependent features

## 📊 Data

To download stock prices run `DataLoader\DataLoader.py`
```bash
cd DataLoader
python DataLoader.py
```
Stock tickers and Dates can be selected as you wish.

For market regimes, value weighted returns of the S&P500 are needed. We provide them from 2008 to 2025 (excluded).
They can be found in `Data/value_weighted_returns.csv`. 
If you require these returns for more dates, go [here](https://wrds-www.wharton.upenn.edu). (Requires an account)

To generate the SigDiffusion paths: In the `SigDiffusion_Generation` folder
1. In `data`: add the returns of your stock in an `.npy`format. 
2. You can then run the `main.py`file using the following line:
```bash
python main.py run-all aapl_returns config/stocks_returns.yaml
```
- The first argument: `run-all` runs the entire SigDiffusion pipeline.
- The second argument: `aapl_returns` is the name of the file where the paths will be saved.
- The thrid argument: `stocks_returns.yaml` is the config file. 
3. The resultings paths can be found under `\data\generated_paths`. These paths should now be move to `DeepHedging_clean\Data`.


Note ⚠️: 
- We have already added the necessary return files for ["AAPL", "GOOGL", "MSFT", "AMZN", "BRK-B"] in `\data`.
- SigDiffusion requires the package `iisignature` to run.
- SigDiffusion code comes from the paper Barancikova, B., Huang, Z., and Salvi, C. SigDiffusions:
Score-Based Diffusion Models for Time Series via Log-
Signature Embeddings. arXiv preprint arXiv:2406.10354,
2024.





## 🔬 Hyperparameter Optimization

The framework includes automated hyperparameter tuning using Optuna:

```python
# Enable in config.yaml
Hedging:
  cross_validation: True

cross_validation:
  n_trials: 30
  n_trails_per_study: 15
```

Optimized parameters include:
- Learning rate
- Batch size
- Network hidden dimensions
- Dropout rates
- Number of layers

Results are saved in `Cross_validation/Cross_validation_results/`



