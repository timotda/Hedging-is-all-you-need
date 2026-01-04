import pandas as pd
import numpy as np
import argparse
import yaml
import sys
import os
from BS.BS_util import run_BS_deephedging
from MarketData.Market_data_util import run_marketdata_deephedging
from Diffusion.Diffusion_util import run_diffusion_deephedging, merge_diffusion_data
from Delta_hedge.delta_hedge import run_delta_hedge
from Cross_validation.optuna_hypparam_marketdata import run_marketdata_crossval, get_marketdata_study_name
from Cross_validation.optuna_hypparam_BS import run_BS_crossval, get_BS_study_name
from Cross_validation.optuna_hypparam_diffusion import run_diffusion_crossval, get_diffusion_study_name


def load_yaml_config(path):
    """This function loads the YAML configuration file. (if it founds it)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r") as f:
        config = yaml.full_load(f)
    return config

def parse_args():
    """Parses command line arguments.
    The only argument that can be used is the path config file. By default the path is config.yaml but 
    can use any config file (that follows the same structure as config.yaml)
    """
    parser = argparse.ArgumentParser(description="Deep Hedging Automation Pipeline")

    # Config File
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the YAML configuration file"
    )

    return parser.parse_args()

def main():
    """
    This main function will run the training and testing for the given method in config.yaml file. 
    Depending on the chosen configuration, it will run a specified dataset for a specified training method.
    """
    args = parse_args()
    config = load_yaml_config(args.config)

    underlying = config['Hedging']['underlying']
    hedge_assets = config['Hedging']['hedge_assets']
    is_plot = config['plotting']['is_plot']
    plot_path = config['plotting']['plot_path']

    model = config['Hedging']['model']
    data_mode = config['Hedging']['data_mode']
    crossval = config['Hedging']['cross_validation']
    
    # --- Load Data Common to All Modes ---
    data_path = config['dataset']['stock_prices_path']
    print(f"--- Loading data from {data_path} ---")
    
    try:
        stock_prices = pd.read_csv(data_path)
        if 'Date' in stock_prices.columns:
            stock_prices.set_index('Date', inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"--- Starting | Mode: {data_mode} ---")

    
    #Run model based on data_mode
    if data_mode == "market_data":
        if model is None:
            print("Error: You must specify --model (either 'SigFormer' or 'RNN_BN_simple' ) when running market_data.")
            sys.exit(1)
        if crossval:
            if model == 'SigFormer':
                    print("Error: cross-valdiation cannot be ran on SigFormer.")
                    sys.exit(1)
            if model == 'RNN_BN_simple':
                print("Cross validation is running")
                # if the cross-validation file doesn't already exit: run the cross validation
                current_dir = os.path.dirname(os.path.abspath(__file__))
                optuna_dir = os.path.join(current_dir, 'Cross_validation/Cross_validation_results')
                study_name = get_marketdata_study_name(underlying, hedge_assets)
                optuna_file = os.path.join(optuna_dir, f'{study_name}_best_params.csv')
                if not os.path.exists(optuna_file):
                    breakpoint()
                    run_marketdata_crossval(asset_hedged=underlying, assets_to_hedge=hedge_assets, stock_prices_path=data_path, n_trials=config['cross_validation']['n_trials'], n_trials_per_study=config['cross_validation']['n_trails_per_study'])
                
        print(f"Running Market Data Hedging with architecture: {model}")
        run_marketdata_deephedging(asset_hedged=underlying, stock_prices=stock_prices, assets_to_hedge=hedge_assets, name_model=model, is_plot=is_plot, plot_path = plot_path)

    elif data_mode == "bs_deephedging":
        print("Running BS Deep Hedging...")
        if crossval:
            if model == 'SigFormer':
                    print("Error: cross-valdiation cannot be ran on SigFormer.")
                    sys.exit(1)
            if model == 'RNN_BN_simple':
                print("Cross validation is running")
                # if the cross-validation file doesn't already exit: run the cross validation
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                optuna_dir = os.path.join(parent_dir, 'Cross-validation/Cross_validation_results')
                study_name = get_BS_study_name(underlying, hedge_assets)
                optuna_file = os.path.join(optuna_dir, f'{study_name}_best_params.csv')
                if not os.path.exists(optuna_file):
                    run_BS_crossval(asset_hedged=underlying, assets_to_hedge=hedge_assets, stock_prices_path=data_path, n_trials=config['cross_validation']['n_trials'], n_trials_per_study=config['cross_validation']['n_trails_per_study'])
                
        run_BS_deephedging(asset_hedged=underlying, stock_prices=stock_prices, assets_to_hedge=hedge_assets, name_model=model, is_plot=is_plot, plot_path = plot_path)

    elif data_mode == "delta_hedge":
        print("Running Delta Hedge...")
        run_delta_hedge(asset_hedged=underlying, stock_prices=stock_prices,assets_to_hedge=hedge_assets,name_model = data_mode, is_plot = is_plot, plot_path = plot_path)

    elif data_mode == "diffusion":
        print("Running Diffusion...")
        if crossval:
            if model == 'SigFormer':
                    print("Error: cross-valdiation cannot be ran on SigFormer.")
                    sys.exit(1)
            if model == 'RNN_BN_simple':
                print("Cross validation is running")
                # if the cross-validation file doesn't already exit: run the cross validation
                current_dir = os.path.dirname(os.path.abspath(__file__))
                optuna_dir = os.path.join(current_dir, 'Cross_validation/Cross_validation_results')
                study_name = get_diffusion_study_name(underlying, hedge_assets)
                optuna_file = os.path.join(optuna_dir, f'{study_name}_best_params.csv')
                if not os.path.exists(optuna_file):
                    run_diffusion_crossval(asset_hedged=underlying, assets_to_hedge=hedge_assets, stock_prices_path=data_path, n_trials=config['cross_validation']['n_trials'], n_trials_per_study=config['cross_validation']['n_trails_per_study'])
                
        run_diffusion_deephedging(asset_hedged=underlying, assets_to_hedge=hedge_assets, name_model=model, is_plot = is_plot, plot_path = plot_path)

    elif data_mode == "all_models":
        print(">>> Running Delta Hedge...")
        run_delta_hedge(asset_hedged=underlying, stock_prices=stock_prices,assets_to_hedge=[underlying])

        print(">>> Running BS Deep Hedging...")
        run_BS_deephedging(asset_hedged=underlying, stock_prices=stock_prices, assets_to_hedge=hedge_assets)

        print(">>> Running Market Data Hedging (RNN_BN_simple)...")
        run_marketdata_deephedging(asset_hedged=underlying,stock_prices=stock_prices,assets_to_hedge=hedge_assets,name_model="RNN_BN_simple",is_plot = is_plot, path = plot_path)

        print(">>> Running Market Data Hedging (SigFormer)...")
        run_marketdata_deephedging(asset_hedged=underlying,stock_prices=stock_prices,assets_to_hedge=hedge_assets, name_model="SigFormer")

if __name__ == "__main__":
    main()


