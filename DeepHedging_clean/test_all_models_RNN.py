import itertools
import subprocess
import sys
import tempfile
from pathlib import Path
import json
import re
import yaml


BASE_CONFIG = Path(__file__).parent / "config.yaml"

HEDGE_ASSETS_LIST = [
    ["AAPL"],
    ["AAPL", "MSFT"],
    ["AAPL", "GOOGL", "MSFT"],
    ["AAPL", "GOOGL", "MSFT", 'AMZN'],
    ["AAPL", "GOOGL", "MSFT", 'AMZN', 'BRK-B']
]

MODELS = ["RNN_BN_simple"]
DATA_MODES = [ "diffusion"]
# DATA_MODES = ["market_data", "diffusion", "bs_deephedging"]

def set_nested(cfg: dict, keys: list[str], value):
    """Set a nested config value, creating dicts as needed."""
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def parse_performance(output: str):
    """Extract performance metrics from model output."""
    metrics = {}
    
    # Pattern for: "The performance of the model is: (loss, mean_pnl)"
    pattern1 = r"The performance of the model is:\s*\(([^,]+),\s*tensor\(([^,]+)"
    match1 = re.search(pattern1, output)
    if match1:
        metrics['loss'] = float(match1.group(1))
        metrics['mean_pnl'] = float(match1.group(2))
        return metrics
    
    # Pattern for: "BS test loss: X, mean PnL: Y"
    pattern2 = r"BS test loss:\s*([^,]+),\s*mean PnL:\s*([^\s]+)"
    match2 = re.search(pattern2, output)
    if match2:
        metrics['loss'] = float(match2.group(1))
        metrics['mean_pnl'] = float(match2.group(2))
        return metrics
    
    return metrics


def run_training_testing_all(verbose=False):
    """Run the full sweep of model/data/hedge combos and log results."""
    base = yaml.safe_load(BASE_CONFIG.read_text())

    combos = list(itertools.product(DATA_MODES, MODELS, HEDGE_ASSETS_LIST))
    
    # Dictionary to store all results
    results = {}

    for data_mode, model, hedge_assets in combos:
        cfg = yaml.safe_load(BASE_CONFIG.read_text())  

        set_nested(cfg, ["Hedging", "hedge_assets"], hedge_assets)
        set_nested(cfg, ["Hedging", "model"], model)
        set_nested(cfg, ["Hedging", "data_mode"], data_mode)

        # nice run identifier (optional)
        run_name = f"ha={','.join(hedge_assets)}__model={model}__mode={data_mode}"
        set_nested(cfg, ["run_name"], run_name)

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "config_override.yaml"
            tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

            main_py = Path(__file__).parent / "main.py"
            cmd = [sys.executable, str(main_py), "--config", str(tmp_path)]
            print("Running:", run_name)
            
            try:
                # Always capture output so metrics can be parsed
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    cwd=Path(__file__).parent,
                    capture_output=True,
                    text=True
                )
                if verbose:
                    print("="*70)
                    print(result.stdout)
                    if result.stderr:
                        print(result.stderr, file=sys.stderr)
                    print("="*70)
                # Parse performance metrics from output
                metrics = parse_performance(result.stdout)
                
                # Store results
                results[run_name] = {
                    'data_mode': data_mode,
                    'model': model,
                    'hedge_assets': hedge_assets,
                    'status': 'success',
                    'metrics': metrics
                }
                
                print(f"✓ Completed: {run_name}")
                if metrics:
                    print(f"  Metrics: {metrics}")
                    
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {run_name}")
                results[run_name] = {
                    'data_mode': data_mode,
                    'model': model,
                    'hedge_assets': hedge_assets,
                    'status': 'failed',
                    'error': str(e)
                }
    
    # Save results to JSON file
    results_path = Path(__file__).parent / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("Summary of Results:")
    print(f"{'='*70}")
    for run_name, data in results.items():
        status = "✓" if data['status'] == 'success' else "✗"
        print(f"{status} {run_name}")
        if 'metrics' in data and data['metrics']:
            for key, val in data['metrics'].items():
                print(f"    {key}: {val}")
    
    return results


if __name__ == "__main__":
    # Set verbose=True to see all training output
    run_training_testing_all(verbose=True)
    with open("test_results.json", "r") as f:
        from plots.plot import plot_pnl_vs_hedge_assets

        results_dict = json.load(f)
        plot_pnl_vs_hedge_assets(results_dict)
