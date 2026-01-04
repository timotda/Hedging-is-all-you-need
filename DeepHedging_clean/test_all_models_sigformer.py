import itertools
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import json
import re
import yaml


BASE_CONFIG = Path(__file__).parent / "config.yaml"

# ---- Choose your sweep values here ----
HEDGE_ASSETS_LIST = [
    ["AAPL"],
    ["AAPL", "MSFT"],
    ["AAPL", "GOOGL", "MSFT"],
    ["AAPL", "GOOGL", "MSFT", 'AMZN'],
    ["AAPL", "GOOGL", "MSFT", 'AMZN', 'BRK-B']
]

MODELS = ["SigFormer"]
DATA_MODES = [ "market_data", "diffusion", "bs_deephedging"]


def format_seconds(seconds: float) -> str:
    """Return a short human-readable duration."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"

def set_nested(cfg: dict, keys: list[str], value):
    """Set a nested config value, creating dicts when missing."""
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
    match1 = re.search(
        r"The performance of the model is:\s*\(([^,]+),\s*tensor\(([^,]+)", output
    )
    if match1:
        metrics["loss"] = float(match1.group(1))
        metrics["mean_pnl"] = float(match1.group(2))

    # Pattern for: "BS test loss: X, mean PnL: Y"
    match2 = re.search(r"BS test loss:\s*([^,]+),\s*mean PnL:\s*([^\s]+)", output)
    if match2:
        metrics["loss"] = float(match2.group(1))
        metrics["mean_pnl"] = float(match2.group(2))

    # Pattern for JAX SigFormer: separate lines "Test loss:" and "Test mean PnL:"
    match_loss = re.search(r"Test loss:\s*([-\d\.eE+]+)", output)
    if match_loss:
        metrics["loss"] = float(match_loss.group(1))

    match_pnl = re.search(r"Test mean PnL:\s*([-\d\.eE+]+)", output)
    if match_pnl:
        metrics["mean_pnl"] = float(match_pnl.group(1))

    return metrics


def main():
    """Run the SigFormer sweep across data modes and hedge universes."""
    base = yaml.safe_load(BASE_CONFIG.read_text())

    combos = list(itertools.product(DATA_MODES, MODELS, HEDGE_ASSETS_LIST))
    total_runs = len(combos)
    
    # Dictionary to store all results
    results = {}

    sweep_start = time.time()

    for idx, (data_mode, model, hedge_assets) in enumerate(combos, start=1):
        cfg = yaml.safe_load(BASE_CONFIG.read_text())  # fresh copy each run

        set_nested(cfg, ["Hedging", "hedge_assets"], hedge_assets)
        set_nested(cfg, ["Hedging", "model"], model)
        set_nested(cfg, ["Hedging", "data_mode"], data_mode)

        
        run_name = f"ha={','.join(hedge_assets)}__model={model}__mode={data_mode}"
        set_nested(cfg, ["run_name"], run_name)

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "config_override.yaml"
            tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

            main_py = Path(__file__).parent / "main.py"
            cmd = [sys.executable, str(main_py), "--config", str(tmp_path)]

            # Pre-run ETA based on completed runs
            if idx == 1:
                print(f"[{idx}/{total_runs}] Running: {run_name} | ETA: calculating after first run finishes")
            else:
                elapsed_total = time.time() - sweep_start
                avg_per_run = elapsed_total / (idx - 1)
                eta_remaining = avg_per_run * (total_runs - idx + 1)
                print(
                    f"[{idx}/{total_runs}] Running: {run_name} | "
                    f"Estimated remaining: {format_seconds(eta_remaining)}"
                )
            
            try:
                run_start = time.time()
                # Stream output live while capturing for metric parsing
                with subprocess.Popen(
                    cmd,
                    cwd=Path(__file__).parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as proc:
                    captured_lines = []
                    for line in proc.stdout:
                        print(line, end="")
                        captured_lines.append(line)
                    proc.wait()
                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(
                            proc.returncode, cmd, output="".join(captured_lines)
                        )

                full_output = "".join(captured_lines)

                # Parse performance metrics from output
                metrics = parse_performance(full_output)
                
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

            # Progress and ETA
            elapsed_total = time.time() - sweep_start
            avg_per_run = elapsed_total / idx
            remaining_runs = total_runs - idx
            eta_remaining = avg_per_run * remaining_runs
            print(
                f"Progress: {idx}/{total_runs} | "
                f"Last run: {format_seconds(time.time() - run_start)} | "
                f"ETA remaining: {format_seconds(eta_remaining)}"
            )
    
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
    main()
    with open("test_results.json", "r") as f:
        from plots.plot import plot_pnl_vs_hedge_assets

        results_dict = json.load(f)
        plot_pnl_vs_hedge_assets(results_dict)
