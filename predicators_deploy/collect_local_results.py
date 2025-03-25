"""Collect and organize results from local experiments.

This script:
1. Collects results from results_deploy/
2. Organizes them by environment and seed
3. Creates a summary of the experiments including success rates
4. Aggregates metrics across seeds

Usage:
    python predicators_deploy/collect_local_results.py [--output_dir results_collected]
"""

import argparse
import os
import shutil
import json
import yaml
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
import glob
import numpy as np

# Define optimal path lengths for each environment
OPTIMAL_PATH_LENGTHS = {
    "mock_spot_pick_place_two_cup": 4,
    "mock_spot_drawer_cleaning": 9,
    "mock_spot_cup_emptiness": 10,  # NOTE: still bug in planning for this
    "mock_spot_sort_weight": 14,
}

def calculate_spl(num_solved: int, total_steps: int, optimal_length: int) -> float:
    """Calculate Success weighted by Path Length (SPL)."""
    if num_solved == 1 and total_steps > 0:
        return optimal_length * num_solved / total_steps
    return 0.0

def aggregate_metrics(yaml_files: List[str], env_name: str) -> Dict[str, Any]:
    """Aggregate metrics across multiple yaml result files."""
    all_metrics = []
    spl_values = []
    optimal_length = OPTIMAL_PATH_LENGTHS[env_name]
    
    for f in yaml_files:
        with open(f, 'r') as yf:
            data = yaml.safe_load(yf)
            if 'results' in data:
                metrics = data['results']
                all_metrics.append(metrics)
                
                # Calculate SPL for this run
                num_solved = metrics['num_solved']
                total_steps = metrics['total_steps']
                spl = calculate_spl(num_solved, total_steps, optimal_length)
                spl_values.append(spl)
    
    if not all_metrics:
        return {}
        
    # Initialize aggregated metrics
    agg_metrics = {
        'num_seeds': len(all_metrics),
        'metrics_per_seed': all_metrics,
        'aggregated': {},
        'spl': {
            'values': spl_values,
            'mean': float(np.mean(spl_values)),
            'std': float(np.std(spl_values)),
            'min': float(np.min(spl_values)),
            'max': float(np.max(spl_values))
        }
    }
    
    # Get all metric keys from first result
    metric_keys = [k for k in all_metrics[0].keys() if not k.startswith('PER_TASK_')]
    
    # Aggregate each metric
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            agg_metrics['aggregated'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
    # Calculate success rate
    if 'num_solved' in metric_keys and 'num_total' in metric_keys:
        success_rates = [m['num_solved'] / m['num_total'] for m in all_metrics]
        agg_metrics['success_rate'] = {
            'mean': float(np.mean(success_rates)),
            'std': float(np.std(success_rates)),
            'min': float(np.min(success_rates)),
            'max': float(np.max(success_rates))
        }
    
    return agg_metrics

def aggregate_metrics_by_planner_and_task(yaml_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics grouped by planner and task."""
    # Structure: planner -> task -> metrics
    grouped_metrics = defaultdict(lambda: defaultdict(list))
    
    for f in yaml_files:
        with open(f, 'r') as yf:
            data = yaml.safe_load(yf)
            if 'config' in data and 'results' in data:
                # Extract planner name from config
                planner = data['config'].get('approach', 'unknown')
                # Extract task/env name from config
                task = data['config'].get('env', 'unknown')
                
                # Store metrics for this planner/task combination
                metrics = data['results']
                grouped_metrics[planner][task].append(metrics)
    
    # Aggregate metrics for each planner/task combination
    final_metrics = {}
    for planner, tasks in grouped_metrics.items():
        final_metrics[planner] = {}
        for task, metrics_list in tasks.items():
            optimal_length = OPTIMAL_PATH_LENGTHS.get(task, 1)
            spl_values = []
            success_rates = []
            
            for metrics in metrics_list:
                num_solved = metrics.get('num_solved', 0)
                total_steps = metrics.get('total_steps', 0)
                num_total = metrics.get('num_total', 1)
                
                # Calculate SPL
                spl = calculate_spl(num_solved, total_steps, optimal_length)
                spl_values.append(spl)
                
                # Calculate success rate
                success_rates.append(num_solved / num_total)
            
            final_metrics[planner][task] = {
                'num_seeds': len(metrics_list),
                'success_rate': {
                    'mean': float(np.mean(success_rates)),
                    'std': float(np.std(success_rates)),
                },
                'spl': {
                    'values': spl_values,
                    'mean': float(np.mean(spl_values)),
                    'std': float(np.std(spl_values)),
                }
            }
    
    return final_metrics

def collect_results(output_dir: str = "results_collected") -> None:
    """Collect and organize results from local experiments."""
    
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    
    # Create directories for results and logs
    results_dir = os.path.join(timestamped_dir, "results")
    logs_dir = os.path.join(timestamped_dir, "logs")
    metrics_dir = os.path.join(timestamped_dir, "metrics")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize metadata with proper structure
    metadata = {
        "timestamp": timestamp,
        "environments": defaultdict(lambda: {
            "seeds": [],
            "num_results": 0  # Initialize as integer
        })
    }
    
    # Collect all runlog files and corresponding yaml files
    runlog_files = glob.glob("runlogs/run_*_seed_*.txt")
    yaml_files = glob.glob("results_deploy/*/*.yaml")  # Look in timestamped subdirectories
    
    # Group yaml files by environment and get their timestamps
    env_to_yaml_files = defaultdict(list)
    for yaml_file in yaml_files:
        # Extract environment and timestamp from directory name
        dirname = os.path.basename(os.path.dirname(yaml_file))
        if "_mock_" in dirname:
            # Extract timestamp from the start of dirname (assuming format: YYYYMMDD_HHMMSS_...)
            file_timestamp = dirname.split("_mock_")[0]
            env = dirname.split("_mock_")[1].split("_")[0]
            env = f"mock_{env}"
            env_to_yaml_files[env].append((yaml_file, file_timestamp))
    
    # Process each runlog file
    for log_file in runlog_files:
        filename = os.path.basename(log_file)
        parts = filename.replace(".txt", "").split("_seed_")
        if len(parts) != 2:
            continue
            
        env_part = parts[0].replace("run_", "")
        seed = int(parts[1])
        
        # Look for corresponding results in results_deploy to get timestamp and planner
        result_dirs = glob.glob(f"results_deploy/*_{env_part}_*")
        for result_dir in result_dirs:
            if os.path.isdir(result_dir):
                # Extract timestamp from result directory
                dir_timestamp = os.path.basename(result_dir).split("_")[0]
                
                # Get planner name from yaml file
                yaml_files = glob.glob(os.path.join(result_dir, "*.yaml"))
                planner_name = "unknown"
                if yaml_files:
                    with open(yaml_files[0], 'r') as yf:
                        data = yaml.safe_load(yf)
                        if 'config' in data:
                            planner_name = data['config'].get('approach', 'unknown')
                
                # Create new log filename with timestamp and planner
                new_log_filename = f"run_{env_part}_planner_{planner_name}_seed_{seed}_{dir_timestamp}.txt"
                log_dest = os.path.join(logs_dir, new_log_filename)
                shutil.copy2(log_file, log_dest)
                
                env_seed_dir = os.path.join(results_dir, f"{env_part}_planner_{planner_name}_seed{seed}_{dir_timestamp}")
                os.makedirs(env_seed_dir, exist_ok=True)
                
                # Copy all files from the results directory
                for item in os.listdir(result_dir):
                    src = os.path.join(result_dir, item)
                    dest = os.path.join(env_seed_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dest)
                
                metadata["environments"][env_part]["seeds"].append(seed)
                metadata["environments"][env_part]["num_results"] += 1
    
    # Collect and group results by planner and task
    yaml_files = glob.glob("results_deploy/*/*.yaml")
    grouped_metrics = aggregate_metrics_by_planner_and_task(yaml_files)
    
    # Save detailed metrics
    metrics_file = os.path.join(metrics_dir, "metrics_by_planner_and_task.yaml")
    with open(metrics_file, 'w') as f:
        yaml.dump(grouped_metrics, f, default_flow_style=False)
    
    # Generate summary
    summary_file = os.path.join(timestamped_dir, "collection_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Results Collection Summary\n")
        f.write(f"========================\n")
        f.write(f"Collected on: {timestamp}\n\n")
        
        for planner, tasks in grouped_metrics.items():
            f.write(f"Planner: {planner}\n")
            f.write("=" * (len(planner) + 9) + "\n")
            
            for task, metrics in tasks.items():
                f.write(f"\nTask: {task}\n")
                f.write(f"  Seeds: {metrics['num_seeds']}\n")
                f.write(f"  Success Rate: {metrics['success_rate']['mean']:.2%} ± {metrics['success_rate']['std']:.2%}\n")
                f.write(f"  SPL: {metrics['spl']['mean']:.3f} ± {metrics['spl']['std']:.3f}\n")
            f.write("\n")
    
    print(f"\nResults collected and organized in: {timestamped_dir}")
    print(f"See {summary_file} for collection summary")
    print("\nResults by Planner and Task:")
    for planner, tasks in grouped_metrics.items():
        print(f"\nPlanner: {planner}")
        for task, metrics in tasks.items():
            print(f"\n  Task: {task}")
            print(f"    Success Rate: {metrics['success_rate']['mean']:.2%} ± {metrics['success_rate']['std']:.2%}")
            print(f"    SPL: {metrics['spl']['mean']:.3f} ± {metrics['spl']['std']:.3f}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=str, default="results_collected",
                       help="Base directory for collected results")
    args = parser.parse_args()
    
    collect_results(args.output_dir)

if __name__ == "__main__":
    main() 