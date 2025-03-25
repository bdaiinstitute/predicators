"""Plot results from local experiments.

This script:
1. Reads results from results_deploy/
2. Calculates success rates and SPL per environment and planner
3. Generates plots and prints statistics
"""

import yaml
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any
import glob

def read_yaml_files(results_dir: str = "results_deploy") -> Dict[str, Any]:
    """Read and organize results from yaml files."""
    # Dictionary to store results by env and planner
    results = {}
    
    # Find all yaml files in results_deploy subdirectories
    yaml_files = glob.glob(f"{results_dir}/*/*.yaml")
    
    for file_path in yaml_files:
        # Extract env and planner from directory name
        dir_name = os.path.basename(os.path.dirname(file_path))
        timestamp, env, planner = dir_name.split("_", 2)
        
        # Initialize if not exists
        if env not in results:
            results[env] = {}
        if planner not in results[env]:
            results[env][planner] = {
                'successes': 0,
                'total': 0,
                'total_steps': [],
                'spl_values': []
            }
            
        # Read yaml file
        with open(file_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if data is None or 'results' not in data:
                    continue
                
                metrics = data['results']
                num_solved = metrics.get('num_solved', 0)
                num_total = metrics.get('num_total', 0)
                total_steps = metrics.get('total_steps', 0)
                
                # Update statistics
                results[env][planner]['total'] += 1
                if num_solved == num_total:  # All tasks solved
                    results[env][planner]['successes'] += 1
                    
                # Store total steps for SPL calculation
                results[env][planner]['total_steps'].append(total_steps)
                
                # Calculate SPL (Success weighted by Path Length)
                # Assuming optimal path length is 4 (as in original script)
                if num_solved == num_total and total_steps > 0:
                    spl = 4 * num_solved / total_steps
                else:
                    spl = 0
                results[env][planner]['spl_values'].append(spl)
                
            except yaml.YAMLError:
                continue
                
    return results

def plot_results(results: Dict[str, Any], output_dir: str = ".") -> None:
    """Generate plots for each environment."""
    
    # Print detailed statistics
    print("\n=== Detailed Results ===")
    for env in results:
        print(f"\nEnvironment: {env}")
        for planner in results[env]:
            data = results[env][planner]
            num_runs = data['total']
            success_rate = data['successes'] / num_runs if num_runs > 0 else 0
            spl_values = data['spl_values']
            spl_mean = np.mean(spl_values) if spl_values else 0
            spl_std = np.std(spl_values) if spl_values else 0
            
            print(f"\nPlanner: {planner}")
            print(f"Success Rate: {success_rate:.2%} ({data['successes']}/{num_runs})")
            print(f"SPL: {spl_mean:.3f} ± {spl_std:.3f}")
            
        # Create plot for this environment
        plt.figure(figsize=(10, 6))
        planners = list(results[env].keys())
        spl_means = []
        spl_errors = []
        
        for planner in planners:
            values = results[env][planner]['spl_values']
            mean = np.mean(values) if values else 0
            std_err = np.std(values) / np.sqrt(len(values)) if values else 0
            spl_means.append(mean)
            spl_errors.append(std_err)
        
        # Create bar plot
        bars = plt.bar(planners, spl_means, yerr=spl_errors, capsize=5)
        
        # Customize plot
        plt.title(f'Success Weighted Path Length (SPL) - {env}')
        plt.ylabel('SPL')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, err in zip(bars, spl_errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + err,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spl_{env}.png'))
        plt.close()

def main() -> None:
    """Main function to read and plot results."""
    results = read_yaml_files()
    plot_results(results)
    
if __name__ == "__main__":
    main() 