import yaml
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
NEW_FORMAT = True
results_folder = 'results'
method_names = ["Random", "VLM Planning Directly", "VLM Captioning Then Planning", 'VLM Labelling/LLM Planning', 'Ours']
NUM_RUNS = 10

if(NEW_FORMAT):
    methods = ['random_options__', 'vlm_open_loop__mpc', 'vlm_captioning__', 'llm_open_loop__mpc', 'oracle__']
    format_index = 3
else:
    methods = ['random_options__', 'vlm_open_loop__', 'vlm_captioning__', 'llm_open_loop__', 'oracle__']
    format_index = 2
    
def read_yaml_files():
    # Define the methods we're looking for
    # Dictionary to store results for each method
    results = {method: {
        'successes': 0, 
        'total': 0,
        'spl_values': [0 for _ in range(NUM_RUNS)]  # Add list to store SPL values
    } for method in methods}
    
    # Get all yaml files in the results directory
    results_dir = Path(results_folder)
    yaml_files = list(results_dir.glob('*.yaml'))
    
    # Process each yaml file
    for file_path in yaml_files:
        # Check if file contains any of our method names
        file_name = file_path.name
        matching_method = None
        for method in methods:
            if method in file_name:
                matching_method = method
                break
        
        if matching_method is None:
            continue
            
        # Read the yaml file
        with open(file_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if data is None:
                    continue
                
                seed = int(str(file_path).split("__")[format_index])
                # Get the success information
                num_solved = data.get('results', {}).get('num_solved', 0)
                total_steps = data.get('results', {}).get('total_steps', 0)
                
                # Update success counts
                results[matching_method]['total'] += 1
                if num_solved == 1:
                    results[matching_method]['successes'] += 1
                
                # Calculate and store SPL
                if num_solved == 1 and total_steps > 0:
                    spl = 4 * num_solved / total_steps  # optimal_length = 4
                else:
                    spl = 0
                    
                    
                results[matching_method]['spl_values'][seed] = spl
                
            except yaml.YAMLError:
                continue

    return results


def plot_spl(results):
    methods = list(results.keys())
    means = []
    errors = []
    stds = []
    success_rates = []
    
    for method in methods:
        values = results[method]['spl_values']
        if values:
            mean = np.mean(values)
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            stds.append(np.std(values, ddof=1))
            means.append(mean)
            errors.append(std_err)
            success_rates.append(results[method]['successes']/NUM_RUNS)
    
    print("====Results====")
    for method, mean, std, success_rate in zip(method_names, means, stds, success_rates):
        print()
        print(method)
        print("SPL Means: "+str(mean))
        print("SPL Stds: "+str(std))
        print("Success Rates: "+str(success_rate))
        print()
    # Create the bar plot with error bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(method_names, means, yerr=errors, capsize=5)
    
    # Customize the plot
    plt.title('Success Weighted Path Length (SPL) by Method')
    plt.ylabel('SPL')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('method_spl.png')
    plt.close()

def main():
    results = read_yaml_files()
    plot_spl(results)
    

if __name__ == "__main__":
    main()
