# Predicators Deployment Scripts

This directory contains scripts for deploying and managing experiments across multiple machines.

## Local Experiments

### Running Experiments Locally

1. `run_local_experiments.sh`
   ```bash
   ./run_local_experiments.sh num_seeds planner_name
   ```
   Runs experiments locally with multiple seeds. Features:
   - Automatic process management
   - Live monitoring through tmux
   - Status/Help window showing commands and process status
   - Automatic PYTHONHASHSEED=0 setting

2. `kill_local_experiments.sh`
   ```bash
   ./kill_local_experiments.sh
   ```
   Kills all locally running experiments and cleans up tmux sessions.

3. `collect_local_results.py`
   ```bash
   python collect_local_results.py [--output_dir results_collected]
   ```
   Collects and analyzes results from local experiments:
   - Organizes results by environment and seed
   - Computes success rates and other metrics
   - Generates detailed summaries
   - Creates aggregated metrics per environment

### Results Directory Structure

- `results/` - Results from running `main.py` directly
- `results_deploy/` - Results from deployment scripts
  ```
  results_deploy/
  ├── YYYYMMDD_HHMMSS_env1_planner/  # Timestamped experiment results
  │   ├── metrics.yaml
  │   └── ...
  ├── YYYYMMDD_HHMMSS_env2_planner/
  └── ...
  ```
- `results_collected/` - Organized results after collection
  ```
  results_collected_YYYYMMDD_HHMMSS/
  ├── results/              # Organized by env/seed
  │   ├── env1_seed0/
  │   ├── env1_seed1/
  │   └── ...
  ├── logs/                # All experiment logs
  ├── metrics/             # Aggregated metrics
  │   ├── env1_metrics.yaml
  │   └── env2_metrics.yaml
  ├── collection_metadata.json
  └── collection_summary.txt
  ```
- `runlogs/` - Live experiment logs
  ```
  runlogs/
  ├── run_env1_seed_0.txt
  ├── run_env1_seed_1.txt
  └── ...
  ```

### Collection Process
The collection script:
1. Finds all experiment results in `results_deploy/`
2. Groups results by environment and seed
3. Computes aggregate metrics (success rates, etc.)
4. Generates summary with:
   - Success rates per environment
   - Number of seeds completed
   - Experiment statistics
5. Saves everything in a timestamped `results_collected_YYYYMMDD_HHMMSS` directory

### Local Testing Scripts

1. `run_single_env.sh`
   ```bash
   ./run_single_env.sh env_name [planner_name]
   ```
   Run single environment locally.

2. `run_multi_envs.sh`
   ```bash
   ./run_multi_envs.sh "env1 env2 ..." [planner_name]
   ```
   Run multiple environments locally.

## Remote Deployment

### Prerequisites

1. SSH Configuration:
   - Either set up SSH config for passwordless access to machines
   - Or have SSH key for OpenStack instances

2. Machine List:
   - Create a `machines.txt` file with one hostname/IP per line
   - Example:
     ```
     enthoo
     entwo
     ```

### Core Deployment Scripts

1. `run_vlm_captioning_open_loop.sh`
   ```bash
   ./run_vlm_captioning_open_loop.sh machines.txt [num_seeds=10]
   ```
   Runs VLM captioning experiments on specific mock environments in parallel.

2. `deploy_with_ssh_config.sh`
   ```bash
   ./deploy_with_ssh_config.sh machines.txt env1 [env2 ...] [--planner planner_name]
   ```
   General deployment script using SSH config.

3. `deploy_multi.sh`
   ```bash
   ./deploy_multi.sh machines.txt username env1 [env2 ...] [--planner planner_name]
   ```
   Legacy deployment script for OpenStack instances.

### Monitoring and Management

1. `monitor_experiments.sh`
   ```bash
   ./monitor_experiments.sh machines.txt
   ```
   Shows live output from all machines in tmux panes.

2. `kill_experiments.sh`
   ```bash
   ./kill_experiments.sh machines.txt
   ```
   Kills all running experiments on specified machines.

3. `collect_script.py`
   ```bash
   # SSH config mode (default)
   python collect_script.py --machines machines.txt
   
   # OpenStack mode
   python collect_script.py --machines machines.txt --sshkey ~/.ssh/cloud.key --user ubuntu
   ```
   Collects results and logs from remote machines into `results_collected/`.

## Typical Workflows

### Local Workflow

1. **Start Local Experiments**
   ```bash
   ./run_local_experiments.sh 5 vlm_captioning_open_loop
   ```

2. **Monitor in tmux**
   - Switch between environments: `Ctrl+B` then window number
   - View Status/Help window: `Ctrl+B` then `0`
   - Scroll in panes with mouse

3. **Kill if needed**
   ```bash
   ./kill_local_experiments.sh
   ```

4. **Collect & Analyze Results**
   ```bash
   python collect_local_results.py
   ```

### Remote Workflow

1. **Start Remote Experiments**
   ```bash
   ./run_vlm_captioning_open_loop.sh machines.txt
   ```

2. **Monitor Progress**
   ```bash
   ./monitor_experiments.sh machines.txt
   ```

3. **If needed, Kill Experiments**
   ```bash
   ./kill_experiments.sh machines.txt
   ```

4. **Collect Results**
   ```bash
   python collect_script.py --machines machines.txt
   ```

## Troubleshooting

1. **No log files found**
   - Make sure experiments have started
   - Check if correct directory paths are used
   - Verify SSH access to machines

2. **SSH Issues**
   - Check SSH config or key permissions
   - Verify machine hostnames/IPs
   - Test SSH connection manually

3. **Process Management**
   - Use appropriate kill script to clean up hanging processes
   - Check tmux sessions
   - Monitor system resources (CPU/memory)

4. **Results Collection Issues**
   - Verify experiments completed successfully
   - Check correct results directory is being used
   - Ensure sufficient disk space for collection 