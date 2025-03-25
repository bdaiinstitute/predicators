#!/bin/bash

# This script runs and monitors local experiments with multiple seeds
# Usage:
#   ./run_local_experiments.sh <num_seeds> [planner_name] [env1 env2 ...]

# Set required environment variables
export PYTHONPATH="${PYTHONPATH:-$PWD}"  # Default to current directory if not set
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# Check FD_EXEC_PATH exists
if [ ! -d "$FD_EXEC_PATH" ]; then
    echo "Warning: FD_EXEC_PATH ($FD_EXEC_PATH) does not exist"
    echo "Please set FD_EXEC_PATH to the correct Fast Downward directory"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Error: Number of seeds must be provided"
    echo "Usage: ./run_local_experiments.sh <num_seeds> [planner_name] [env1 env2 ...]"
    echo "Example: ./run_local_experiments.sh 5 vlm_captioning_open_loop mock_spot_drawer_cleaning"
    exit 1
fi

NUM_SEEDS=$1
PLANNER=${2:-"vlm_captioning_open_loop"}  # Default planner
shift 2  # Remove first two arguments

# Default environments if none provided
if [ $# -eq 0 ]; then
    ENVS=("mock_spot_drawer_cleaning" "mock_spot_pick_place_two_cup" "mock_spot_sort_weight")
else
    ENVS=("$@")
fi

# Create runlogs directory if it doesn't exist
mkdir -p runlogs

# Create results_deploy directory if it doesn't exist
mkdir -p results_deploy

# Function to run a single experiment
run_experiment() {
    local env=$1
    local seed=$2
    local planner=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local results_dir="results_deploy/${timestamp}_${env}_${planner}"
    local logfile="runlogs/run_${env}_seed_${seed}.txt"
    
    echo "Starting experiment: env=$env seed=$seed planner=$planner"
    echo "Results will be saved in: $results_dir"
    
    # Create results directory
    mkdir -p "$results_dir"
    
    # Run experiment with both environment variable and command line argument
    RESULTS_DIR="$results_dir" python -u scripts/mock_experiments.py \
        --env "$env" \
        --seed "$seed" \
        --planner "$planner" \
        --results_dir "$results_dir" \
        &> "$logfile" &
}

# Kill any existing monitoring session
tmux kill-session -t local_monitor 2>/dev/null

# Show environment variables being used
echo "=== Environment Variables ==="
echo "PYTHONPATH: $PYTHONPATH"
echo "FD_EXEC_PATH: $FD_EXEC_PATH"
echo "PYTHONHASHSEED: $PYTHONHASHSEED"
echo "PYTHONUNBUFFERED: $PYTHONUNBUFFERED"
echo ""

# Run all experiments
for env in "${ENVS[@]}"; do
    for ((seed=0; seed<NUM_SEEDS; seed++)); do
        run_experiment "$env" "$seed" "$PLANNER"
    done
done

# Create monitoring session in tmux
SESSION="local_monitor"
tmux new-session -d -s $SESSION

# Create a window for each environment
for ((i=0; i<${#ENVS[@]}; i++)); do
    env="${ENVS[$i]}"
    if [ $i -eq 0 ]; then
        # First window already exists
        tmux rename-window -t $SESSION:0 "$env"
    else
        # Create new window for each additional environment
        tmux new-window -t $SESSION -n "$env"
    fi

    # Split window for each seed
    WINDOW="$SESSION:$i"
    FIRST=true
    for ((seed=0; seed<NUM_SEEDS; seed++)); do
        if [ "$FIRST" = true ]; then
            # First pane already exists
            FIRST=false
        else
            # Create new pane for additional seeds
            tmux split-window -t $WINDOW
        fi
        
        # Set up log monitoring command
        LOG_CMD="echo '=== $env (seed $seed) ==='; tail -f runlogs/run_${env}_seed_${seed}.txt"
        tmux send-keys -t $WINDOW.$seed "$LOG_CMD" C-m
    done
    
    # Arrange panes in tiled layout
    tmux select-layout -t $WINDOW tiled
done

# Create a status/help window
tmux new-window -t $SESSION -n "Status/Help"
STATUS_CMD="while true; do 
    clear
    echo '=== Process Status ==='
    date
    echo ''
    echo 'Running Processes:'
    ps aux | grep 'python.*mock_experiments' | grep -v grep || echo 'No experiments running'
    echo ''
    echo '=== Environment Variables ==='
    echo "PYTHONPATH: $PYTHONPATH"
    echo "FD_EXEC_PATH: $FD_EXEC_PATH"
    echo "PYTHONHASHSEED: $PYTHONHASHSEED"
    echo ''
    echo '=== Results Location ==='
    echo 'Results are saved in: ./results_deploy/<timestamp>_<env>_<planner>/'
    echo ''
    echo '=== Commands ==='
    echo '1. Process Management:'
    echo '   - Kill all experiments:  ./predicators_deploy/kill_local_experiments.sh'
    echo '   - Check processes:       ps aux | grep mock_experiments'
    echo ''
    echo '2. Tmux Navigation:'
    echo '   - Switch windows:        Ctrl+B then <number>'
    echo '   - Next window:           Ctrl+B then n'
    echo '   - Previous window:       Ctrl+B then p'
    echo '   - List windows:          Ctrl+B then w'
    echo '   - Scroll mode:           Ctrl+B then ['
    echo '   - Exit scroll mode:      q'
    echo ''
    echo '3. Session Management:'
    echo '   - Detach session:        Ctrl+B then d'
    echo '   - Kill session:          Ctrl+B then x, or run: tmux kill-session -t local_monitor'
    echo '   - Reattach later:        tmux attach -t local_monitor'
    echo ''
    echo '4. Experiment Logs:'
    echo '   - Location:              ./runlogs/'
    echo '   - View all logs:         ls -l runlogs/'
    echo '   - Tail specific log:     tail -f runlogs/run_<env>_seed_<N>.txt'
    sleep 5
done"
tmux send-keys -t $SESSION:$((${#ENVS[@]})) "$STATUS_CMD" C-m

# Set mouse mode on and select first window
tmux set -g mouse on
tmux select-window -t $SESSION:0

# Show initial instructions
echo "Starting monitoring session..."
echo "Windows:"
for ((i=0; i<${#ENVS[@]}; i++)); do
    echo "  $i: ${ENVS[$i]}"
done
echo "  $((${#ENVS[@]})): Status/Help"
echo ""
echo "Quick Commands:"
echo "  - Kill experiments:  ./predicators_deploy/kill_local_experiments.sh"
echo "  - Detach session:    Ctrl+B then d"
echo "  - Switch windows:    Ctrl+B then <number>"
echo ""
echo "See Status/Help window (last window) for more commands"

# Attach to session
tmux attach-session -t $SESSION 