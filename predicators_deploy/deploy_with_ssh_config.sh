#!/bin/bash

# This script deploys experiments using SSH config (no username/key needed)
# Usage:
#   ./deploy_with_ssh_config.sh <machines_file> <env_name>
#   ./deploy_with_ssh_config.sh <machines_file> --envs "env1 env2 ..."
#   ./deploy_with_ssh_config.sh <machines_file> <env_name> --planner <planner_name>
# Example:
#   ./deploy_with_ssh_config.sh machines.txt mock_spot_drawer_cleaning
#   ./deploy_with_ssh_config.sh machines.txt --envs "mock_spot_drawer_cleaning mock_spot_sort_weight"
#   ./deploy_with_ssh_config.sh machines.txt mock_spot_drawer_cleaning --planner oracle

# Check required arguments
if [ "$#" -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: ./deploy_with_ssh_config.sh <machines_file> <env_name>"
    echo "       ./deploy_with_ssh_config.sh <machines_file> --envs \"env1 env2 ...\""
    echo "       ./deploy_with_ssh_config.sh <machines_file> <env_name> --planner <planner_name>"
    echo "Available environments (from mock_spot_env.py):"
    echo "  - mock_spot_drawer_cleaning"
    echo "  - mock_spot_pick_place_two_cup"
    echo "  - mock_spot_sort_weight"
    echo "  - mock_spot_cup_emptiness"
    echo "Available planners:"
    echo "  - oracle"
    echo "  - random"
    echo "  - llm_closed_loop"
    echo "  - vlm_closed_loop"
    echo "  - vlm_captioning"
    exit 1
fi

MACHINES_FILE=$1
shift

# Base command
CMD="python deploy_script.py --machines $MACHINES_FILE --use_ssh_config"

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --envs)
            CMD="$CMD --envs \"$2\""
            shift 2
            ;;
        --planner)
            CMD="$CMD --planner $2"
            shift 2
            ;;
        *)
            # Single environment case
            CMD="$CMD --envs \"$1\""
            shift
            ;;
    esac
done

# Run the command
echo "Running command: $CMD"
eval $CMD 