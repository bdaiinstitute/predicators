#!/bin/bash

# This script runs multiple environments with optional planner specification
# Usage:
#   ./run_multi_envs.sh "env1 env2 ..." [planner_name]
# Example:
#   ./run_multi_envs.sh "mock_spot_drawer_cleaning mock_spot_sort_weight"
#   ./run_multi_envs.sh "mock_spot_drawer_cleaning mock_spot_sort_weight" oracle

# Check if environments are provided
if [ -z "$1" ]; then
    echo "Error: Environment names must be provided"
    echo "Usage: ./run_multi_envs.sh \"env1 env2 ...\" [planner_name]"
    echo "Available environments:"
    echo "  - mock_spot_drawer_cleaning"
    echo "  - mock_spot_pick_place_two_cup"
    echo "  - mock_spot_sort_weight"
    echo "  - mock_spot_cup_emptiness"
    exit 1
fi

ENV_NAMES=$1
PLANNER_NAME=$2

# Base command
CMD="python scripts/mock_experiments.py --envs $ENV_NAMES"

# Add planner if specified
if [ ! -z "$PLANNER_NAME" ]; then
    CMD="$CMD --planner $PLANNER_NAME"
fi

# Run the command
echo "Running command: $CMD"
$CMD 