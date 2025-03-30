#!/bin/bash

# This script runs a single environment with optional planner specification
# Usage:
#   ./run_single_env.sh <env_name> [planner_name]
# Example:
#   ./run_single_env.sh mock_spot_drawer_cleaning oracle
#   ./run_single_env.sh mock_spot_sort_weight vlm_closed_loop

# Check if environment name is provided
if [ -z "$1" ]; then
    echo "Error: Environment name must be provided"
    echo "Usage: ./run_single_env.sh <env_name> [planner_name]"
    echo "Available environments:"
    echo "  - mock_spot_drawer_cleaning"
    echo "  - mock_spot_pick_place_two_cup"
    echo "  - mock_spot_sort_weight"
    echo "  - mock_spot_cup_emptiness"
    exit 1
fi

ENV_NAME=$1
PLANNER_NAME=$2

# Base command
CMD="python scripts/mock_experiments.py --env $ENV_NAME"

# Add planner if specified
if [ ! -z "$PLANNER_NAME" ]; then
    CMD="$CMD --planner $PLANNER_NAME"
fi

# Run the command
echo "Running command: $CMD"
$CMD 