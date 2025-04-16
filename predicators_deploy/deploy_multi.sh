#!/bin/bash

# This script deploys experiments to multiple machines
# Usage:
#   ./deploy_multi.sh <machines_file> <username> "env1 env2 ..." [planner_name]
# Example:
#   ./deploy_multi.sh machines.txt myuser "mock_spot_drawer_cleaning mock_spot_sort_weight"
#   ./deploy_multi.sh machines.txt myuser "mock_spot_drawer_cleaning mock_spot_sort_weight" oracle

# Check required arguments
if [ "$#" -lt 3 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: ./deploy_multi.sh <machines_file> <username> \"env1 env2 ...\" [planner_name]"
    echo "Available environments:"
    echo "  - mock_spot_drawer_cleaning"
    echo "  - mock_spot_pick_place_two_cup"
    echo "  - mock_spot_sort_weight"
    echo "  - mock_spot_cup_emptiness"
    exit 1
fi

MACHINES_FILE=$1
USERNAME=$2
ENV_NAMES=$3
PLANNER_NAME=$4

# Base command
CMD="python deploy_script.py --machines $MACHINES_FILE --username $USERNAME --envs $ENV_NAMES"

# Add planner if specified
if [ ! -z "$PLANNER_NAME" ]; then
    CMD="$CMD --planner $PLANNER_NAME"
fi

# Run the command
echo "Running command: $CMD"
$CMD 