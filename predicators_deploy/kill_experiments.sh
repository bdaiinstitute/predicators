#!/bin/bash

# This script kills all running experiments on the specified machines
# Usage:
#   ./kill_experiments.sh <machines_file>

if [ -z "$1" ]; then
    echo "Error: Machines file must be provided"
    echo "Usage: ./kill_experiments.sh <machines_file>"
    exit 1
fi

MACHINES_FILE=$1

# Read machines from file
MACHINES=$(cat "$MACHINES_FILE")

for MACHINE in $MACHINES; do
    echo "=== Killing processes on $MACHINE ==="
    # Kill all python processes running mock_experiments.py
    ssh "$MACHINE" 'pkill -f "python.*mock_experiments.py"'
    # Also kill any hanging tmux sessions
    ssh "$MACHINE" 'tmux kill-server 2>/dev/null || true'
done

echo "Done killing processes. You can now restart experiments." 