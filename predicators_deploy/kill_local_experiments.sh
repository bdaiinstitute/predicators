#!/bin/bash

# This script kills all running local experiments
# Usage:
#   ./kill_local_experiments.sh

echo "=== Killing Local Experiments ==="

# Kill all running experiments and clean up results
echo "Killing all running experiments..."
pkill -f "python.*mock_experiments"

# Kill any existing monitoring session
echo "Killing tmux monitoring session..."
tmux kill-session -t local_monitor 2>/dev/null

# Clean up results directories
echo "Cleaning up results..."
rm -rf results_deploy/*
rm -rf runlogs/*

# Verify all processes are killed
echo "Verifying no experiments are running..."
ps aux | grep "python.*mock_experiments" | grep -v grep || echo "All experiments killed"

echo "Done! All experiments killed and results cleaned up."
echo "You can verify with: ps aux | grep 'python.*mock_experiments' | grep -v grep" 