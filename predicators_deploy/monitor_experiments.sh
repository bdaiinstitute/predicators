#!/bin/bash

# This script monitors experiment progress in real-time using tmux
# Usage:
#   ./monitor_experiments.sh <machines_file>

if [ -z "$1" ]; then
    echo "Error: Machines file must be provided"
    echo "Usage: ./monitor_experiments.sh <machines_file>"
    exit 1
fi

MACHINES_FILE=$1

# Read machines from file
MACHINES=$(cat "$MACHINES_FILE")

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  macOS: brew install tmux"
    echo "  Ubuntu: sudo apt-get install tmux"
    exit 1
fi

# Kill existing monitoring session if it exists
tmux kill-session -t monitor 2>/dev/null

# Create a new tmux session
SESSION="monitor"
WINDOW="${SESSION}:0"

# Start new session in detached mode
tmux new-session -d -s $SESSION

# Function to add a pane and run command
add_pane() {
    local machine=$1
    local pane_cmd="echo '=== Monitoring $machine ==='; "
    pane_cmd+="echo 'Tailing all experiment logs...'; "
    pane_cmd+="ssh $machine 'tail -n 2 -f predicators/runlogs/*.txt 2>/dev/null || echo \"No log files found yet...\"'"
    
    # Split window and run command
    tmux split-window -t $WINDOW "$pane_cmd"
    # Set pane title
    tmux select-pane -T "$machine"
}

# Add first pane
FIRST_MACHINE=$(echo "$MACHINES" | head -n1)
tmux send-keys -t $WINDOW "echo '=== Monitoring $FIRST_MACHINE ==='; ssh $FIRST_MACHINE 'tail -n 2 -f predicators/runlogs/*.txt 2>/dev/null || echo \"No log files found yet...\"'" C-m

# Add panes for remaining machines
for machine in $(echo "$MACHINES" | tail -n +2); do
    add_pane "$machine"
done

# Arrange panes in tiled layout
tmux select-layout -t $WINDOW tiled

# Set window title
tmux rename-window -t $WINDOW "Experiment Logs"

# Add a status pane at the bottom
tmux split-window -v -p 20 -t $WINDOW
tmux send-keys -t $WINDOW "echo '=== Status Monitor ==='; while true; do clear; date; echo ''; for m in $MACHINES; do echo \"=== \$m ===\"; ssh \$m 'ps aux | grep \"python.*mock_experiments\" | grep -v grep || echo \"No experiments running\"'; echo ''; done; sleep 5; done" C-m

# Select the status pane
tmux select-pane -t $WINDOW.$(tmux list-panes -t $WINDOW | wc -l)

# Set mouse mode on
tmux set -g mouse on

# Attach to session
echo "Starting monitoring session..."
echo "Use Ctrl+B then D to detach (session will keep running)"
echo "Use 'tmux attach -t monitor' to reattach later"
echo "Use 'tmux kill-session -t monitor' to stop monitoring"
tmux attach-session -t $SESSION 