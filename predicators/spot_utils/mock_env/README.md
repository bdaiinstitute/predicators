# Mock Spot Environment

This directory contains a mock environment system for the Spot robot that allows building and using transition graphs with RGB-D images. The system supports both manual image collection and automated collection using the Spot robot.

## Overview

The mock environment is a POMDP where:
- States are latent (we don't know actual poses)
- Observations are RGB-D images + gripper state + object detections
- Actions can succeed or fail based on available images
- Transitions between states are stored in a graph structure

## Components

- `mock_spot_env.py`: The main environment class that implements the POMDP
- `mock_creator_manual.py`: Tool for manually creating mock environments with provided images
- `mock_creator_spot.py`: Tool for creating mock environments using the Spot robot
- `mock_spot_task.py`: Task definitions and operators for the mock environment

## Directory Structure

```
spot_mock_data/
├── images/           # RGB-D images for each state
│   ├── rgb_*.npy    # RGB images as numpy arrays
│   └── depth_*.npy  # Depth images as numpy arrays
└── graph.json       # Transition graph and state data
```

## Usage

### Manual Mock Environment Creation

Use this when you want to create a mock environment with your own images:

```bash
# Create a new mock environment
python -m predicators.spot_utils.mock_env.mock_creator_manual \
    --path_dir spot_mock_data \
    --dir_name manual_test

# Add states and transitions
# The tool will prompt for:
# - RGB and depth images
# - Objects in view
# - Objects being held
# - Gripper state
# - Transitions between states
```

### Spot-Based Mock Environment Creation

Use this when you want to create a mock environment using the Spot robot:

```bash
# Set Spot credentials
export BOSDYN_CLIENT_USERNAME=<username>
export BOSDYN_CLIENT_PASSWORD=<password>

# Create a new mock environment
python -m predicators.spot_utils.mock_env.mock_creator_spot \
    --hostname <spot_ip> \
    --path_dir spot_mock_data \
    --dir_name spot_test

# The tool will:
# 1. Connect to Spot
# 2. Allow you to move the robot and capture images
# 3. Detect objects in view
# 4. Build the transition graph
```

### Using the Mock Environment

```python
from predicators.envs import get_env
from predicators.spot_utils.mock_env.mock_spot_task import MockSpotTask

# Create environment and task
env = get_env("mock_spot")
task = MockSpotTask()

# Set environment for task
task.set_environment(env)

# Reset environment
obs = env.reset("train", 0)

# Take actions
action = Action(...)  # Create action with name in extra_info
next_obs, reward, done = env.step(action)
```

## Data Formats

### RGB Images
- Format: NumPy array (.npy)
- Shape: (H, W, 3)
- Type: uint8 or float32 (0-1 range)
- Color space: RGB

### Depth Images
- Format: NumPy array (.npy)
- Shape: (H, W)
- Type: float32
- Units: Meters

### Graph Data
- Format: JSON
- Contains:
  - Transitions between states
  - Object detections
  - Gripper state
  - References to image files

## Creating Custom Tasks

1. Create a new task class inheriting from `MockSpotTask`
2. Define operators with preconditions and effects
3. Create states and transitions using either mock creator
4. Use the task with the mock environment

Example:
```python
class MySpotTask(MockSpotTask):
    def __init__(self) -> None:
        super().__init__()
        # Add custom operators
        self.operators.update({
            "CustomAction": MockSpotOperator(
                name="CustomAction",
                parameters=["?obj"],
                preconditions=["ObjectInView(?obj)"],
                effects_add=["Holding(?obj)"],
                effects_delete=["HandEmpty"]
            )
        })
``` 