# Mock Spot Environment

This directory contains tools for creating and managing mock environments for the Spot robot. The mock environment allows for testing and development without requiring access to the physical robot.

## Overview

The mock environment is a POMDP where:
- States are latent (we don't know actual poses)
- Observations are RGB-D images + gripper state + object detections
- Actions can succeed or fail based on available images
- Transitions between states are stored in a graph structure

## Components

### Environment (`mock_spot_env.py`)
The main environment class that implements the POMDP:
- Defines types, predicates, and operators at module level
- Uses a graph-based state representation with RGB-D images
- Provides methods to add states and transitions
- Stores data persistently on disk

### Mock Environment Creators
The mock environment system provides two ways to create mock environments:

1. Manual Creation (`ManualMockEnvCreator`):
   - Create mock environments using existing RGB-D images
   - Manually specify object positions and gripper states
   - Useful for creating test environments or reproducing specific scenarios

2. Robot-Based Creation (`SpotMockEnvCreator`):
   - Create mock environments using the actual Spot robot
   - Automatically captures RGB-D images and robot state
   - Useful for recording real-world scenarios for later testing

Both creators inherit from `MockEnvCreatorBase`, which provides common functionality for:
- Managing states and transitions
- Generating states using planning
- Validating operators and transitions
- Saving and loading environment data

### Perceiver (`mock_spot_perceiver.py`)
- Provides mock perception capabilities
- Manages RGB-D images and object detections
- Tracks gripper state and objects in view/hand
- Handles image saving and loading

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

## Usage

### Manual Creation

```python
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator

# Initialize creator with path to store environment data
creator = ManualMockEnvCreator("path/to/data_dir")

# Add a state using RGB-D images
creator.add_state_from_images(
    rgb_path="path/to/rgb.npy",
    objects_in_view=["cube1", "target1"],
    objects_in_hand=["cube1"],
    gripper_open=False
)

# Add transitions between states
creator.add_transition("0", "1", "pick")  # state_id, next_state_id, operator_name
```

### Robot-Based Creation

```python
from predicators.spot_utils.mock_env.mock_env_creator_spot import SpotMockEnvCreator

# Initialize creator with robot hostname and data directory
creator = SpotMockEnvCreator("spot.example.com", "path/to/data_dir")

# Add a state by capturing current robot state
creator.add_state_from_robot(
    objects_in_view=["cube1", "target1"],
    objects_in_hand=["cube1"],
    gripper_open=False
)

# Add transitions between states
creator.add_transition("0", "1", "pick")
```

### Basic Environment Usage

```python
from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.perception.mock_spot_perceiver import MockSpotPerceiver

# Initialize environment
env = MockSpotEnv(data_dir="path/to/data_dir")

# Initialize perceiver
perceiver = MockSpotPerceiver(data_dir="path/to/data_dir")

# Get initial observation
obs = env.reset()

# Take actions
action = env.action_space.sample()  # Replace with your action selection
next_obs, reward, done, info = env.step(action)
```

## Directory Structure

The mock environment data is stored in the following structure:

```
data_dir/
├── images/
│   ├── state_0/
│   │   ├── rgb.npy
│   │   └── depth.npy
│   ├── state_1/
│   │   ├── rgb.npy
│   │   └── depth.npy
│   └── ...
└── graph.json  # Contains state and transition information
```

## Additional Documentation

- [Collector Guide](collector.md) - How to collect environment data
- [Tasks Guide](tasks.md) - How to create and use tasks
- [Development Guide](development.md) - Best practices and guidelines
- [Testing Guide](testing.md) - Information about testing
- [Image Collection](spot_collect_rgbd.md) - Details on collecting RGB-D images 