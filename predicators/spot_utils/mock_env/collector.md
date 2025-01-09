# Mock Environment Collector

This document describes how to collect data for the mock Spot environment using both manual and robot-based methods.

## Overview

The mock environment collector helps you:
1. Generate all possible states and transitions for a task using planning
2. Collect RGB-D images for each state
3. Build a complete transition graph

## Automatic State Generation

The collector uses planning to automatically generate all possible states and transitions:

```python
from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators.spot_utils.mock_env.mock_env_creator_spot import SpotMockEnvCreator

# Initialize environment and creator
env = MockSpotEnv(data_dir="path/to/data_dir")
creator = ManualMockEnvCreator("path/to/data_dir")  # or SpotMockEnvCreator

# Get a task (e.g., pick and place)
task = env.get_train_tasks()[0]

# Generate all possible states and transitions
states = creator.generate_states_and_transitions(task)
```

The generated states include:
- Initial state
- Goal state
- All intermediate states reachable through operators
- Valid transitions between states

## Manual Image Collection

For manual collection, you'll need:
1. Pre-recorded RGB-D images
2. Object information for each state
3. Gripper state information

Example workflow:

```python
# Initialize manual creator
creator = ManualMockEnvCreator("path/to/data_dir")

# Generate states
states = creator.generate_states_and_transitions(task)

# Collect images for each state
for state_id, ground_atoms in states.items():
    print(f"\nState {state_id}:")
    for atom in ground_atoms:
        print(f"  {atom}")
        
    # Get image path from user
    rgb_path = input("Enter path to RGB-D image for this state: ")
    
    # Extract object information from ground atoms
    objects_in_view = []
    objects_in_hand = []
    gripper_open = True
    
    for atom in ground_atoms:
        if atom.predicate.name == "InView":
            objects_in_view.append(atom.objects[0].name)
        elif atom.predicate.name == "InHand":
            objects_in_hand.append(atom.objects[0].name)
        elif atom.predicate.name == "HandEmpty":
            gripper_open = True
    
    # Add state with image
    creator.add_state_from_images(
        rgb_path=rgb_path,
        objects_in_view=objects_in_view,
        objects_in_hand=objects_in_hand,
        gripper_open=gripper_open
    )
```

## Robot-Based Collection

For collection using the Spot robot:

```python
# Initialize robot creator
creator = SpotMockEnvCreator("spot.example.com", "path/to/data_dir")

# Generate states
states = creator.generate_states_and_transitions(task)

# Collect images for each state
for state_id, ground_atoms in states.items():
    print(f"\nState {state_id}:")
    for atom in ground_atoms:
        print(f"  {atom}")
        
    input("Position robot and press Enter to capture...")
    
    # Extract object information from ground atoms
    objects_in_view = []
    objects_in_hand = []
    gripper_open = True
    
    for atom in ground_atoms:
        if atom.predicate.name == "InView":
            objects_in_view.append(atom.objects[0].name)
        elif atom.predicate.name == "InHand":
            objects_in_hand.append(atom.objects[0].name)
        elif atom.predicate.name == "HandEmpty":
            gripper_open = True
    
    # Add state using robot cameras
    creator.add_state_from_robot(
        objects_in_view=objects_in_view,
        objects_in_hand=objects_in_hand,
        gripper_open=gripper_open
    )
```

## Command Line Interface

You can also use the command line interface:

```bash
# For manual collection
python -m predicators.spot_utils.mock_env.mock_env_creator_manual \
    --path_dir spot_mock_data \
    --dir_name manual_test

# For robot-based collection
python -m predicators.spot_utils.mock_env.mock_env_creator_spot \
    --hostname spot.example.com \
    --path_dir spot_mock_data \
    --dir_name spot_test
```

## Best Practices

1. Image Collection:
   - Ensure consistent lighting conditions
   - Keep camera angle consistent for similar states
   - Capture clear views of all relevant objects

2. State Organization:
   - Use descriptive state IDs
   - Document object configurations
   - Verify transitions are valid

3. Data Management:
   - Back up collected data regularly
   - Version control your environment data
   - Document any special conditions or requirements

## Troubleshooting

Common issues and solutions:

1. Image Loading Errors:
   - Verify file paths are correct
   - Check image format (should be .npy)
   - Ensure RGB and depth dimensions match

2. Robot Connection Issues:
   - Check network connectivity
   - Verify robot credentials
   - Ensure robot is powered on and ready

3. State Generation Issues:
   - Check task definition is valid
   - Verify operators are properly defined
   - Look for cycles in the transition graph 