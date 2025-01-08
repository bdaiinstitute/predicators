# Mock Spot Environment

This document describes the mock environment system for the Spot robot, which allows creating image-based environments with predefined transitions.

## Overview

The mock environment system consists of two main components:
1. `MockSpotEnv`: A partially observable environment that uses pre-captured images and a transition graph
2. `MockEnvCreator`: A helper tool to create mock environments by building the transition graph

The key idea is to:
1. Capture images of different states manually
2. Define the transitions between states using predefined operators
3. Use this to test planning without requiring the real robot

## Environment Structure

### States
Each state in the environment consists of:
- Set of camera images
- Set of objects in view
- Set of objects in hand
- Gripper state (open/closed)

### Transitions
Transitions are defined by:
- Source state ID
- Operator name
- Target state ID
- Success flag (whether the action succeeds or fails)

### Operators
The environment provides a set of predefined operators:
- `MoveToReachObject`: Move to reach an object
- `MoveToHandViewObject`: Move to view an object with hand camera
- `PickObjectFromTop`: Pick up an object from above
- `PlaceObjectOnTop`: Place held object on a surface
- `DropObjectInside`: Drop held object into a container
- `SweepIntoContainer`: Sweep objects into a container

## Creating a Mock Environment

### Option 1: Interactive Creation
Use the interactive tool:
```python
from predicators.spot_utils.mock_env_creation import create_mock_env_interactive

env = create_mock_env_interactive()
```

This will guide you through:
1. Adding the goal state
2. Adding intermediate states
3. Defining transitions
4. Visualizing the graph
5. Checking missing transitions

### Option 2: Programmatic Creation
Create the environment directly:
```python
from predicators.envs.mock_spot_env import MockSpotEnv
from pathlib import Path

env = MockSpotEnv()

# Add states
env.add_state(
    state_id="initial",
    image_path=Path("path/to/image.png"),
    objects_in_view={"cup", "table"},
    objects_in_hand=set(),
    gripper_open=True
)

# Add transitions
env.add_transition(
    from_state="initial",
    action="PickObjectFromTop",
    to_state="holding_cup",
    success=True
)
```

## Using with a Planner

### Required Components

1. Task Definition:
   - Create a task class inheriting from `BaseEnv`
   - Define operators with preconditions and effects
   - Example:
   ```python
   class SpotPickPlaceTask(BaseEnv):
       def __init__(self):
           self.operators = {
               "PickObjectFromTop": {
                   "preconditions": ["HandEmpty", "ObjectReachable"],
                   "effects": ["Holding"]
               },
               # ... more operators
           }
   ```

2. Ground Truth Transitions:
   - The mock environment provides ground truth transitions through its transition graph
   - No sampling is needed since transitions are deterministic
   - Failed transitions are explicitly marked

3. State Mapping:
   - Images map to symbolic states through object sets
   - Example: `{"cup", "table"}` in view → `On(cup, table)`

### Integration Steps

1. Create Task:
   ```python
   task = SpotPickPlaceTask()
   ```

2. Create Mock Environment:
   ```python
   env = create_mock_env_interactive()  # or create programmatically
   ```

3. Link Task and Environment:
   ```python
   task.set_environment(env)
   ```

4. Plan:
   ```python
   initial_obs = env.reset()
   plan = planner.plan(task, initial_obs)
   ```

## Tips and Best Practices

1. Image Organization:
   - Use descriptive state IDs
   - Store images with clear naming convention
   - Example: `pick_cup_s1.png`, `pick_cup_s2.png`

2. State Design:
   - Create states for key points in manipulation
   - Include failure states for robustness
   - Consider partial observability

3. Transition Graph:
   - Make graph complete for all relevant actions
   - Include both success and failure transitions
   - Use visualization to verify coverage

4. Testing:
   - Test all paths to goal
   - Verify operator preconditions/effects
   - Check handling of failed actions 