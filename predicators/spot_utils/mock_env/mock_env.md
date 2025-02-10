# Mock Spot Environment

This document describes the mock environment system for the Spot robot, which allows creating image-based environments with predefined transitions. The environment also supports belief space planning capabilities for tasks involving uncertainty.

## Overview

The mock environment system consists of two main components:
1. `MockSpotEnv`: A partially observable environment that uses pre-captured images and a transition graph
2. `MockEnvCreator`: A helper tool to create mock environments by building the transition graph

The key idea is to:
1. Capture images of different states manually
2. Define the transitions between states using predefined operators
3. Use this to test planning without requiring the real robot

## Basic Environment Structure

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

### Basic Operators
The environment provides these physical manipulation operators:
- `MoveToReachObject`: Move to reach an object
- `MoveToHandViewObject`: Move to view an object with hand camera
- `PickObjectFromTop`: Pick up an object from above
- `PlaceObjectOnTop`: Place held object on a surface
- `DropObjectInside`: Drop held object into a container

## Basic Usage: Physical World State Planning

### Creating a Basic Mock Environment
There are two ways to create a mock environment:

1. Using the Interactive Tool (Recommended for beginners):
```python
from predicators.spot_utils.mock_env_creation import create_mock_env_interactive
env = create_mock_env_interactive()
```
This tool will guide you through:
- Adding states with images
- Defining transitions
- Visualizing the graph
- Checking for missing transitions

2. Creating Programmatically:
```python
from predicators.envs.mock_spot_env import MockSpotEnv
from pathlib import Path

env = MockSpotEnv()

# Add states
env.add_state(
    state_id="initial",
    image_path=Path("path/to/image.png"),
    objects_in_view={"block", "table"},
    objects_in_hand=set(),
    gripper_open=True
)

# Add transitions
env.add_transition(
    from_state="initial",
    action="PickObjectFromTop",
    to_state="holding_block",
    success=True
)
```

### Understanding the Environment

#### Ground Truth Transitions
- The mock environment provides ground truth transitions through its transition graph
- Each transition is deterministic and explicitly defined
- Failed transitions are marked with success=False
- The transition graph is saved in graph.json

#### State Mapping
- Images map to symbolic states through object sets
- Example: `{"block", "table"}` in view → `On(block, table)`
- Object sets determine what predicates are true in each state

### Basic Task Definition
```python
class SpotPickPlaceTask(BaseEnv):
    def __init__(self):
        self.operators = {
            "PickObjectFromTop": {
                "preconditions": ["HandEmpty", "ObjectReachable"],
                "effects": ["Holding"]
            },
            "PlaceObjectOnTop": {
                "preconditions": ["Holding"],
                "effects": ["On", "HandEmpty"]
            }
        }
```

### Basic Integration Steps
1. Create Task:
   ```python
   task = SpotPickPlaceTask()
   ```

2. Create Mock Environment:
   ```python
   env = create_mock_env_interactive()
   ```

3. Link and Plan:
   ```python
   task.set_environment(env)
   initial_obs = env.reset()
   plan = planner.plan(task, initial_obs)
   ```

## Advanced Usage: Belief Space Planning

The environment can be extended to support belief space planning for tasks involving uncertainty (e.g., determining if a cup contains water).

### Enabling Belief Space Planning
```python
# Enable belief space operators
MockSpotEnv.use_belief_space_operators = True
env = MockSpotEnv()
```

### Additional Belief Space Operators
When belief space planning is enabled, these operators become available:
- `MoveToHandObserveObjectFromTop`: Move to observe a container from above
- `ObserveContainerContent`: Observe if a container has water

### Example: Cup Emptiness Task
```python
# Add states with belief predicates
env.add_state(
    state_id="initial",
    image_path=Path("path/to/image.png"),
    objects_in_view={"cup", "table"},
    objects_in_hand=set(),
    gripper_open=True,
    belief_predicates={"ContainingWaterUnknown(cup)"}
)

# Add belief space transitions
env.add_transition(
    from_state="initial",
    action="MoveToHandObserveObjectFromTop",
    to_state="observing_cup",
    success=True
)
```

### Belief Space Task Definition
```python
class SpotCupEmptinessTask(BaseEnv):
    def __init__(self):
        self.operators = {
            # Belief space operators
            "ObserveContainerContent": {
                "preconditions": ["ContainingWaterUnknown", "InHandViewFromTop"],
                "effects": ["ContainingWaterKnown"]
            }
        }
```

### Understanding Belief Space States

#### Ground Truth and Belief States
- The environment tracks both physical and belief states
- Physical state: actual world state (e.g., object positions)
- Belief state: knowledge state (e.g., whether cup contents are known)

#### State Mapping with Beliefs
- Images still map to physical predicates through object sets
- Belief predicates track uncertain properties
- Example: `{"cup", "table"}` in view + unknown content → `On(cup, table) ∧ ContainingWaterUnknown(cup)`

#### Transitions with Beliefs
- Physical transitions work as in basic usage
- Belief transitions can change knowledge state
- Both types can be mixed in a single plan
- All transitions are still deterministic and explicitly defined

## Tips and Best Practices

### For Basic Usage
1. Image Organization:
   - Use descriptive state IDs
   - Example: `pick_block_s1.png`, `place_block_s2.png`

2. State Design:
   - Create states for key points in manipulation
   - Include failure states for robustness

3. Transition Graph:
   - Make graph complete for all relevant actions
   - Include both success and failure transitions
   - Use visualization to verify coverage

4. Testing:
   - Test all paths to goal
   - Verify operator preconditions/effects
   - Check handling of failed actions

### Additional Tips for Belief Space Planning
1. State Design:
   - Track belief predicates explicitly
   - Consider uncertainty in state representation

2. Transition Graph:
   - Include both physical and belief transitions
   - Verify belief state changes

3. Testing:
   - Test belief space operators
   - Verify belief predicate updates
   - Check handling of uncertainty resolution 