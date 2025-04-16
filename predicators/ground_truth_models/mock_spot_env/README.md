# Mock Spot Environment

This directory contains the ground truth models for the mock Spot environment, which is a simplified version of the real Spot environment used for testing and development.

## Components

### Environment (`mock_spot_env.py`)
- Defines types, predicates, and operators at module level for easy access
- Uses a graph-based state representation with RGB-D images
- Provides methods to add states and transitions
- Stores data persistently on disk

### NSRTs (`nsrts.py`)
- Reuses operators from the environment instead of defining new ones
- Uses dummy samplers since we don't need actual sampling
- Creates NSRTs by converting STRIPSOperators using `make_nsrt()`

### Options (`options.py`)
- Creates one option per operator
- Uses operator names in action's `extra_info` field instead of one-hot vectors
- Uses dummy parameter spaces since we don't need real parameters
- All options are always initiable and terminal

### Perceiver (`mock_spot_perceiver.py`)
- Provides mock perception capabilities
- Manages RGB-D images and object detections
- Tracks gripper state and objects in view/hand

## Tests

### Environment Tests (`test_mock_spot_env.py`)
- Tests environment initialization
- Tests state creation and transitions
- Tests graph data persistence
- Tests predicate and operator functionality

### NSRTs and Options Tests (`test_nsrts_and_options.py`)
- Tests NSRT creation from operators
- Tests option creation and properties
- Tests policy generation and execution
- Tests parameter matching between NSRTs and options

### Perceiver Tests (`test_mock_spot_perceiver.py`)
- Tests perceiver initialization
- Tests image saving and loading
- Tests state updates and observations
- Tests object detection simulation

## Usage

1. Initialize the environment:
```python
from predicators.envs.mock_spot_env import MockSpotEnv

env = MockSpotEnv(data_dir="spot_mock_data")
```

2. Add states and transitions:
```python
state_id_1 = env.add_state(rgbd=None, gripper_open=True, 
                          objects_in_view={"cup", "table"})
state_id_2 = env.add_state(rgbd=None, gripper_open=False, 
                          objects_in_view={"cup", "table"}, 
                          objects_in_hand={"cup"})
env.add_transition(state_id_1, "PickObjectFromTop", state_id_2)
```

3. Use the perceiver:
```python
from predicators.perception.mock_spot_perceiver import MockSpotPerceiver

perceiver = MockSpotPerceiver(data_dir="spot_mock_data")
perceiver.update_state(gripper_open=True, 
                      objects_in_view={"cup", "table"})
obs = perceiver.get_observation()
```

## Development Notes

1. When adding new operators:
   - Add them to `_create_operators()` in `mock_spot_env.py`
   - No need to modify NSRTs or options - they'll pick up new operators automatically

2. When adding new predicates:
   - Add them to the module-level exports in `mock_spot_env.py`
   - Update `PREDICATES` and `GOAL_PREDICATES` if needed

3. When modifying tests:
   - Use `utils.reset_config()` to set up the environment
   - Use temporary directories for test data
   - Clean up test directories in finally blocks 