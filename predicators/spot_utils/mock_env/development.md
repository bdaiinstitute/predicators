# Development Guide for Mock Spot Environment

This guide covers best practices and guidelines for developing with the mock Spot environment.

## Development Workflow

### Adding New Operators

1. Add the operator to `_create_operators()` in `mock_spot_env.py`:
   ```python
   def _create_operators(self) -> Iterator[STRIPSOperator]:
       # ... existing operators ...
       
       # Add new operator
       robot = Variable("?robot", _robot_type)
       obj = Variable("?object", _movable_object_type)
       parameters = [robot, obj]
       preconds = {...}
       add_effs = {...}
       del_effs = {...}
       yield STRIPSOperator("NewOperator", parameters, preconds,
                           add_effs, del_effs)
   ```

2. No need to modify NSRTs or options - they'll pick up new operators automatically
3. Update tests to cover the new operator
4. Document the operator's purpose and requirements

### Adding New Predicates

1. Add predicates to module-level exports in `mock_spot_env.py`:
   ```python
   _NewPredicate = Predicate("NewPredicate", 
                            [_robot_type, _movable_object_type],
                            _dummy_classifier)
   ```

2. Update `PREDICATES` and `GOAL_PREDICATES` if needed
3. Add tests for the new predicate
4. Document predicate semantics

### Modifying State Representation

1. Update `MockSpotObservation` dataclass if needed
2. Modify `add_state()` method to handle new state components
3. Update state serialization in `_save_graph_data()`
4. Update tests to verify state changes

## Best Practices

### Code Organization

1. Keep related functionality together:
   - Environment logic in `mock_spot_env.py`
   - Creation tools in `mock_creator_*.py`
   - Task definitions in `mock_spot_task.py`

2. Use clear module-level organization:
   ```python
   # Types
   _robot_type = Type(...)
   
   # Predicates
   _On = Predicate(...)
   
   # Classes
   class MockSpotEnv(BaseEnv):
       ...
   ```

3. Follow consistent naming conventions:
   - Operator names: CamelCase (e.g., `PickObjectFromTop`)
   - Predicate names: CamelCase (e.g., `ObjectInView`)
   - Variable names: snake_case
   - Private methods: _leading_underscore

### Image Organization

1. Use descriptive state IDs:
   - Task-specific: `pick_cup_s1`, `place_cup_s2`
   - Action-specific: `pre_pick`, `post_pick`
   - Goal-related: `goal_state`, `subgoal_1`

2. Follow consistent naming convention:
   ```
   spot_mock_data/
   ├── images/
   │   ├── rgb_<state_id>.npy
   │   └── depth_<state_id>.npy
   ```

3. Document image metadata:
   - Camera parameters
   - Object annotations
   - Capture conditions

### State Design

1. Create states for key manipulation points:
   - Pre-grasp
   - Grasp
   - Post-grasp
   - Pre-place
   - Place
   - Post-place

2. Include failure states:
   - Grasp failures
   - Object occlusions
   - Collision states

3. Consider partial observability:
   - Multiple viewpoints
   - Occluded objects
   - Uncertain object states

### Transition Graph

1. Make graph complete:
   - Connect all related states
   - Include recovery transitions
   - Document unreachable states

2. Include both success and failure:
   - Success transitions with expected outcomes
   - Failure transitions with error states
   - Recovery transitions to safe states

3. Verify graph properties:
   - Connectivity
   - Reachability
   - Cycle detection

## Error Handling

1. Use descriptive error messages:
   ```python
   if state_id not in self._observations:
       raise ValueError(f"Unknown state ID: {state_id}")
   ```

2. Log important events:
   ```python
   logging.debug("Added state %s with data: %s", state_id, data)
   ```

3. Validate inputs:
   ```python
   def add_transition(self, from_state: str, action: str, to_state: str) -> None:
       if not any(op.name == action for op in self._operators):
           raise ValueError(f"Unknown operator: {action}")
   ```

## Documentation

1. Use descriptive docstrings:
   ```python
   def add_state(self, rgbd: Optional[RGBDImageWithContext] = None,
                gripper_open: bool = True) -> str:
       """Add a new state to the environment.
       
       Args:
           rgbd: RGB-D image with context, if available
           gripper_open: Whether gripper is open
           
       Returns:
           state_id: Unique ID for the new state
           
       Raises:
           ValueError: If state data is invalid
       """
   ```

2. Keep README up to date:
   - Update when adding features
   - Document breaking changes
   - Include examples

3. Add inline comments for complex logic:
   ```python
   # Convert observation to serializable format
   # We need to handle special cases for RGB-D data
   observations_data = {}
   for state_id, obs in self._observations.items():
       observations_data[state_id] = {
           "gripper_open": obs.gripper_open,
           # ... more fields ...
       }
   ```
``` 