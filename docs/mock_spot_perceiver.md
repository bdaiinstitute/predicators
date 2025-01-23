# Mock Spot Perceiver Documentation

## Overview

The mock Spot perceiver system provides a simulated perception pipeline for testing and development without requiring physical Spot robot hardware. It handles:
- RGBD image observations
- Object detection and tracking
- VLM (Vision Language Model) predicate evaluation
- State estimation and belief updates
- Drawer observation and content belief updates

## Object Handling

### Object System

Objects in the mock environment are managed by the environment creator and passed through as `Container[Object]`. Key points:

1. **Object Types**:
   ```python
   @dataclass
   class Object:
       name: str           # Unique identifier for the object
       type: str          # Object type (e.g. "cup", "table", "drawer")
       parent_type: Optional[str] = None  # Parent type for inheritance
   ```

2. **Object Storage**:
   - All objects are stored and passed as `Container[Object]`
   - No string-based object references
   - Direct object instance usage throughout the system
   - Objects are created and managed by the environment creator

3. **Object Flow**:
```
Environment Creator
    │
    ├── Creates and manages Object instances
    │   ├── Stores objects in Dict[str, Object]
    │   └── Maintains objects with proper types
    │
    ├── Passes Container[Object] to environment
    │   └── Covariant container for type compatibility
    │
    └── Environment and perceiver use objects directly
```

4. **Key Implementation Details**:
   - Use `Container[Object]` instead of `Set[Object]` for covariant type compatibility
   - Environment creator maintains object dictionary for lookup
   - Objects are passed by reference to maintain identity
   - Type hierarchy supports inheritance through parent_type

5. **Best Practices**:
   - Always use Container[Object] for collections to avoid type variance issues
   - Create objects through environment creator to ensure consistency
   - Maintain object identity across the system
   - Use type hierarchy for predicate compatibility

### Observation Structure

Two main observation classes are used:

1. `_SavedMockSpotObservation`:
```python
@dataclass(frozen=True)
class _SavedMockSpotObservation:
    images: Optional[Dict[str, RGBDImageWithContext]]  # Camera images
    gripper_open: bool                                 # Gripper state
    objects_in_view: Container[Object]                 # Visible objects
    objects_in_hand: Container[Object]                 # Held objects
    state_id: str                                     # Unique state ID
    atom_dict: Dict[str, bool]                        # Predicate atoms
    non_vlm_atom_dict: Optional[Set[GroundAtom]]      # Non-VLM atoms
```

2. `_MockSpotObservation` (extends `_SavedMockSpotObservation`):
```python
@dataclass(frozen=True)
class _MockSpotObservation:
    # Inherits all fields from _SavedMockSpotObservation
    vlm_atom_dict: Optional[Dict[VLMGroundAtom, bool]]  # VLM predicate results
    vlm_predicates: Optional[Set[VLMPredicate]]         # Active VLM predicates
```

### Object Creation and Management

1. **Environment Creator**:
   ```python
   class ManualMockEnvCreator(MockEnvCreatorBase):
       def __init__(self, image_dir: str, objects: Dict[str, Object]) -> None:
           super().__init__(image_dir)
           self._objects = objects  # Store object dictionary
   ```

2. **State Management**:
   ```python
   def add_state(self,
                state_id: str,
                views: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                objects_in_view: Container[Object],
                objects_in_hand: Container[Object],
                gripper_open: bool = True) -> None:
       # Save state with object references
       state_data = {
           "objects_in_view": [obj.name for obj in objects_in_view],
           "objects_in_hand": [obj.name for obj in objects_in_hand],
           ...
       }
   ```

3. **Loading Objects**:
   ```python
   def add_state_from_images(self,
                           state_id: Optional[str] = None,
                           objects_in_view: Optional[Container[Object]] = None,
                           objects_in_hand: Optional[Container[Object]] = None,
                           ...):
       # Initialize empty containers if None
       if objects_in_view is None:
           objects_in_view = set()
       if objects_in_hand is None:
           objects_in_hand = set()
   ```

## Perceiver System

### Core Components

1. **State Management**:
   ```python
   class MockSpotPerceiver:
       _camera_images: Dict[str, RGBDImageWithContext]  # Current images
       _gripper_open: bool                              # Gripper state
       _objects_in_view: Set[Object]                       # Visible objects
       _objects_in_hand: Set[Object]                       # Held objects
       _vlm_predicates: Set[VLMPredicate]               # VLM predicates
       _vlm_atom_dict: Dict[VLMGroundAtom, bool]        # VLM atom values
       _non_vlm_atoms: Set[GroundAtom]                  # Non-VLM atoms
   ```

2. **Key Methods**:
   - `get_observation()`: Returns current observation
   - `_update_state_from_observation()`: Updates internal state
   - `_obs_to_state()`: Converts observation to State object

### State vs Observation

1. **State**:
   - Represents complete world state
   - Includes all ground atoms (VLM and non-VLM)
   - Used for planning and action selection
   - Created by combining:
     - VLM predicate evaluations
     - Non-VLM predicates from environment
     - Object positions and properties

2. **Observation**:
   - Raw sensor data and immediate percepts
   - Includes:
     - RGBD images
     - Detected objects
     - Gripper state
     - VLM predicate results
   - More limited than state (partial observability)

## Belief State Updates

### VLM Predicate Handling

1. **Predicate Types**:
   ```python
   # Non-VLM predicates (environment-defined)
   _On = Predicate("On", [_movable_object_type, _base_object_type])
   
   # VLM predicates (vision-based)
   _On = VLMPredicate(
       "On", [_movable_object_type, _base_object_type],
       prompt="This predicate typically describes..."
   )
   ```

2. **Belief Update Process**:
   The belief update happens in three steps:

   a) **Check Consistency of New Labels**:
   ```python
   # Collect Known/Unknown pairs from current VLM evaluation
   # Being pessimistic: if unknown is true OR known is false, treat as unknown
   if known_val and unknown_val:  # Both True is inconsistent
       logging.warning("Inconsistent Known/Unknown values...")
   if unknown_val or not known_val:
       curr_vlm_atom_values[known_atom] = False
       curr_vlm_atom_values[unknown_atom] = True
   ```

   b) **Basic Update**:
   - Update any atom that has a non-None value from current observation
   - Preserves previous values for atoms not in current observation

   c) **Override with Previous Knowledge**:
   - If a predicate was Known=True in previous step, it stays Known=True
   - Corresponding Unknown predicate is set to False
   - Otherwise, keep values from current observation

3. **Key Principles**:
   - Known predicates cannot become unknown (monotonic knowledge)
   - Being pessimistic: treat as unknown if:
     - Unknown is true OR Known is false
     - Both Known and Unknown are true/false (inconsistent)
   - Non-VLM predicates come directly from environment
   - VLM predicates are evaluated online using images

### State Construction

1. **State Components**:
   - Complete world state representation
   - Combines VLM and non-VLM atoms
   - Includes camera images and visible objects
   - Tracks belief state for partially observable predicates

2. **State Construction Code**:
   ```python
   def _obs_to_state(self, obs: _MockSpotObservation) -> State:
       # Create state with all atoms
       state_dict = {}
       
       # Add VLM atoms if enabled
       if CFG.mock_env_vlm_eval_predicate and self._vlm_atom_dict:
           for atom, value in self._vlm_atom_dict.items():
               if value:
                   state_dict[atom] = True
       
       # Add non-VLM atoms
       if self._non_vlm_atoms:
           for atom in self._non_vlm_atoms:
               state_dict[atom] = True
       
       # Create partial perception state with additional info
       state = _PartialPerceptionState(
           state_dict,  # Base state data
           camera_images=self._camera_images if CFG.mock_env_vlm_eval_predicate else None,
           visible_objects=self._objects_in_view,
           vlm_atom_dict=self._vlm_atom_dict,
           vlm_predicates=self._vlm_predicates,
       )
       
       return state
   ```

3. **Key Components**:
   - `state_dict`: Core state data with all true atoms
   - `camera_images`: Current RGBD images (if VLM enabled)
   - `visible_objects`: Objects currently in view
   - `vlm_atom_dict`: Current VLM predicate evaluations
   - `vlm_predicates`: Active VLM predicates

4. **Partial Observability**:
   - State tracks both fully and partially observable predicates
   - Camera images maintained for VLM evaluation
   - Visible objects list for perception tracking
   - VLM atom dictionary preserves belief state

### Drawer Observation System

1. **Drawer State Predicates**:
   ```python
   # Physical state predicates
   _DrawerOpen = Predicate("DrawerOpen", [_container_type])
   _DrawerClosed = Predicate("DrawerClosed", [_container_type])
   
   # Belief state predicates
   _Unknown_ContainerEmpty = Predicate("Unknown_ContainerEmpty", [_container_type])
   _Known_ContainerEmpty = Predicate("Known_ContainerEmpty", [_container_type])
   _BelieveTrue_ContainerEmpty = Predicate("BelieveTrue_ContainerEmpty", [_container_type])
   _BelieveFalse_ContainerEmpty = Predicate("BelieveFalse_ContainerEmpty", [_container_type])
   ```

2. **Belief Update Process**:
   The drawer observation system follows these principles:

   a) **Initial State**:
   - Drawer content starts as unknown (`Unknown_ContainerEmpty`)
   - Physical state (open/closed) is known
   - No beliefs about contents

   b) **Observation Process**:
   ```python
   # Drawer must be open for observation
   if not _DrawerOpen(drawer):
       return  # Cannot observe closed drawer
   
   # Update knowledge state
   del_effs.add(_Unknown_ContainerEmpty(drawer))
   add_effs.add(_Known_ContainerEmpty(drawer))
   
   # Update belief based on observation
   if found_empty:
       add_effs.add(_BelieveTrue_ContainerEmpty(drawer))
   else:
       add_effs.add(_BelieveFalse_ContainerEmpty(drawer))
   ```

   c) **Knowledge Persistence**:
   - Once drawer content is known, it stays known
   - Beliefs about contents can be updated
   - Physical state can change independently

3. **Key Principles**:
   - Drawer must be open for observation
   - Knowledge is monotonic (cannot become unknown)
   - Beliefs can change with new observations
   - Physical and belief states are tracked separately

### State Construction

1. **State Components**:
   - Complete world state representation
   - Combines physical and belief predicates
   - Tracks drawer state and contents
   - Maintains observation history

2. **State Construction Code**:
   ```python
   def _obs_to_state(self, obs: _MockSpotObservation) -> State:
       # Create state with all atoms
       state_dict = {}
       
       # Add physical state atoms
       if self._non_vlm_atoms:
           for atom in self._non_vlm_atoms:
               state_dict[atom] = True
       
       # Add belief state atoms
       if self._belief_atoms:
           for atom in self._belief_atoms:
               state_dict[atom] = True
       
       # Create state with additional info
       state = _PartialPerceptionState(
           state_dict,
           camera_images=self._camera_images,
           visible_objects=self._objects_in_view,
           belief_atoms=self._belief_atoms
       )
       
       return state
   ```

3. **Key Components**:
   - `state_dict`: Core state data with all true atoms
   - `_belief_atoms`: Set of current belief predicates
   - `_non_vlm_atoms`: Physical state predicates
   - `_camera_images`: Current visual observations

## Testing

### Test Cases

1. **Basic Drawer Observation**:
   ```python
   def test_drawer_observation():
       # Initial state - drawer content unknown
       initial_atoms = {
           _HandEmpty(robot),
           _DrawerOpen(drawer),
           _Unknown_ContainerEmpty(drawer)
       }
       
       # After observation finding empty
       observation_atoms = {
           _Known_ContainerEmpty(drawer),
           _BelieveTrue_ContainerEmpty(drawer)
       }
       
       # After observation finding objects
       observation_atoms = {
           _Known_ContainerEmpty(drawer),
           _BelieveFalse_ContainerEmpty(drawer),
           _Inside(apple, drawer)
       }
   ```

2. **State Transitions**:
   ```python
   # Valid transitions:
   Unknown -> Known + BelieveTrue   # Found empty
   Unknown -> Known + BelieveFalse  # Found objects
   
   # Invalid transitions:
   Known -> Unknown  # Cannot lose knowledge
   ```

3. **Best Practices**:
   - Always check drawer is open before observation
   - Update both knowledge and belief state
   - Maintain monotonicity of knowledge
   - Track physical state separately from beliefs

## TODOs

### High Priority
1. **Object Management**:
   - [ ] Update environment creator to properly save object type information
   - [ ] Add object versioning/validation when loading from disk
   - [ ] Handle object type inheritance in predicate evaluation

2. **VLM Integration**:
   - [ ] Improve VLM predicate evaluation with object type checking
   - [ ] Add caching for VLM results to avoid redundant evaluation
   - [ ] Handle VLM service failures gracefully

3. **State Management**:
   - [ ] Add proper state validation when loading from disk
   - [ ] Implement state diffing for efficient updates
   - [ ] Add state versioning for backward compatibility

### Medium Priority
1. **Testing**:
   - [ ] Add comprehensive tests for object type handling
   - [ ] Add tests for VLM predicate evaluation
   - [ ] Add tests for state serialization/deserialization

2. **Documentation**:
   - [ ] Add examples for common object manipulation scenarios
   - [ ] Document best practices for object type design
   - [ ] Add troubleshooting guide for common issues

3. **Performance**:
   - [ ] Optimize VLM batch evaluation
   - [ ] Add object caching for frequently used objects
   - [ ] Optimize state updates for large object sets

### Low Priority
1. **Features**:
   - [ ] Add support for dynamic object type creation
   - [ ] Add object property validation
   - [ ] Add object relationship tracking

2. **Tooling**:
   - [ ] Add visualization tools for object relationships
   - [ ] Add debugging tools for object state
   - [ ] Add profiling tools for VLM evaluation

## Best Practices

1. **VLM Predicate Design**:
   - Use clear, specific prompts
   - Consider object type hierarchies
   - Handle ambiguous cases

2. **State Management**:
   - Preserve known predicate values
   - Handle partial observability
   - Maintain belief consistency

3. **Error Handling**:
   - Validate object types
   - Check image availability
   - Handle VLM classification failures
