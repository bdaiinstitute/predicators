# Mock Spot Perceiver Documentation

## Overview

The mock Spot perceiver system provides a simulated perception pipeline for testing and development without requiring physical Spot robot hardware. It handles:
- RGBD image observations
- Object detection and tracking
- VLM (Vision Language Model) predicate evaluation
- State estimation and belief updates

****## Object Handling

### Object System

Objects in the mock environment are managed by the environment creator and passed through as `Set[Object]`. Key points:

1. **Object Types**:
   ```python
   @dataclass
   class Object:
       name: str           # Unique identifier for the object
       type: str          # Object type (e.g. "cup", "table")
       parent_type: Optional[str] = None  # Parent type for inheritance
   ```

2. **Object Storage**:
   - All objects are stored and passed as `Set[Object]` or `Container[Object]`
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
   - VLM predicates are evaluated online using images
   - Known predicates cannot become unknown
   - Updates preserve previous knowledge
   - Non-VLM predicates come directly from environment

### Key Principles

1. **Predicate Selection**:
   - When VLM enabled:
     - Use VLM predicates for perception-based predicates
     - Filter out non-VLM counterparts
     - Keep non-VLM predicates without VLM versions
   - When VLM disabled:
     - Use all non-VLM predicates

2. **State Construction**:
   ```python
   def _obs_to_state(self, obs: _MockSpotObservation) -> State:
       state = State({})  # Empty state
       
       # Add VLM atoms if enabled
       if CFG.spot_vlm_eval_predicate and self._vlm_atom_dict:
           for atom, value in self._vlm_atom_dict.items():
               if value:  # Only add True atoms
                   state = state.copy()
                   state.set_atoms({atom})
       
       # Add non-VLM atoms from environment
       if self._non_vlm_atoms:
           state = state.copy()
           state.set_atoms(self._non_vlm_atoms)
           
       return state
   ```

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
