# Mock Spot Perceiver Documentation

## Overview

The mock Spot perceiver system provides a simulated perception pipeline for testing and development without requiring physical Spot robot hardware. It handles:
- RGBD image observations
- Object detection and tracking
- VLM (Vision Language Model) predicate evaluation
- State estimation and belief updates

## Observation System

### Observation Structure

Two main observation classes are used:

1. `_SavedMockSpotObservation`:
```python
@dataclass(frozen=True)
class _SavedMockSpotObservation:
    images: Optional[Dict[str, RGBDImageWithContext]]  # Camera images
    gripper_open: bool                                 # Gripper state
    objects_in_view: Set[str]                         # Visible objects
    objects_in_hand: Set[str]                         # Held objects
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

### Observation Creation

1. **Environment Side** (`MockSpotEnv._build_observation`):
   - Retrieves saved observation data (images, objects, etc.)
   - Handles VLM predicate evaluation if enabled:
     - Creates Object instances for visible objects
     - Generates VLM atom combinations
     - Performs batch classification of VLM predicates
     - Updates VLM atom values while preserving history

2. **Saving Observations**:
   - Observations are saved in the environment's data directory
   - Images are stored separately in an images directory
   - Metadata (gripper state, objects) saved in graph.json

## Perceiver System

### Core Components

1. **State Management**:
   ```python
   class MockSpotPerceiver:
       _camera_images: Dict[str, RGBDImageWithContext]  # Current images
       _gripper_open: bool                              # Gripper state
       _objects_in_view: Set[str]                       # Visible objects
       _objects_in_hand: Set[str]                       # Held objects
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
