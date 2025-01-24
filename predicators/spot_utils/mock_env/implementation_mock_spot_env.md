# Mock Spot Environment Implementation Note

## Overview
This document outlines the implementation of the mock spot environment that:
- Separates environment loading from perception
- Uses environment creator for state management
- Handles observation generation and belief updates
- Manages state loading and saving
- Integrates VLM-based perception

## 1. Core Components

### Environment Structure
```python
class MockSpotEnv(BaseEnv):
    """Mock environment for Spot robot."""
    
    def __init__(self, use_gui: bool = True) -> None:
        """Initialize environment."""
        self._data_dir = Path(CFG.mock_env_data_dir)
        self._images_dir = self._data_dir / "images"
        self._current_state_id: Optional[str] = None
        self._current_observation: Optional[_MockSpotObservation] = None
        
        # Create constant objects
        self._spot_object = Object("robot", _robot_type)
        self._objects: Dict[str, Object] = {"robot": self._spot_object}
        
        # Load transitions and observations
        self._str_transitions: List[Tuple[str, Dict[str, Any], str]] = []  # (source_id, op_dict, dest_id)
        self._observations: Dict[str, _SavedMockSpotObservation] = {}
        self._transition_metadata: Dict[str, Any] = {}
        
        # Load transitions and objects
        self._load_transitions()

        # Create operators
        self._operators = list(self._create_operators())
```

### Transition System
The environment now uses a simplified transition system stored in `transition_system.json`:
```json
{
    "objects": {
        "obj_name": {
            "type": "type_name",
            "parent_type": "parent_type_name"
        }
    },
    "states": {
        "state_id": {
            "atoms": ["atom_str1", "atom_str2"],
            "fluent_atoms": ["fluent_atom1"]
        }
    },
    "transitions": [
        ["source_id", {"name": "op_name", "objects": ["obj1", "obj2"]}, "dest_id"]
    ],
    "metadata": {
        "fluent_predicates": ["pred1", "pred2"],
        "predicate_types": {
            "pred_name": ["type1", "type2"]
        }
    }
}
```

### State Loading Process
1. Environment Initialization:
   - Load transition system from JSON
   - Create objects from type information
   - Initialize transition graph
   - Create operators

2. State Loading:
   - Load state observation from disk
   - Convert objects for state loading
   - Initialize VLM predicates if needed

3. Action Handling:
   - Extract operator info from action
   - Find matching transition
   - Load next state observation
   - Update beliefs for observation operators

## 2. Key Features

### Predicate System
1. Ground Truth Predicates:
   - Basic manipulation predicates (On, Inside, etc.)
   - Robot state predicates (HandEmpty, Holding, etc.)
   - Object property predicates (IsPlaceable, HasFlatTopSurface, etc.)
   - Belief state predicates (ContainingWaterKnown, Known_ContainerEmpty, etc.)

2. VLM Predicates:
   - Vision-language predicates for perception
   - Integration with VLM-based classification
   - Belief update through observation operators

### Operator System
1. Base Operators:
   - MoveToReachObject: Move robot to reach a movable object
   - MoveToHandViewObject: Move robot's hand to view an object
   - PickObjectFromTop: Pick up an object from a surface
   - PlaceObjectOnTop: Place a held object on a surface
   - DropObjectInside: Drop a held object inside a container

2. Observation Operators:
   - ObserveCupContent: Check if a cup contains water
   - ObserveDrawerContentFindEmpty: Look in drawer and find it empty
   - ObserveDrawerContentFindNotEmpty: Look in drawer and find objects

## 3. Current Status

### Completed
1. Core Environment:
   - [x] Basic environment structure
   - [x] State loading and saving
   - [x] Action handling
   - [x] Observation generation
   - [x] Transition system JSON format
   - [x] Object type hierarchy loading

2. Predicate System:
   - [x] Ground truth predicates
   - [x] VLM predicates
   - [x] Belief predicates

3. Operator System:
   - [x] Base manipulation operators
   - [x] Observation operators
   - [x] State transitions

### In Progress
1. Type System:
   - [ ] Fix type covariance issues
   - [ ] Improve type safety
   - [ ] Handle VLM predicate inheritance

2. Testing:
   - [ ] Add comprehensive tests
   - [ ] Test belief updates
   - [ ] Test VLM integration

### Future Work
1. Improvements:
   - [ ] Better error handling
   - [ ] More flexible state loading
   - [ ] Enhanced VLM integration

2. Documentation:
   - [ ] API documentation
   - [ ] Usage examples
   - [ ] Testing guide

## Notes

1. Design Principles:
   - Clear separation between environment and perception
   - Strong typing and error handling
   - Flexible belief update system

2. VLM Integration:
   - Semantic understanding of scenes
   - Belief updates through observation
   - Configurable prompts

3. Testing Strategy:
   - Unit tests for components
   - Integration tests for workflows
   - VLM evaluation tests

## 1. Image Handling

### Image Types
We will use two main image types:
```python
# Simple RGB images with basic context
# Provide depth if available
UnposedImageWithContext:
    rgb: NDArray[np.uint8]  # RGB image array
    depth: Optional[NDArray[np.float32]]  # Optional depth array
    camera_name: str  # Name of camera that captured image
    image_rot: Optional[float] = None  # Optional rotation
```

### Image Storage
- RGB images stored as both `.npy` (for processing) and `.jpg` (for preview)
- Optional depth images stored as `.npy`
- Flat naming structure: `state_0/cam1.seed0.rgb.npy`
- Metadata (transforms, etc.) stored in JSON

### State Storage
```python
@dataclass(frozen=False)
class _SavedMockSpotObservation:
    """Core state storage class."""
    images: Optional[Dict[str, UnposedImageWithContext]]
    gripper_open: bool
    objects_in_view: Set[Object]
    objects_in_hand: Set[Object]
    state_id: str
    atom_dict: Dict[str, bool]
    non_vlm_atom_dict: Optional[Set[GroundAtom]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_state(self, save_dir: Optional[Path] = None) -> None:
        """Save state data and metadata."""
        
    @classmethod
    def load_state(cls, state_id: str, save_dir: Path, objects: Dict[str, Object]) -> "_SavedMockSpotObservation":
        """Load state data and metadata."""
```

## 2. Environment Creator Integration

### Base Creator Class
```python
class MockEnvCreatorBase:
    """Base class for loading/saving environment data."""
    
    def __init__(self, output_dir: str, env_info: Dict[str, Any]) -> None:
        """Initialize the mock environment creator."""
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.transitions_dir = self.output_dir / "transitions"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.transitions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state storage
        self.states: Dict[str, _SavedMockSpotObservation] = {}
        self.transitions: List[Tuple[str, str, Any]] = []
        
        # Store environment info
        self.types = {t.name: t for t in env_info["types"]}
        self.predicates = {p.name: p for p in env_info["predicates"]}
        self.options = {o.name: o for o in env_info["options"]}
        self.nsrts = env_info["nsrts"]
        
        # Initialize objects dictionary
        self.objects: Dict[str, Object] = {}

    def add_state_from_raw_images(
        self,
        raw_images: Dict[str, Tuple[str, str]],
        state_id: Optional[str] = None,
        objects_in_view: Optional[Set[Object]] = None,
        objects_in_hand: Optional[Set[Object]] = None,
        gripper_open: bool = True
    ) -> None:
        """Add a state with raw images to the environment."""
        pass
```

## 3. Environment Loading

### Loading Process
1. Environment initialization:
   ```python
   def __init__(self):
       # Initialize environment
   ```
2. State loading:
   ```python
   def load_state(self, state_id: str) -> _SavedMockSpotObservation:
       """Load state data from disk."""
       return _SavedMockSpotObservation.load_state(state_id, self.output_dir, self.objects)
   ```

## 4. Visualization

### Interactive Graph
- Uses Cytoscape.js for interactive visualization
- Shows state transitions with operator details
- Color-coded states based on predicates
- Highlights shortest path to goal

### Graph Features
1. State Display:
   - State ID and number
   - Current predicates
   - Objects in view/hand
   - Self-loop operators

2. Transition Display:
   - Operator name and parameters
   - Add/delete effects
   - Color-coded for belief changes
   - Path highlighting

## 5. TODOs

### High Priority
1. Image Handling
   - [x] Update `UnposedImageWithContext` to use rgb/depth naming
   - [x] Move image saving/loading to `_SavedMockSpotObservation`
   - [x] Add proper image type conversion utilities
   - [x] Implement flat image naming structure

2. Environment Creator
   - [x] Move loading logic to base creator class
   - [x] Implement clean save/load interface
   - [x] Add proper error handling
   - [x] Add object tracking

3. Environment Loading
   - [x] Update environment to use creator for loading
   - [x] Fix object handling in loading process
   - [x] Add proper state initialization
   - [x] Handle type safety in image loading

### Medium Priority
1. VLM Integration
   - [x] Clean up VLM predicate handling
   - [x] Move VLM functions to appropriate location
   - [x] Add proper type hints for VLM functions

2. Testing
   - [x] Add tests for image loading
   - [x] Add tests for state loading
   - [x] Add tests for VLM integration
   - [x] Add tests for object tracking

### Low Priority
1. Documentation
   - [x] Add docstrings for new methods
   - [x] Update class documentation
   - [x] Add examples
   - [ ] Add architecture diagrams

2. Cleanup
   - [x] Remove unused code
   - [x] Consolidate duplicate functionality
   - [x] Improve error messages
   - [ ] Add more type hints

## Notes

1. Image Handling:
   - Flat naming structure for clarity
   - Preview images for easy visualization
   - Type-safe image loading
   - Proper error handling

2. State Loading:
   - Automatic object tracking
   - Type-safe state loading
   - Proper error handling
   - Clear metadata structure

3. VLM Integration:
   - Clean separation of VLM logic
   - Type-safe predicate handling
   - Proper error handling
   - Optional VLM functionality

4. Directory Structure:
```
output_dir/
  ├── images/
  │   ├── state_0/
  │   │   ├── state_metadata.json   # Robot state, objects, atoms
  │   │   ├── image_metadata.json   # Image paths and transforms
  │   │   ├── cam1.seed0.rgb.npy   # RGB image data
  │   │   ├── cam1.seed0.rgb.jpg   # RGB preview
  │   │   └── cam1.seed0.depth.npy # Optional depth data
  │   └── state_1/
  │       └── ...
  └── transitions/
      └── {task_name}.html  # Interactive visualization
```

## Metadata Files

### state_metadata.json
```json
{
    "gripper_open": true,
    "objects_in_view": ["cup1", "table"],
    "objects_in_hand": [],
    "atom_dict": {
        "HandEmpty": true,
        "On_cup1_table": true
    }
}
```

### image_metadata.json
```json
{
    "cam1.seed0": {
        "rgb_path": "cam1.seed0.rgb.npy",
        "depth_path": "cam1.seed0.depth.npy",
        "camera_name": "hand_camera",
        "image_rot": 0.0
    }
}
```

## Loading Process
1. Load state metadata
2. Load image metadata
3. Load image data using paths from metadata
4. Create UnposedImageWithContext objects
5. Create SavedMockSpotObservation with all data
6. Track objects for type safety 