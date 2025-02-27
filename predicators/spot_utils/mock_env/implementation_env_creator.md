# Mock Environment Creator Implementation Note

## Overview
This document outlines the implementation of the mock environment creator with support for:
- State and transition management
- Image and observation storage
- Transition system generation
- Belief space planning support
- Interactive data collection for Spot robot

## 1. Data Structure

### Directory Structure
```
mock_env_data/
├── images/
│   ├── state_0/
│   │   ├── state_metadata.json  # State data and metadata
│   │   ├── cam1_rgb.npy        # RGB image data
│   │   ├── cam1_rgb.jpg        # RGB preview image
│   │   └── cam1_depth.npy      # Optional depth data
│   └── state_1/
│       └── ...
├── transition_system.json       # Full transition system
└── transitions/                 # Visualization files
    └── {task_name}.html        # Interactive visualization
```

### Transition System Format
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

### State Metadata Format
```json
{
    "gripper_open": true,
    "objects_in_view": ["obj1", "obj2"],
    "objects_in_hand": ["obj3"],
    "atom_dict": {
        "atom_str1": true,
        "atom_str2": false
    },
    "non_vlm_atom_dict": {
        "atom_str3": true
    },
    "metadata": {
        "custom_field": "value"
    }
}
```

## 2. Core Components

### State Management
```python
class MockEnvCreatorBase:
    """Base class for loading/saving environment data."""
    
    def __init__(self, output_dir: str, env_info: Dict[str, Any]) -> None:
        # Initialize state storage
        self.states: Dict[str, _SavedMockSpotObservation] = {}  # For observation data
        self.state_to_id: Dict[FrozenSet[GroundAtom], str] = {}  # Maps state atoms to IDs
        self.id_to_state: Dict[str, Set[GroundAtom]] = {}  # Maps IDs to state atoms
        self.transitions: List[Tuple[str, _GroundNSRT, str]] = []  # (source_id, op, dest_id)
        self.str_transitions: List[Tuple[str, Dict[str, Any], str]] = []  # For saving to JSON
```

### State Exploration
```python
def explore_states(self, init_atoms: Set[GroundAtom], objects: Set[Object]) -> None:
    """Explore all reachable states and transitions."""
    # Start with initial state
    init_state_frozen = frozenset(init_atoms)
    init_id = "0"
    self.state_to_id[init_state_frozen] = init_id
    self.id_to_state[init_id] = init_atoms
    
    # BFS through state space
    frontier = [(init_atoms, None)]
    visited = {init_state_frozen}
    state_count = 1
    
    while frontier:
        curr_atoms, _ = frontier.pop(0)
        curr_id = self.state_to_id[frozenset(curr_atoms)]
        
        # Get applicable operators
        applicable_ops = self._get_applicable_operators(curr_atoms, objects)
        
        # Explore transitions
        for op in applicable_ops:
            next_atoms = self._get_next_atoms(curr_atoms, op)
            next_frozen = frozenset(next_atoms)
            
            # Add new state if not seen
            if next_frozen not in visited:
                visited.add(next_frozen)
                next_id = str(state_count)
                state_count += 1
                self.state_to_id[next_frozen] = next_id
                self.id_to_state[next_id] = next_atoms
                frontier.append((next_atoms, next_id))
            
            # Add transition
            next_id = self.state_to_id[next_frozen]
            self.add_transition(curr_id, op, next_id)
```

### Transition System Saving
```python
def save_transitions(self) -> None:
    """Save transition system to JSON."""
    transition_system = {
        "objects": {
            obj.name: {
                "type": obj.type.name,
                "parent_type": obj.type.parent.name if obj.type.parent else None
            }
            for obj in self.objects.values()
        },
        "states": {
            state_id: {
                "atoms": [str(atom) for atom in atoms],
                "fluent_atoms": [str(atom) for atom in atoms 
                               if atom.predicate.name in self.fluent_predicates]
            }
            for state_id, atoms in self.id_to_state.items()
        },
        "transitions": self.str_transitions,
        "metadata": {
            "fluent_predicates": list(self.fluent_predicates),
            "predicate_types": {
                name: [t.name for t in pred.types] 
                for name, pred in self.predicates.items()
            }
        }
    }
    
    with open(self.output_dir / "transition_system.json", "w") as f:
        json.dump(transition_system, f, indent=2)
```

## 3. Current Status

### Completed
1. Core Functionality:
   - [x] State exploration and storage
   - [x] Transition system generation
   - [x] Image and metadata saving
   - [x] Object type hierarchy support

2. Visualization:
   - [x] Interactive transition graph
   - [x] State content display
   - [x] Operator effect visualization

3. Planning Support:
   - [x] Fluent predicate tracking
   - [x] Belief state handling
   - [x] Plan generation and saving

### In Progress
1. Testing:
   - [ ] Core functionality tests
   - [ ] State exploration tests
   - [ ] Image handling tests

2. Documentation:
   - [ ] API documentation
   - [ ] Usage examples
   - [ ] Testing guide

### Future Work
1. Improvements:
   - [ ] Better error handling
   - [ ] More efficient state exploration
   - [ ] Enhanced visualization features

2. Extensions:
   - [ ] Support for multiple robots
   - [ ] Custom camera calibration
   - [ ] Advanced visualization tools

## Notes

1. Design Principles:
   - Clear separation of concerns
   - Strong typing and validation
   - Efficient state exploration

2. Performance Considerations:
   - Lazy loading of images
   - Efficient state storage
   - Optimized transition lookup

3. Error Handling:
   - Validate state consistency
   - Check metadata integrity
   - Handle missing files gracefully

## 4. Spot-Specific Extensions

### Interactive Data Collection
1. Command-Line Interface:
```python
class SpotDataCollector:
    def collect_state(self, state_id: str, prompt: str = None) -> bool:
        """Collect data for a state interactively.
        
        Args:
            state_id: ID of state to collect
            prompt: Optional custom prompt
            
        Returns:
            True if data collected, False if skipped/ended
        """
        # Show state info
        print(f"\nCollecting data for State {state_id}")
        if prompt:
            print(prompt)
            
        # Get user input
        choice = input("[t]ake photo / [s]kip / [e]nd: ").lower()
        if choice == 'e':
            return False
        elif choice == 's':
            return True
            
        # Collect RGB-D data
        self._collect_rgbd()
        return True
```

2. State Visualization:
```python
def visualize_current_state(self, state: Set[GroundAtom]) -> None:
    """Show current state information."""
    # Group atoms by predicate
    atoms_by_pred = defaultdict(list)
    for atom in sorted(state, key=str):
        atoms_by_pred[atom.predicate.name].append(atom)
        
    # Create formatted display
    console = Console()
    console.print("\n[bold]Current State:[/bold]")
    
    # Show predicates in groups
    for pred_name, atoms in atoms_by_pred.items():
        console.print(f"\n[cyan]{pred_name}:[/cyan]")
        for atom in atoms:
            console.print(f"  {atom}")
```

### Spot Robot Integration
1. Robot Connection:
```python
class SpotMockEnvCreator(MockEnvCreatorBase):
    def __init__(self, hostname: str, output_dir: str):
        super().__init__(output_dir)
        self.robot = self._connect_robot(hostname)
        self.image_client = self.robot.ensure_client(ImageClient)
        
    def _connect_robot(self, hostname: str) -> Robot:
        """Connect to Spot robot."""
        sdk = bosdyn.client.create_standard_sdk('MockEnvCreator')
        robot = sdk.create_robot(hostname)
        robot.authenticate(username, password)
        return robot
```

2. Image Capture:
```python
def capture_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
    """Capture synchronized RGB-D images."""
    # Get images from robot
    images = self.image_client.get_image_from_sources([
        "hand_depth_in_hand_color_frame",
        "hand_color_image"
    ])
    
    # Process depth
    depth = process_depth_image(
        images[0].shot.image.data,
        (images[0].shot.image.rows, images[0].shot.image.cols)
    )
    
    # Process RGB
    rgb = cv2.imdecode(
        np.frombuffer(images[1].shot.image.data, dtype=np.uint8),
        -1
    )
    
    return rgb, depth
```

### Data Collection Workflow
1. Plan-Based Collection:
```python
def collect_plan_data(self, plan: List[Action]) -> None:
    """Collect data following a plan."""
    collector = SpotDataCollector(self.robot)
    
    for i, action in enumerate(plan):
        # Show action info
        print(f"\nStep {i+1}/{len(plan)}")
        print(f"Action: {action}")
        
        # Collect state data
        if not collector.collect_state(
            str(i),
            prompt=f"Executing: {action}"
        ):
            break
```

2. Manual Collection:
```python
def collect_manual_data(self) -> None:
    """Collect data manually."""
    collector = SpotDataCollector(self.robot)
    state_count = 0
    
    while True:
        # Show options
        print("\nOptions:")
        print("1. Add new state")
        print("2. Review collected states")
        print("3. End collection")
        
        choice = input("Choice: ")
        if choice == "3":
            break
            
        if choice == "1":
            if not collector.collect_state(str(state_count)):
                break
            state_count += 1
```

### Data Organization
1. Directory Structure:
```
mock_env_data/
├── images/
│   ├── state_0/
│   │   ├── rgb.npy          # RGB image data
│   │   ├── depth.npy        # Depth image data
│   │   ├── metadata.json    # Camera/robot state
│   │   ├── preview_rgb.jpg  # Preview images
│   │   └── preview_rgbd.jpg
│   └── state_1/
│       └── ...
├── transitions.json         # State transition graph
└── plan.yaml               # Optional plan data
```

2. Metadata Format:
```json
{
    "state_id": "0",
    "timestamp": "2024-03-20T10:30:00",
    "camera": {
        "transform": [...],
        "intrinsics": [...],
        "name": "hand_color"
    },
    "robot": {
        "position": [...],
        "orientation": [...],
        "gripper_open": true
    },
    "objects": {
        "in_view": ["cup1", "table"],
        "in_hand": []
    }
}
```

### Visualization Tools
1. Live Preview:
```python
def show_preview(self, rgb: np.ndarray, depth: np.ndarray) -> None:
    """Show live RGB-D preview."""
    # Create RGB-D overlay
    overlay = create_rgbd_overlay(rgb, depth)
    
    # Show images
    cv2.imshow("RGB", rgb[...,::-1])  # BGR for OpenCV
    cv2.imshow("Depth", colorize_depth(depth))
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(1)
```

2. Collection Progress:
```python
def show_progress(self) -> None:
    """Show collection progress."""
    console = Console()
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("State")
    table.add_column("Images")
    table.add_column("Objects")
    table.add_column("Status")
    
    # Add rows
    for state_id, data in self.states.items():
        table.add_row(
            state_id,
            "✓" if data.images else "✗",
            ", ".join(obj.name for obj in data.objects_in_view),
            "[green]Complete[/green]" if data.is_complete() else "[yellow]Partial[/yellow]"
        )
    
    console.print(table)
``` 