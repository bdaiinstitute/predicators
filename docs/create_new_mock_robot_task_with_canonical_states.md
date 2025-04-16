# Creating Mock Robot Tasks with Images and Canonical State Mappings

This guide explains how to create mock robot tasks with unique image states and canonical state mappings.
Canonical states are states that are equivalent under key predicates.

## Overview

The mock environment system now supports:
1. Tracking unique (canonical) states based on view and world state
2. Mapping ground atoms to states and canonical states
3. Managing key atoms for important state properties
4. Using manually collected images (e.g. from phone) for state observations

## State Management

### Unique States

States are considered unique based on two criteria:
1. View state: Objects in view and in hand
2. World state: Ground atoms that are true

Multiple images can map to the same logical state. For example:
- Different camera angles of the same scene
- Different lighting conditions
- Slight variations in object positions

### Ground Atom Mappings

The system tracks:
1. Which states have each ground atom
2. Which canonical states have each ground atom
3. Key atoms that are important for the task

Example:
```python
# Create environment and creator
env = MockSpotDrawerCleaningEnv()
creator = MockEnvCreatorBase("mock_env_data/my_task", env=env)

# Get states where drawer is closed
drawer_closed_atom = GroundAtom(_DrawerClosed, [env.drawer])
drawer_states = creator.get_states_with_atom(drawer_closed_atom)
print(f"States with closed drawer: {drawer_states}")

# Get canonical states where cup is in container
inside_cup_atom = GroundAtom(_Inside, [env.cup, env.container])
canonical_states = creator.get_canonical_states_with_atom(inside_cup_atom)
print(f"Canonical states with cup in container: {canonical_states}")

# Add new key atom to track
creator.add_key_atom(inside_cup_atom)
```

## Using Phone Images

### Method 1: CLI Tool

1. Take photos of each state with your phone
2. Transfer HEIC images to your computer
3. Run the CLI tool:

```bash
python -m predicators.spot_utils.mock_env.mock_env_creator_base \
    --output_dir mock_env_data/my_task \
    --image_dir path/to/phone/images \
    --env_name my_new_task_env
```

The tool will:
- Show available states and their properties
- Let you map each image to a state
- Track unique states and their mappings
- Save mappings for future use

### Method 2: Python API

1. Organize your images:
```
phone_images/
├── drawer_closed.HEIC
├── drawer_open.HEIC
└── ...
```

2. Create mapping script:
```python
from pathlib import Path
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase

# Create environment and creator
env = MyNewTaskEnv()
creator = MockEnvCreatorBase("mock_env_data/my_task", env=env)

# Define image mappings
image_dir = Path("phone_images")
state_images = {
    "0": {"cam1.seed0.rgb": (str(image_dir / "drawer_closed.HEIC"), "rgb")},
    "1": {"cam1.seed0.rgb": (str(image_dir / "drawer_open.HEIC"), "rgb")},
    ...
}

# Add images for each state
for state_id, images in state_images.items():
    creator.add_state_from_raw_images(
        raw_images=images,
        state_id=state_id,
        objects_in_view=...,  # Define based on state
        objects_in_hand=...,  # Define based on state
        gripper_open=...      # Define based on state
    )

# Save state mapping
creator.save_state_mapping()
```

## State Mapping File

The system saves state mappings in `state_mapping.yaml`:

```yaml
state_to_canonical:
  "0": "0"    # This state is canonical
  "1": "0"    # This state maps to state 0
canonical_state_to_id:
  "0": ["0", "1"]  # State 0 has two equivalent states
unique_view_states:
  "view:cup,table|hand:": ["0", "1"]  # States with same view
unique_world_states:
  "world:HandEmpty,On_cup_table": ["0", "1"]  # States with same atoms
atom_to_states:
  "DrawerClosed(drawer)": ["0", "6"]  # States where drawer is closed
atom_to_canonical_states:
  "Inside(cup,container)": ["3", "5"]  # Canonical states with cup in container
key_atoms:
  - "DrawerClosed(drawer)"
  - "Inside(cup,container)"
```

## Best Practices

1. State Management:
   - Use meaningful state IDs
   - Document state meanings
   - Track important atoms as key atoms

2. Image Collection:
   - Take consistent photos
   - Use good lighting
   - Capture key state features

3. Testing:
   - Verify state uniqueness
   - Check atom mappings
   - Test with equivalent states

4. Documentation:
   - Document state meanings
   - List key atoms
   - Explain state equivalences 