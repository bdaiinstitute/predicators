# Creating New Mock Robot Tasks (Inspired by Spot Environment)

This guide explains how to create new mock tasks that simulate Spot robot behaviors. Mock tasks are used to test and develop task planning without requiring the physical robot. We'll use the two-cup pick-and-place task as an example.

## Overview

A mock task consists of:
1. A task-specific environment that defines objects, states, and valid actions
2. A sequence of RGB-D images that represent the robot's observations
3. A transition graph showing how actions change the environment state
4. Metadata about object visibility and gripper state for each observation

## 1. Create a Task-Specific Environment

Create a new environment class that inherits from `MockSpotEnv`. This class defines what objects exist in the environment and their relationships:

```python
class MockSpotPickPlaceTwoCupEnv(MockSpotEnv):
    """Mock environment for Spot robot to pick and place two cups."""
    
    def __init__(self, name: str = "pick_place") -> None:
        super().__init__()
        self.name = name
        
        # Define task objects using Spot's type system
        self.robot = Object("robot", _robot_type)
        self.cup1 = Object("cup1", _container_type)
        self.cup2 = Object("cup2", _container_type)
        self.table = Object("table", _immovable_object_type)
        self.target = Object("target", _container_type)
        
        # Store all objects that exist in this task
        self.objects = {self.robot, self.cup1, self.cup2, self.table, self.target}
        
        # Define initial state (what is true at the start)
        self.initial_atoms = {
            # Robot starts with empty hand
            GroundAtom(_HandEmpty, [self.robot]),
            GroundAtom(_NotHolding, [self.robot, self.cup1]),
            GroundAtom(_NotHolding, [self.robot, self.cup2]),
            
            # Cups start on the table
            GroundAtom(_On, [self.cup1, self.table]),
            GroundAtom(_On, [self.cup2, self.table]),
            GroundAtom(_On, [self.target, self.table]),
            
            # Objects are reachable and visible
            GroundAtom(_Reachable, [self.robot, self.cup1]),
            GroundAtom(_Reachable, [self.robot, self.cup2]),
            GroundAtom(_InHandView, [self.robot, self.cup1]),
            GroundAtom(_InHandView, [self.robot, self.cup2]),
            
            # ... other necessary predicates ...
        }
        
        # Define goal state (what should be true at the end)
        self.goal_atoms = {
            GroundAtom(_Inside, [self.cup1, self.target]),
            GroundAtom(_Inside, [self.cup2, self.target])
        }
```

## 2. Select Required Operators

Choose which actions from the Spot robot's capabilities are needed for this task:

```python
def _create_operators(self) -> Iterator[STRIPSOperator]:
    """Define what actions are available in this task."""
    all_operators = list(super()._create_operators())
    
    # Select only the operators needed for pick-and-place
    op_names_to_keep = {
        "MoveToReachObject",    # Move robot to reach an object
        "MoveToHandViewObject", # Move hand camera to see object
        "PickObjectFromTop",    # Pick up object from above
        "PlaceObjectOnTop",     # Place object on surface
        "DropObjectInside"      # Place object in container
    }
    
    # Filter operators
    for op in all_operators:
        if op.name in op_names_to_keep:
            yield op
```

## 3. Create a Test File

Create a test that demonstrates the task execution and validates the mock environment:

```python
def test_two_cup_pick_place_with_manual_images():
    """Test a mock task where Spot moves two cups into a target container."""
    # Set up configuration
    test_name = "test_two_cup_pick_place"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create mock environment for this specific task
    env = MockSpotPickPlaceTwoCupEnv(name=test_name)
    
    # Create environment creator to manage states and images
    creator = ManualMockEnvCreator(test_dir, env=env)
    
    # Generate and visualize the task's transition graph
    name = f'Transition Graph, {env.name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects, task_name=name)
```

## 4. Add Mock Observations

For each state in the task, provide mock RGB-D images that represent what Spot's cameras would see:

```python
# Define what Spot observes in each state
test_state_images = {
    "state_0": {  # Initial state - both cups on table
        "view1": {  # Front view
            "cam1": {  # Hand camera
                "rgb_img": (rgb_path, "rgb"),
                "depth_img": (depth_path, "depth")
            },
            "cam2": {  # Navigation camera
                "rgb_img": (rgb_path, "rgb")
            }
        },
        "view2": {  # Side view
            "cam1": {
                "rgb_img": (rgb_path, "rgb")
            }
        }
    },
    "state_1": {  # Holding first cup
        "view1": {
            "cam1": {
                "rgb_img": (rgb_path, "rgb"),
                "depth_img": (depth_path, "depth")
            }
        }
    },
    # ... states for entire task sequence ...
}

# Add each state's observations to the environment
for state_id, views in test_state_images.items():
    creator.add_state_from_multiple_images(
        views,
        state_id=state_id,
        objects_in_view=list({env.cup1.name, env.cup2.name, env.table.name, env.target.name}),
        objects_in_hand=[env.cup1.name] if state_id == "state_1" else 
                       [env.cup2.name] if state_id == "state_3" else [],
        gripper_open=(state_id not in ["state_1", "state_3"])
    )
```

## 5. Directory Structure

The mock task creates this structure to simulate Spot's data:
```
mock_env_data/test_two_cup_pick_place/
├── images/                      # Mock camera observations
│   ├── state_0/                # Initial state
│   │   ├── view1/              # Front view
│   │   │   ├── cam1_rgb.npy    # Hand camera RGB
│   │   │   ├── cam1_depth.npy  # Hand camera depth
│   │   │   ├── cam2_rgb.npy    # Navigation camera RGB
│   │   │   └── metadata.yaml   # What objects are visible/held
│   │   └── view2/              # Side view
│   │       └── cam1_rgb.npy    # Another camera angle
│   └── state_1/                # Next state
│       └── ...
├── transitions/                 # Task execution visualization
│   └── Transition Graph.html   # Shows action sequence
└── plan.yaml                   # Task execution plan
```

## Tips for Mock Tasks

1. **Task Definition**:
   - Keep tasks focused and specific
   - Define clear initial and goal states
   - Only include necessary objects and predicates

2. **Operators**:
   - Use only operators that match Spot's capabilities
   - Include all prerequisites (e.g., need to see object before picking)
   - Consider physical constraints (e.g., reachability)

3. **Mock Images**: 
   - Use realistic camera positions (hand camera, navigation camera)
   - Include depth information for manipulation
   - Match image content to state description

4. **Testing**:
   - Verify all state transitions are possible
   - Check that images match state descriptions
   - Validate gripper state and object tracking
   - Test with different initial conditions 