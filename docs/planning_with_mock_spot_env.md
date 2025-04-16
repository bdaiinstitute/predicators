# Planning with the Mock Spot Environment

This guide explains how to use the mock Spot environment for testing planning functionality without requiring actual hardware. It covers environment setup, task creation, perception handling, and the planning pipeline.

## Overview

The mock Spot environment provides a way to test planning using saved images and VLM-based perception, similar to the real Spot environment but without requiring physical robot interaction. The key components are:

- **MockSpotEnv**: Simulates the Spot environment for testing, handles state transitions and action execution
- **MockSpotPerceiver**: Manages image-based perception using saved RGBD images, tracks environment state
- **ManualMockEnvCreator**: Creates and manages mock states, handles RGB-D image storage

## VLM Predicate Support

The mock perceiver supports VLM-based predicates similar to the real Spot environment:

1. Configuration:
```python
utils.reset_config({
    "env": "mock_spot",
    "spot_vlm_eval_predicate": True,  # Enable VLM predicate evaluation
})
```

2. VLM Predicate Definition:
```python
# Define types for predicates
obj_type = Type("object", ["x", "y", "z"])
container_type = Type("container", ["x", "y", "z"], parent=obj_type)

# Create VLM predicates
inside_pred = VLMPredicate(
    "Inside", [obj_type, container_type],
    prompt="This predicate is true if the first object is inside the second object (container)."
)
on_pred = VLMPredicate(
    "On", [obj_type, obj_type],
    prompt="This predicate is true if the first object is on top of the second object."
)
```

3. State Management:
```python
# Initialize perceiver with VLM predicates
perceiver.update_state(
    gripper_open=True,
    objects_in_view=set(),
    objects_in_hand=set(),
    vlm_predicates={inside_pred, on_pred}
)

# Save images for perception
perceiver.save_image(RGBDImageWithContext(
    rgb=rgb_image,
    depth=depth_image,
    camera_name="mock_camera",
    image_rot=0.0,
    world_tform_camera=SE3Pose(x=0.0, y=0.0, z=0.0, rot=np.eye(3)),
    depth_scale=1.0,
    transforms_snapshot=FrameTreeSnapshot(),
    frame_name_image_sensor="mock_camera",
    camera_model=None
))
```

## Pipeline Understanding

### 1. Main Pipeline Flow (main.py)
1. **Environment Creation**
   ```python
   env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
   ```
   - Creates MockSpotEnv instance
   - Sets up action space and predicates
   - Initializes environment state

2. **Perceiver Setup**
   ```python
   perceiver = create_perceiver(CFG.perceiver)  # Creates MockSpotPerceiver
   ```
   - Manages observations from mock environment
   - Handles RGB-D images and object states
   - Tracks gripper and object states

3. **Task Creation**
   ```python
   env_train_tasks = env.get_train_tasks()
   train_tasks = [perceiver.reset(t) for t in env_train_tasks]
   ```
   - Defines initial and goal states
   - Sets up object configurations
   - Creates task instances

4. **Approach Setup**
   ```python
   approach = create_approach(approach_name, preds, options, env.types,
                            env.action_space, stripped_train_tasks)
   ```
   - Sets up planning approach (oracle for testing)
   - Configures predicates and options
   - Initializes action space

5. **Planning and Execution**
   ```python
   cogman = CogMan(approach, perceiver, execution_monitor)
   results = _run_testing(env, cogman)
   ```
   - Plans actions to achieve goals
   - Executes actions in environment
   - Monitors execution progress

### 2. Mock Environment Components

1. **ManualMockEnvCreator** (from test_mock_env_manual_images.py)
   - Creates and manages mock states
   - Handles RGB-D image storage
   - Tracks object and gripper states
   - Manages state transitions

2. **MockSpotPerceiver** (mock_spot_perceiver.py)
   - Provides observations to planner
   - Manages mock sensor data
   - Tracks environment state
   - Interfaces with MockSpotEnv

## Inference Time Setup

### 1. MockSpotPerceiver Configuration
```python
# Enable VLM predicate evaluation in config
utils.reset_config({
    "env": "mock_spot",
    "spot_vlm_eval_predicate": True,  # Enable VLM predicate evaluation
})

# Initialize perceiver with data directory
perceiver = MockSpotPerceiver(data_dir="path/to/mock_env_data")
```

### 2. State Management
The mock perceiver maintains several key pieces of state:
- RGBD images with camera context
- Objects visible to the robot
- Objects held in the gripper
- Gripper state (open/closed)
- VLM predicates and atoms for perception-based planning

### 3. Observation Flow
1. **Image Updates**:
```python
# Save new RGBD image for perception
perceiver.save_image(RGBDImageWithContext(
    rgb=rgb_image,
    depth=depth_image,
    camera_name="mock_camera",
    image_rot=0.0,
    world_tform_camera=SE3Pose(x=0.0, y=0.0, z=0.0, rot=np.eye(3)),
    depth_scale=1.0,
    transforms_snapshot=FrameTreeSnapshot(),
    frame_name_image_sensor="mock_camera",
    camera_model=None
))
```

2. **State Updates**:
```python
# Update environment state
perceiver.update_state(
    gripper_open=True,
    objects_in_view={"cup", "table"},
    objects_in_hand=set(),
    vlm_predicates=vlm_predicates,  # Set of VLM predicates
    vlm_atom_dict=vlm_atom_dict,    # Dictionary of VLM atoms
    camera_images=camera_images      # Current camera images
)
```

3. **Getting Observations**:
```python
# Get current observation
obs = perceiver.get_observation()
assert obs.rgbd is not None          # RGBD image data
assert obs.gripper_open              # Gripper state
assert obs.objects_in_view          # Set of visible objects
assert obs.objects_in_hand          # Set of held objects
assert obs.vlm_predicates is not None  # VLM predicates if enabled
assert obs.vlm_atom_dict is not None   # VLM atoms if enabled
```

### 4. Directory Structure
Mock environment data should be organized as:
```
mock_env_data/
├── images/
│   ├── state_0/
│   │   ├── rgb.npy
│   │   └── depth.npy
│   ├── state_1/
│   │   ├── rgb.npy
│   │   └── depth.npy
│   └── ...
└── graph.json  # Contains state and transition information
```

### 5. Best Practices
1. **Image Management**:
   - Store RGBD images in numpy format
   - Include camera context for proper transformation
   - Maintain consistent image dimensions

2. **State Tracking**:
   - Update state after each action
   - Track objects entering/leaving view
   - Maintain gripper state accuracy
   - Keep VLM predicates synchronized

3. **Error Handling**:
   - Validate image data before saving
   - Check state consistency after updates
   - Handle missing or corrupt data gracefully

4. **Testing**:
   - Verify image loading/saving
   - Test state transitions
   - Validate VLM predicate evaluation
   - Check observation consistency

## Directory Structure

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

## Environment Setup

1. Configure the mock environment:
```python
utils.reset_config({
    "env": "mock_spot",
    "approach": "oracle",
    "num_test_tasks": 1,
    "mock_env_data_dir": "path/to/mock_env_data",
    "spot_vlm_eval_predicate": True
})
```

2. Create an instance of MockSpotEnv:
```python
env = create_new_env("mock_spot")
```

## Task Creation

Define objects, initial state, and goal state for a task:

```python
# Create objects
robot = Object("robot", next(t for t in env.types if t.name == "robot"))
cube = Object("cube", next(t for t in env.types if t.name == "movable_object"))
target = Object("target", next(t for t in env.types if t.name == "immovable_object"))

# Define initial state
init_atoms = {
    GroundAtom(next(p for p in env.predicates if p.name == "HandEmpty"), [robot]),
    GroundAtom(next(p for p in env.predicates if p.name == "InView"), [robot, cube]),
    GroundAtom(next(p for p in env.predicates if p.name == "On"), [cube, target])
}
init_state = State({
    robot: np.array([0.0, 0.0, 0.0]),  # x,y,z position
    cube: np.array([0.5, 0.5, 0.0]),
    target: np.array([1.0, 1.0, 0.0])
}, init_atoms)

# Define goal
goal_atoms = {
    GroundAtom(next(p for p in env.predicates if p.name == "HandEmpty"), [robot]),
    GroundAtom(next(p for p in env.predicates if p.name == "On"), [cube, target])
}

# Create task
task = EnvironmentTask(init_state, goal_atoms)
```

The task creation process involves:

1. Creating objects that will be part of the task, such as the robot, manipulable objects (e.g., cube), and target locations. The types of these objects should match the types defined in the environment.

2. Defining the initial state of the task, which includes:
   - The state of each object, represented as a dictionary mapping objects to their numeric properties (e.g., position)
   - The initial set of true atoms, specifying the relationships and properties of objects at the start of the task

3. Defining the goal state of the task, which is a set of atoms that should be true at the end of the task execution. These atoms represent the desired final state of the objects.

4. Creating an `EnvironmentTask` instance that encapsulates the initial state and goal atoms, representing the complete task specification.

By following this process, you can define custom tasks for the mock Spot environment, specifying the objects involved, their initial configuration, and the desired goal state. The planner will then attempt to find a sequence of actions that transforms the initial state into the goal state.

## Perceiver Setup

Initialize MockSpotPerceiver and reset it for a task:

```python
# Create perceiver
perceiver = MockSpotPerceiver(data_dir="path/to/mock_env_data")

# Initialize with task
perceiver.reset(task)

# Verify observation
obs = perceiver.get_observation()
assert obs is not None
assert obs.vlm_predicates is not None
```

## Planning Pipeline

1. Create approach, perceiver, execution monitor, and CogMan:
```python
# Create approach
options = get_gt_options(env.get_name())
approach = create_approach("oracle", env.predicates, options, env.types, env.action_space, [])

# Create execution monitor
exec_monitor = create_execution_monitor("trivial")

# Create cogman
cogman = CogMan(approach, perceiver, exec_monitor)
```

2. Run planning:
```python
# Reset cogman with task
cogman.reset(task)

# Run planning
metrics = _run_testing(env, cogman)
assert metrics["num_solved"] > 0
```

## Action Wrapping

The mock Spot environment uses a graph-based state representation where transitions are defined by operators. Each operator has a corresponding `ParameterizedOption` that creates actions with proper operator information.

### Creating Actions with Operator Information

1. Get options from the environment:
```python
# Get options from MockSpotGroundTruthOptionFactory
options = get_gt_options("mock_spot")

# Each option corresponds to an operator and will create actions with proper extra_info
for option in options:
    # When the option's policy is called, it creates an Action with:
    # - A dummy action array (since we don't need real continuous control)
    # - extra_info containing the operator name
    action = option.policy(state, memory, objects, params)
    # action.extra_info will have {"operator_name": option.name}
```

2. The environment's `step` method uses this operator information to:
   - Look up the appropriate transition in the graph
   - Update the state based on the operator's effects
   - Return the next observation

### Example: Pick and Place Task

```python
# Create environment and get options
env = MockSpotEnv()
options = get_gt_options(env.get_name())

# Create approach with options
approach = create_approach("oracle", env.predicates, options, env.types, env.action_space, tasks)

# Run episode
max_steps = 20
(states, actions), solved, metrics = _run_episode(
    cogman,
    env,
    "test",
    0,
    max_steps,
    do_env_reset=True,
    terminate_on_goal_reached=True
)

# Verify action sequence
action_names = [action.extra_info["operator_name"] for action in actions]
expected_operators = {"PickObjectFromTop", "DropObjectInside"}
assert all(name in expected_operators for name in action_names)
```

### Key Components

1. `MockSpotGroundTruthOptionFactory` creates options that match the environment's operators:
```python
def get_options(cls, env_name: str, types: Dict[str, Type],
               predicates: Dict[str, Predicate],
               action_space: Box) -> Set[ParameterizedOption]:
    # Creates one option per operator
    # Each option's policy stores operator name in action's extra_info
```

2. Option policies create actions with operator information:
```python
def policy(state: State, memory: Dict, objects: Sequence[Object],
          params: Array) -> Action:
    # Create a dummy action array but store operator name in extra_info
    arr = np.zeros(1, dtype=np.float32)
    return Action(arr, extra_info={"operator_name": operator_name})
```

3. The environment's `step` method uses this information to execute transitions:
```python
def step(self, action: Action) -> _MockSpotObservation:
    # Get operator name from action
    operator_name = action.extra_info["operator_name"]
    # Look up transition in graph
    next_state_id = self._str_transitions[self._current_state_id][operator_name]
    # Update state and return observation
```

## Testing

1. Set up test data using ManualMockEnvCreator:
```python
# Create test directory
test_dir = "mock_env_data/pick_place_test"
creator = ManualMockEnvCreator(test_dir)
```

2. Write unit tests:
```python
def test_mock_pick_and_place_planning():
    # Setup
    test_dir = "mock_env_data/pick_place_test"
    creator = ManualMockEnvCreator(test_dir)
    setup_mock_environment(creator)
    
    # Create environment and perceiver
    env = MockSpotEnv()
    perceiver = MockSpotPerceiver(test_dir)
    
    # Create task
    robot = Object("robot", robot_type)
    block = Object("block", movable_type)
    table = Object("table", surface_type)
    
    init_atoms = {
        GroundAtom(predicates["HandEmpty"], [robot]),
        GroundAtom(predicates["On"], [block, table])
    }
    goal_atoms = {
        GroundAtom(predicates["HandEmpty"], [robot]),
        GroundAtom(predicates["On"], [block, target])
    }
    task = Task(init_atoms, goal_atoms)
    
    # Run planning
    approach = create_approach("oracle", predicates, options)
    cogman = CogMan(approach, perceiver)
    results = run_testing(env, cogman)
    
    # Verify results
    assert results["num_solved"] == 1
    traj = results["trajectory"]
    verify_trajectory_states(traj, expected_states)
```

3. Run integration tests:
```python
def test_mock_spot_pipeline():
    # Setup mock data
    setup_mock_environment()
    
    # Run main with test config
    args = [
        "--env", "mock_spot",
        "--approach", "oracle",
        "--seed", "0",
        "--num_test_tasks", "1"
    ]
    results = run_main_with_args(args)
    
    # Verify results
    verify_planning_results(results)
    verify_execution_results(results)
```

## Verification

Verify state transitions, plan correctness, and perception:

- Check object positions
- Verify gripper state
- Validate transitions
- Verify action sequence
- Check predicate changes
- Validate goal achievement
- Validate observations
- Check image loading
- Verify object tracking

## Execution Steps

1. Set up environment:
```bash
# Create test directory
mkdir -p mock_env_data/pick_place_test
# Copy example images
cp tests/spot_utils/example_manual_mock_task_1/* mock_env_data/pick_place_test/
```

2. Run unit tests:
```bash
pytest tests/spot_utils/test_mock_spot_planning.py -v
```

3. Run integration test:
```bash
python predicators/main.py --env mock_spot --approach oracle --seed 0
```

4. Verify results:
- Check test output
- Examine generated plans
- Verify state transitions
- Review visualization output

## Next Steps

- Add more complex scenarios
- Test error cases
- Add belief space planning tests 