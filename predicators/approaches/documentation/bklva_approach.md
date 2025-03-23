# BKLVA Setup and Approach

## Overview

BKLVA (Belief-space planning with K-fluents and language-based goal grounding) is an approach for integrated perception and belief-space planning using large vision-language models (VLMs) as state estimation modules.
It extends the task planning and perception pipeline of the bilevel planning approaches to a symbolic belief space using belief-space predicates and operators.
The approach works with both real robot and synthetic environments.

## Key Components

### Shared Pipeline Architecture

The BKLVA approach consists of these core components that are shared across both real robot and synthetic environments:

- **VLM Perception**: Uses vision-language models to evaluate visual predicates from images
- **Task Planner**: PDDL-based symbolic planner (referred to as the "oracle planner") that generates plans in belief space
- **Belief Space Planning**: Support for tasks involving uncertainty about object properties using K-fluents
- **Execution Monitoring**: Detects unexpected outcomes and triggers replanning when needed

The primary workflow in BKLVA is:
1. **Perception**: VLM evaluates predicates from current observations
2. **Planning**: Task planner generates a plan in belief space
3. **Execution**: Robot executes actions from the plan
4. **Monitoring**: System compares expected vs. actual outcomes and replans if they don't match

### Execution Monitoring

BKLVA supports different execution monitoring modes:
- **expected_atoms**: Compares the expected state with the perceived state after each action. If they don't match, the system replans from the current state.
- **mpc**: Model Predictive Control style monitoring with a rolling horizon.

### Environment Types

BKLVA can be run in two environment types:
1. **Real Robot Environment**: Uses live camera feeds and robot hardware
2. **Synthetic Environment**: Uses pre-captured images and a transition graph

The `MockSpotEnv` provides a synthetic environment with the following capabilities:
- Deterministic transition graph between pre-defined states
- Pre-captured images for each state
- Same predicate and operator structure as real robot
- Supports both ground truth and VLM-based predicate evaluation

## Real Robot Implementation

### Modeling Environment on Real Robot

The real robot implementation uses belief-space predicates and operators to model uncertain properties and observation actions:

- **Belief Predicates**: Track known/unknown state of object properties (e.g., `Unknown_ContainerEmpty`, `Known_ContainerEmpty`)
- **Physical Operators**: Actions that physically manipulate objects (e.g., `PickObjectFromTop`, `PlaceObjectOnTop`)
- **Observation Operators**: Actions that observe uncertain properties (e.g., `ObserveContainerContent`)

#### Example Belief Predicates

In the real Spot robot environment (`spot_env.py`), belief predicates are defined to model uncertainty about container contents:

```
_Unknown_ContainerEmpty = VLMPredicate(
    "Unknown_ContainerEmpty", [_container_type],
    prompt="[Answer: yes/no only] This predicate is true if you do not know whether the container is empty or not."
)

_Known_ContainerEmpty = VLMPredicate(
    "Known_ContainerEmpty", [_container_type],
    prompt="[Answer: yes/no only] This predicate is true if you can determine whether the container is empty or contains objects inside."
)

_BelieveTrue_ContainerEmpty = VLMPredicate(
    "BelieveTrue_ContainerEmpty", [_container_type],
    prompt="[Answer: yes/no only] This predicate is true if you believe the container is empty based on what you can see."
)

_BelieveFalse_ContainerEmpty = VLMPredicate(
    "BelieveFalse_ContainerEmpty", [_container_type],
    prompt="[Answer: yes/no only] This predicate is true if you believe the container contains objects based on what you can see."
)
```

#### Example Observation Operators

The robot's perception capabilities are modeled through observation operators that update belief states:

```
# ObserveCupContentFindEmpty (belief update action)
robot = Variable("?robot", _robot_type)
cup = Variable("?cup", _container_type)
surface = Variable("?surface", _immovable_object_type)
parameters = [robot, cup, surface]
preconds = {
    LiftedAtom(_On, [cup, surface]),
    LiftedAtom(_InHandViewFromTop, [robot, cup]),
    LiftedAtom(_HandEmpty, [robot]),
    LiftedAtom(_NotHolding, [robot, cup]),
    LiftedAtom(_Unknown_ContainerEmpty, [cup]),
}
add_effs = {
    LiftedAtom(_Known_ContainerEmpty, [cup]),
    LiftedAtom(_BelieveTrue_ContainerEmpty, [cup]),
}
del_effs = {
    LiftedAtom(_Unknown_ContainerEmpty, [cup])
}
```

This operator allows the robot to observe a cup's contents and update its belief state to reflect that the cup is empty.

### Running on Real Robot (Spot)

To run the perception and planning pipeline on the Spot robot:

1. Configure robot settings (see spot utils; TODO: add instructions)
2. Connect to the robot and launch the perception system
3. Run the planner with appropriate flags:

```
python3 predicators/main.py --env spot_pick_place_belief \
  --approach vlm_oracle_bilevel_planning --seed 0 \
  --perceiver spot_perceiver --vlm_model_name o3-mini \
  --bilevel_plan_without_sim True --execution_monitor expected_atoms
```

Examples of common Spot robot environment tasks:

```bash
# Run cup emptiness detection task with VLM predicate evaluation
python predicators/main.py --spot_robot_ip 192.168.80.3 \
  --spot_graph_nav_map b45-621 --env lis_spot_empty_cup_box_env \
  --approach spot_wrapper[oracle] --bilevel_plan_without_sim True \
  --seed 0 --perceiver spot_perceiver --spot_vlm_eval_predicate True \
  --num_train_tasks 0 --num_test_tasks 1 --vlm_model_name gpt-4o

# Run block placement task with execution monitoring
python predicators/main.py --spot_robot_ip 192.168.80.3 \
  --spot_graph_nav_map b45-621 --env lis_spot_block_in_box_env \
  --approach spot_wrapper[oracle] --bilevel_plan_without_sim True \
  --seed 0 --perceiver spot_perceiver --execution_monitor expected_atoms \
  --num_train_tasks 0 --num_test_tasks 1
```

## Synthetic Environment Implementation

### Modeling Synthetic Environment

The synthetic environment (`MockSpotEnv`) models the same predicates and operators as the real robot but uses pre-captured images instead of live perception. The key difference in implementation is the use of two types of predicates:

1. **GroundTruthPredicates**: Used to construct the deterministic transition graph of the environment
2. **VLMPredicates**: A subset of predicates that require visual perception and are evaluated at runtime by VLMs

In `mock_env_utils.py`, these are defined as:

```python
# GroundTruthPredicate example (used for environment transitions)
_Unknown_ContainerEmpty = GroundTruthPredicate("Unknown_ContainerEmpty", 
                                             [_container_type], 
                                             _dummy_classifier)

# VLMPredicate example (evaluated at runtime)
_Unknown_ContainerEmpty = VLMPredicate(
    "Unknown_ContainerEmpty", [_container_type],
    prompt="[Answer: yes/no only] This predicate is true if you do not know whether the container is empty or not."
)
```

The `MockSpotEnv` has these key characteristics:

- **State Representation**: Each state consists of images, objects in view/hand, and gripper state
- **Transition Graph**: Explicitly defined transitions between states via operators
- **Belief State Tracking**: Belief predicates track knowledge about uncertain object properties
- **Canonical States**: States that differ in key predicates (defined in `mock_env_creator_base.py`)

Key predicates that determine canonical states are manually defined in `mock_env_creator_base.py`:
```python
KEY_PREDICATES = {
    "Inside",      # Object containment
    "On",          # Object placement
    "DrawerOpen",  # Drawer state
    "DrawerClosed" # Drawer state
}
```

### Creating Mock Environments

The creation of mock environments involves a three-step process:

1. **Collect Images**: First, capture images for each state in your task
2. **Import Images**: Use the creator tool in "image mode" to import your images
3. **Create States and Transitions**: Use the creator tool in "state mode" to define states and transitions

The state mode is particularly important as it defines the canonical states based on the key predicates. The creator tool will automatically identify which states are equivalent based on these predicates.

#### 1. Image Collection Phase

Capture images for different states of the environment. For example, when creating a cup emptiness task:
- Take photos of the initial state (cups on table)
- Take photos of the robot viewing the cups
- Take photos from the hand camera perspective looking into cups
- Take photos of any necessary manipulation actions

Images can be captured with any camera (e.g., phone camera) and placed in a directory structure.

#### 2. Creating the Environment

```bash
# Import images into a mock environment (image mode)
python -m predicators.spot_utils.mock_env.mock_env_creator_manual \
  --output_dir mock_env_data/my_cup_emptiness_task \
  --image_dir path/to/images \
  --env_name MockSpotCupEmptiness \
  --mode image

# Define states and transitions (state mode) - CRITICAL STEP
python -m predicators.spot_utils.mock_env.mock_env_creator_manual \
  --output_dir mock_env_data/my_cup_emptiness_task \
  --env_name MockSpotCupEmptiness \
  --mode state
```

The state mode step is where you define which predicates are true in each state and create the transition graph. The creator tool will focus on canonical states based on KEY_PREDICATES defined in mock_env_creator_base.py.

You can also use automated tests to generate environments:

```bash
# Create drawer cleaning environment from phone images
python -m pytest tests/mock_robot/build_phone_drawer_cleaning.py -v -s

# Create pick and place environment with 2 cups
python -m pytest tests/mock_robot/build_phone_pick_place_2_cups.py -v -s
```

#### 3. Testing the Environment

Run tests to verify the environment was created correctly:

```bash
# Test transitions between states
python -m pytest tests/mock_robot/test_mock_env_transitions.py -v -s

# Test for cup emptiness environments
python -m pytest tests/mock_robot/test_mock_env_cup_emptiness.py -v -s
```

## Available Synthetic Tasks

The framework includes several synthetic environments defined in `mock_spot_env.py`, each modeling different tasks of increasing complexity:

### 1. Pick and Place Tasks (`MockSpotPickPlaceTwoCupEnv`)
Basic manipulation of objects:
- **Description**: Two cups on a table with a goal to move them to target locations
- **Key Challenge**: Basic object manipulation
- **Data Directory**: `mock_env_data/MockSpotPickPlaceTwoCupEnv`

### 2. Cup Emptiness Tasks (`MockSpotCupEmptiness`)
Belief-space planning for cup contents:
- **Description**: Two cups with unknown contents (empty or containing objects)
- **Key Challenge**: Visual observation to detect cup contents and update beliefs
- **Goal**: Place empty cups in a container
- **Data Directory**: `mock_env_data/MockSpotCupEmptiness`

### 3. Drawer Cleaning Tasks (`MockSpotDrawerCleaningEnv`)
Complex task with drawer and multiple objects:
- **Description**: Objects inside a drawer that need to be removed and placed elsewhere
- **Key Challenge**: Drawer manipulation and object detection inside containers
- **Data Directory**: `mock_env_data/MockSpotDrawerCleaningEnv`

### 4. Weight Sorting Tasks (`MockSpotSortWeight`)
Reasoning about physical properties:
- **Description**: Objects with different weights that need to be sorted
- **Key Challenge**: "Measuring" object weights and making comparisons
- **Goal**: Place heavy objects in one container and light objects in another
- **Data Directory**: `mock_env_data/MockSpotSortWeight`

## Planning Approaches and Running Experiments

The system supports running different planning approaches on the synthetic environments. These can be systematically evaluated using the `scripts/mock_experiments.py` script, which provides a convenient way to run multiple planners on the same environment.

### Available Planning Approaches

#### 1. BKLVA: VLM Perception + PDDL Task Planner (`oracle`)
Our BKLVA approach combines VLM perception with a classic PDDL task planner:
- **Perception**: Uses VLM to evaluate predicates from images
- **Planning**: Uses a PDDL task planner to generate plans in belief space
- **Key Strength**: Combines perceptual capabilities of VLMs with the reliability of symbolic planning

```bash
# Running BKLVA approach
python predicators/main.py --env mock_spot_pick_place_two_cup \
  --approach oracle --seed 0 --perceiver mock_spot_perceiver \
  --mock_env_vlm_eval_predicate True --num_train_tasks 0 \
  --num_test_tasks 1 --bilevel_plan_without_sim True \
  --horizon 20
```

#### 2. Random Options Baseline (`random_options`)
A baseline approach that randomly selects options:
- **Perception**: Uses regular perception without VLM
- **Planning**: Randomly selects actions from available options
- **Key Use**: Serves as a lower bound baseline for performance comparison

```bash
# Running random options baseline
python predicators/main.py --env mock_spot_drawer_cleaning \
  --approach random_options --seed 0 --perceiver mock_spot_perceiver \
  --random_options_max_tries 1000 --max_num_steps_option_rollout 100 \
  --num_train_tasks 0 --num_test_tasks 1 --bilevel_plan_without_sim True \
  --timeout 60 --horizon 20
```

#### 3. LLM Open-Loop Planner (`llm_open_loop`)
Uses a large language model to generate plans:
- **Perception**: Regular perception, with state descriptions provided to LLM
- **Planning**: LLM generates entire plan before execution
- **Key Feature**: Can leverage commonsense reasoning from LLMs, but lacks visual perception

```bash
# Running LLM open loop planner
python predicators/main.py --env mock_spot_drawer_cleaning \
  --approach llm_open_loop --seed 0 --perceiver mock_spot_perceiver \
  --llm_model_name gpt-4o --llm_temperature 0.2 \
  --num_train_tasks 0 --num_test_tasks 1 --bilevel_plan_without_sim True \
  --horizon 20 --load_approach
```

#### 4. LLM Closed-Loop Planner (`llm_open_loop` with execution monitoring)
LLM planner with execution monitoring to detect and recover from failures:
- **Perception**: Regular perception plus execution monitoring
- **Planning**: LLM generates plan, with replanning when execution outcomes don't match expectations
- **Key Advantage**: More robust to execution failures and surprises than open-loop planning

```bash
# Running LLM closed loop planner (with MPC monitoring)
python predicators/main.py --env mock_spot_cup_emptiness \
  --approach llm_open_loop --seed 0 --perceiver mock_spot_perceiver \
  --llm_model_name gpt-4o --llm_temperature 0.2 --execution_monitor mpc \
  --num_train_tasks 0 --num_test_tasks 1 --bilevel_plan_without_sim True \
  --horizon 20 --load_approach
```

#### 5. VLM Open-Loop Planner (`vlm_open_loop`)
Uses a vision-language model for both perception and planning:
- **Perception**: VLM evaluates visual predicates from images
- **Planning**: VLM generates entire plan before execution
- **Key Feature**: Integrates visual perception directly into planning

```bash
# Running VLM open loop planner
python predicators/main.py --env mock_spot_sort_weight \
  --approach vlm_open_loop --seed 0 --perceiver mock_spot_perceiver \
  --mock_env_vlm_eval_predicate True --vlm_model_name gpt-4o \
  --vlm_temperature 0.2 --num_train_tasks 0 --num_test_tasks 1 \
  --bilevel_plan_without_sim True --load_approach --horizon 20
```

#### 6. VLM Closed-Loop Planner (`vlm_open_loop` with execution monitoring)
VLM planner with execution monitoring:
- **Perception**: VLM perception with execution monitoring
- **Planning**: VLM generates plan, with replanning when execution outcomes don't match expectations
- **Key Advantage**: Combines visual perception with robust execution monitoring

```bash
# Running VLM closed loop planner (with MPC monitoring)
python predicators/main.py --env mock_spot_drawer_cleaning \
  --approach vlm_open_loop --seed 0 --perceiver mock_spot_perceiver \
  --mock_env_vlm_eval_predicate True --vlm_model_name gpt-4o \
  --vlm_temperature 0.2 --execution_monitor mpc \
  --num_train_tasks 0 --num_test_tasks 1 --bilevel_plan_without_sim True \
  --load_approach --horizon 20
```

#### 7. VLM with Image History (`vlm_open_loop` with image history)
VLM planner that maintains history of past observations:
- **Perception**: VLM with access to previous image observations
- **Planning**: VLM generates plans with context from image history
- **Key Feature**: Can reason about changes over time and maintain state information

```bash
# Running VLM planner with image history
python predicators/main.py --env mock_spot_pick_place_two_cup \
  --approach vlm_open_loop --execution_monitor expected_atoms \
  --bilevel_plan_without_sim True --seed 0 \
  --perceiver mock_spot_perceiver --mock_env_vlm_eval_predicate True \
  --vlm_enable_image_history True --vlm_max_history_steps 5 \
  --vlm_max_images_per_prompt 10 --num_train_tasks 0 \
  --num_test_tasks 1 --vlm_model_name gpt-4o \
  --vlm_temperature 0.7 --horizon 20 --load_approach
```

#### 8. VLM Captioning Approach (`vlm_captioning`)
Uses VLM to caption scenes and derive state information:
- **Perception**: VLM generates detailed captions of the scene
- **Planning**: Uses the captions to inform planning decisions
- **Key Feature**: Extracts rich semantic information from images through captions

```bash
# Running VLM captioning approach
python predicators/main.py --env mock_spot_drawer_cleaning \
  --approach vlm_captioning --seed 0 --perceiver vlm_perceiver \
  --vlm_model_name gpt-4o --vlm_temperature 0.2 --execution_monitor mpc \
  --num_train_tasks 0 --num_test_tasks 1 --bilevel_plan_without_sim True \
  --horizon 20 --load_approach
```

## Running Available Synthetic Tasks

The framework includes several synthetic tasks of increasing complexity:

### Commands

#### 1. Pick and Place Tasks

Basic manipulation of objects:

- **mock_spot_pick_place**: Simple pick and place with one object
- **mock_spot_pick_place_two_cup**: Pick and place with two cups

#### 2. Belief-Space Tasks
Tasks requiring observation and belief updates:

- **Cup Emptiness**: Determine if cups contain objects
  ```bash
  python predicators/main.py --env mock_spot_cup_emptiness \
    --approach oracle --seed 0 --perceiver mock_spot_perceiver \
    --mock_env_vlm_eval_predicate True --num_train_tasks 0 \
    --num_test_tasks 1 --log_rich True --bilevel_plan_without_sim True
  ```
- **Drawer Cleaning**: Clean up objects from a drawer
  ```bash
  python predicators/main.py --env mock_spot_drawer_cleaning \
    --approach vlm_open_loop --seed 0 --perceiver mock_spot_perceiver \
    --mock_env_vlm_eval_predicate True --num_train_tasks 0 \
    --num_test_tasks 1 --vlm_model_name gpt-4o \
    --bilevel_plan_without_sim True --load_approach
  ```
- **Weight Sorting**: Sort objects based on relative weight
  ```bash
  python predicators/main.py --env mock_spot_sort_weight \
    --approach oracle --seed 0 --perceiver mock_spot_perceiver \
    --mock_env_vlm_eval_predicate True --num_train_tasks 0 \
    --num_test_tasks 1 --log_rich True --bilevel_plan_without_sim True
  ```

### Running Systematic Experiments

The `scripts/mock_experiments.py` script provides a convenient way to run multiple planners on the same environment for systematic comparison:

```bash
# Run all planners on drawer cleaning task
python scripts/mock_experiments.py --env mock_spot_drawer_cleaning

# Run specific planner on cup emptiness task
python scripts/mock_experiments.py --env mock_spot_cup_emptiness --planner vlm_closed_loop

# Run with different seed
python scripts/mock_experiments.py --env mock_spot_sort_weight --seed 42
```

## Testing and Development

### Test Suite Overview

The `tests/mock_robot` directory contains a comprehensive set of tests for developing and validating mock environments:

#### State and Transition Tests
- **test_mock_env_transitions.py**: Tests state transitions and operator effects
- **test_mock_env_loading.py**: Tests loading environments from disk
- **test_mock_env_graph_building.py**: Tests building the transition graph

#### Predicate and Perception Tests
- **test_mock_env_view.py**: Tests object viewing functionality
- **test_mock_spot_perceiver.py**: Tests the perception system for mock environments

#### Task-Specific Tests
- **Test for cup emptiness environments**:
  ```bash
  python -m pytest tests/mock_robot/test_mock_env_cup_emptiness.py -v -s
  ```
  This test verifies:
  - Correct behavior of cup emptiness predicates (Unknown/Known/BelieveTrue/BelieveFalse)
  - Observation operators for updating beliefs
  - Planning with belief-space operators

- **Test for drawer cleaning environments**:
  ```bash
  python -m pytest tests/mock_robot/test_mock_env_drawer_compare.py -v -s
  ```
  This test verifies:
  - Drawer open/closed state tracking
  - Object containment beliefs
  - Planning to clean objects from drawers

- **Test for weight sorting environments**:
  ```bash
  python -m pytest tests/mock_robot/test_mock_env_sort_weight.py -v -s
  ```
  This test verifies:
  - Weight measurement operators
  - Weight comparison predicates
  - Planning to sort objects by weight

#### Environment Creation Tests
- **test_mock_env_manual_images.py**: Tests creating environments with manual images
- **build_phone_drawer_cleaning.py**: Creates drawer cleaning environment from phone images
- **build_phone_pick_place_2_cups.py**: Creates pick and place environment with 2 cups

### Planning Tests
- **test_mock_spot_planning.py**: Tests planning in the mock environment
  ```bash
  python -m pytest tests/mock_robot/test_mock_spot_planning.py -v -s
  ```
  This test verifies:
  - Plan generation with various planning approaches
  - Plan execution with different monitoring strategies
  - Handling of unexpected outcomes

### Perception Pipeline

In the BKLVA approach, perception works as follows:

1. For the real robot, VLMPredicates are evaluated directly from robot camera images
2. For synthetic environments, VLMPredicates are evaluated from pre-captured images
3. GroundTruthPredicates are used for environment transitions (not requiring perception)
4. The planner uses the evaluated VLMPredicates to make decisions about what actions to take

This separation allows the system to work with both real-time perception on real robots and simulated perception in synthetic environments.

## Development Workflow

### 1. Define Task Requirements
- Identify required belief-space predicates
- Define necessary belief-space operators
- Determine task goals (in belief space)

### 2. Create Mock Environment
- Collect images for required states
- Import images using the creator tool (image mode)
- Define states and transitions using the creator tool (state mode)
- Test environment graph

### 3. Test With VLM Perception + PDDL Planner
- Verify environment correctness with the basic BKLVA approach
- Debug any transition or belief update issues

### 4. Test With Foundation Model-based Planning Approaches
- Test LLM- or VLM-based planning with execution monitoring
- Compare performance to basic approach

### 5. Deploy to Real Robot
- Define operators and predicates for real robot environment
- Update perception system as needed
- Test on real robot with same planning approach

Refer to the mock environment documentation for detailed information on creating and working with synthetic environments.

