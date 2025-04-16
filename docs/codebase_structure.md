# Codebase Structure and Usage Guide

## Overview
This codebase implements a framework for bilevel planning with learned neuro-symbolic relational abstractions. The system is designed to work with both simulated environments and real robotic systems, with a focus on learning symbolic operators for task and motion planning.

## Directory Structure

```
predicators/
├── approaches/           # Planning and learning approaches
│   ├── base_approach.py
│   ├── oracle.py
│   └── ...
├── envs/                # Environment implementations
│   ├── base_env.py
│   ├── sokoban.py
│   └── ...
├── perception/          # Perception systems
│   ├── base_perceiver.py
│   ├── spot_perceiver.py
│   └── ...
├── nsrt_learning/       # NSRT learning implementation
│   └── nsrt_learning_main.py
├── spot_utils/          # Spot robot utilities
│   ├── perception/
│   └── mock_env/
└── structs.py          # Core data structures
```

## Core Components

### 1. Environment System (`predicators/envs/`)
Environments define the world state and how it changes with actions.

#### BaseEnv Interface
```python
class BaseEnv:
    @property
    def predicates(self) -> Set[Predicate]:
        """Return the predicates for this environment."""
        
    @property
    def types(self) -> Set[Type]:
        """Return the types for this environment."""
        
    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Reset to initial state of given task."""
        
    def step(self, action: Action) -> Observation:
        """Execute action and return observation."""
```

#### Environment Categories
1. **Simulated Environments**
   - Fully observable (e.g., Sokoban, Cover)
   - Direct state transitions
   - No perception needed

2. **Robot Environments**
   - Partially observable (e.g., Spot)
   - Real-world interaction
   - Requires perception pipeline
   - Often needs mock version for testing

### 2. Perception System (`predicators/perception/`)

#### BasePerceiver Interface
```python
class BasePerceiver:
    def reset(self, env_task: EnvironmentTask) -> Task:
        """Convert initial observation to symbolic task."""
        
    def step(self, observation: Observation) -> State:
        """Convert observation to symbolic state."""
```

#### Perception Pipeline
1. **Raw Observation Processing**
   - Image processing
   - Object detection
   - Point cloud processing

2. **State Estimation**
   - Object tracking
   - Feature extraction
   - Predicate evaluation

### 3. Planning and Learning (`predicators/approaches/`)

#### Approach Types
1. **Oracle Approaches**
   - Use ground truth models
   - Useful for testing

2. **Learning Approaches**
   - NSRT learning
   - Predicate invention
   - Skill learning

#### NSRT Learning Pipeline (`nsrt_learning/`)
1. Data segmentation
2. Symbolic operator learning
3. Option learning
4. Sampler learning
5. NSRT finalization

### 4. Mock Spot Robot Environment System
For testing robot environments without hardware:

#### Components
1. **Mock Spot Robot Environment**
   ```python
   class MockEnv(BaseEnv):
       def __init__(self):
           self._transition_graph = self._load_transitions()
           self._image_database = self._load_images()
           
       def reset(self):
           return self._get_mock_observation()
           
       def step(self, action):
           return self._simulate_transition(action)
   ```

2. **Mock Perceiver**
   ```python
   class MockPerceiver(BasePerceiver):
       def reset(self, env_task):
           return self._create_symbolic_task(env_task)
           
       def step(self, observation):
           return self._process_mock_observation(observation)
   ```

3. **Data Collection Tools**
   ```python
   class DataCollector:
       def collect_state_data(self):
           """Collect images and state information."""
           
       def build_transition_graph(self):
           """Create state transition graph."""
   ```

## Running the System

### 1. Basic Usage
```bash
# Run with oracle approach
python predicators/main.py --env sokoban --approach oracle --seed 0

# Run with learning
python predicators/main.py --env cover --approach nsrt_learning --seed 0
```

### 2. Environment Setup
```bash
# Required environment variables
export PYTHONHASHSEED=0  # Required for deterministic hashing
export PYTHONPATH=/path/to/predicators  # Add to Python path
```

### 3. Development Setup
```bash
# Install development dependencies
pip install -e .[develop]

# Run tests
pytest tests/

# Run type checking
mypy .

# Run linter
pytest . --pylint
```

## Workflow Examples

### 1. Creating a New Environment

```python
# 1. Define environment
class NewEnv(BaseEnv):
    def __init__(self):
        self._setup_types()
        self._setup_predicates()
    
    def _setup_predicates(self):
        self._IsGraspable = Predicate("IsGraspable", 
                                     [self.object_type],
                                     self._IsGraspable_holds)
        # ... more predicates

# 2. Create perceiver if needed
class NewPerceiver(BasePerceiver):
    def reset(self, env_task):
        state = self._observation_to_state(env_task.init_obs)
        return Task(state, self._create_goal(env_task))

# 3. Define mock version for testing
class MockNewEnv(BaseEnv):
    def __init__(self):
        self._load_mock_data()
```

### 2. Running with Mock Environment

1. **Collect Data**

2. **Create Mock Environment**

3. **Test Pipeline**

### 3. Development Workflow

1. **Implementation**
   - Implement environment classes
   - Create perceiver if needed
   - Define predicates and types
   - Create mock version

2. **Testing**
   ```bash
   # Run unit tests
   pytest tests/envs/test_new_env.py
   
   # Run integration tests
   pytest tests/test_integration.py
   ```

3. **Debugging**
   - Use mock environment for testing
   - Add logging and visualization
   - Test predicate evaluation

