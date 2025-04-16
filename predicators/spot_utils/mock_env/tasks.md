# Mock Environment Tasks

This document describes how to set up and use tasks in the mock Spot environment.

## Overview

Tasks in the mock environment define:
1. Initial state configuration
2. Goal conditions
3. Available operators
4. Objects and their types
5. Predicates for state representation

## Task Structure

A basic task consists of:

```python
from typing import Set, List
from predicators.structs import Task, Type, Predicate, Object, State
from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.ground_truth_models.mock_spot_env.nsrts import MockSpotGroundTruthNSRTFactory

class MockSpotTask:
    """Base class for mock Spot tasks."""
    
    def __init__(self) -> None:
        # Get environment info
        env = MockSpotEnv()
        self.types = env.types
        self.predicates = env.predicates
        
        # Get operators from NSRTs
        self.nsrts = MockSpotGroundTruthNSRTFactory.get_nsrts(
            env_name="mock_spot",
            types=self.types,
            predicates=self.predicates
        )
        
    def get_task(self, objects: Set[Object]) -> Task:
        """Create a task instance with given objects."""
        # Create initial state
        init_atoms = {
            self.predicates["HandEmpty"]([]),
            self.predicates["InView"]([objects["cube1"]]),
            self.predicates["OnTable"]([objects["cube1"]])
        }
        init_state = State(init_atoms)
        
        # Define goal
        goal_atoms = {
            self.predicates["InContainer"]([objects["cube1"], objects["container1"]])
        }
        
        return Task(init_state, goal_atoms)
```

## Creating Custom Tasks

### 1. Define Objects and Types

```python
class PickPlaceTask(MockSpotTask):
    def __init__(self) -> None:
        super().__init__()
        
        # Define object types
        self.cube_type = self.types["cube"]
        self.container_type = self.types["container"]
        
        # Create objects
        self.objects = {
            "cube1": Object("cube1", self.cube_type),
            "container1": Object("container1", self.container_type)
        }
```

### 2. Define Initial State

```python
    def get_initial_state(self) -> State:
        """Create initial state for the task."""
        init_atoms = {
            self.predicates["HandEmpty"]([]),
            self.predicates["InView"]([self.objects["cube1"]]),
            self.predicates["OnTable"]([self.objects["cube1"]]),
            self.predicates["InView"]([self.objects["container1"]])
        }
        return State(init_atoms)
```

### 3. Define Goal Conditions

```python
    def get_goal(self) -> Set[GroundAtom]:
        """Define goal conditions."""
        return {
            self.predicates["InContainer"]([
                self.objects["cube1"], 
                self.objects["container1"]
            ])
        }
```

### 4. Create Task Instance

```python
    def get_task(self) -> Task:
        """Create complete task instance."""
        return Task(
            init_state=self.get_initial_state(),
            goal=self.get_goal()
        )
```

## Available Predicates

The mock environment provides these basic predicates:

```python
PREDICATES = {
    "HandEmpty": Predicate("HandEmpty", [], []),
    "InView": Predicate("InView", ["obj"], ["object"]),
    "InHand": Predicate("InHand", ["obj"], ["object"]),
    "OnTable": Predicate("OnTable", ["obj"], ["object"]),
    "InContainer": Predicate("InContainer", ["obj", "container"], 
                            ["object", "container"])
}
```

## Available Operators

Basic operators include:

1. Movement:
   ```python
   "MoveToReachObject": {
       "parameters": ["?obj"],
       "preconditions": ["InView(?obj)", "HandEmpty"],
       "effects_add": ["CanReach(?obj)"],
       "effects_delete": []
   }
   ```

2. Manipulation:
   ```python
   "PickObjectFromTop": {
       "parameters": ["?obj"],
       "preconditions": ["CanReach(?obj)", "HandEmpty", "OnTable(?obj)"],
       "effects_add": ["InHand(?obj)"],
       "effects_delete": ["HandEmpty", "OnTable(?obj)"]
   }
   ```

## Example Tasks

### 1. Pick and Place Task

```python
class PickPlaceTask(MockSpotTask):
    """Task to pick an object and place it in a container."""
    
    def __init__(self) -> None:
        super().__init__()
        self.objects = {
            "cube1": Object("cube1", self.types["cube"]),
            "container1": Object("container1", self.types["container"])
        }
        
    def get_task(self) -> Task:
        # Initial state
        init_atoms = {
            self.predicates["HandEmpty"]([]),
            self.predicates["InView"]([self.objects["cube1"]]),
            self.predicates["OnTable"]([self.objects["cube1"]]),
            self.predicates["InView"]([self.objects["container1"]])
        }
        init_state = State(init_atoms)
        
        # Goal
        goal_atoms = {
            self.predicates["InContainer"]([
                self.objects["cube1"], 
                self.objects["container1"]
            ])
        }
        
        return Task(init_state, goal_atoms)
```

### 2. Multi-Object Task

```python
class MultiObjectTask(MockSpotTask):
    """Task to manipulate multiple objects."""
    
    def __init__(self) -> None:
        super().__init__()
        self.objects = {
            "cube1": Object("cube1", self.types["cube"]),
            "cube2": Object("cube2", self.types["cube"]),
            "container1": Object("container1", self.types["container"])
        }
        
    def get_task(self) -> Task:
        # Initial state
        init_atoms = {
            self.predicates["HandEmpty"]([]),
            self.predicates["InView"]([self.objects["cube1"]]),
            self.predicates["OnTable"]([self.objects["cube1"]]),
            self.predicates["InView"]([self.objects["cube2"]]),
            self.predicates["OnTable"]([self.objects["cube2"]]),
            self.predicates["InView"]([self.objects["container1"]])
        }
        init_state = State(init_atoms)
        
        # Goal: both cubes in container
        goal_atoms = {
            self.predicates["InContainer"]([
                self.objects["cube1"], 
                self.objects["container1"]
            ]),
            self.predicates["InContainer"]([
                self.objects["cube2"], 
                self.objects["container1"]
            ])
        }
        
        return Task(init_state, goal_atoms)
```

## Using Tasks with Environment

```python
# Create environment and task
env = MockSpotEnv(data_dir="path/to/data_dir")
task = PickPlaceTask()

# Initialize environment with task
env.set_task(task)

# Get initial observation
obs = env.reset()

# Take actions
action = env.action_space.sample()  # Replace with your action selection
next_obs, reward, done, info = env.step(action)
```

## Best Practices

1. Task Design:
   - Keep tasks modular and focused
   - Reuse common predicates and operators
   - Document task requirements clearly

2. State Management:
   - Verify state consistency
   - Handle edge cases
   - Include error checking

3. Testing:
   - Create unit tests for tasks
   - Verify goal conditions
   - Test with different object configurations 