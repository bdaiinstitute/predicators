"""Task definition for mock Spot environment."""

from typing import List, Set, Dict, Optional, Tuple, Sequence
import numpy as np

from predicators.structs import Object, State, EnvironmentTask, GoalDescription
from predicators.envs.mock_spot_env import MockSpotEnv, MockSpotObservation


class MockSpotTask:
    """Task definition for mock Spot environment.
    
    This defines task-specific logic like:
    - Goal descriptions
    - Task generation
    - Converting observations to states
    """

    def __init__(self) -> None:
        self.env: Optional[MockSpotEnv] = None

    def set_environment(self, env: MockSpotEnv) -> None:
        """Set the environment for this task."""
        self.env = env

    def observation_to_state(self, obs: MockSpotObservation) -> State:
        """Convert observation to symbolic state."""
        if self.env is None:
            raise ValueError("Environment not set")
            
        # Create objects
        objects = {Object(name, self.env._movable_object_type) for name in obs.objects_in_view}
        robot = Object("robot", self.env._robot_type)
        objects.add(robot)
        
        # Create object data dictionary
        data = {}
        for obj in objects:
            if obj.type == self.env._robot_type:
                data[obj] = np.array([
                    float(obs.gripper_open),  # gripper_open_percentage
                    0.0,  # x
                    0.0,  # y 
                    0.0,  # z
                ], dtype=np.float32)
            else:
                # For mock environment, we don't track actual positions
                data[obj] = np.array([
                    0.0,  # x
                    0.0,  # y
                    0.0,  # z
                ], dtype=np.float32)
                
        return State(data)

    def generate_goal_description(self) -> GoalDescription:
        """Generate goal description for the task."""
        # This should be overridden by specific task implementations
        raise NotImplementedError("Subclasses must implement generate_goal_description")

    def get_dry_task(self, train_or_test: str, task_idx: int) -> EnvironmentTask:
        """Generate a dry run task.
        
        This should be overridden by specific task implementations to create
        tasks with appropriate initial states and goals.
        """
        raise NotImplementedError("Subclasses must implement get_dry_task") 