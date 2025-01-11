"""Task definition for mock Spot environment."""

from typing import List, Set, Dict, Optional, Tuple, Sequence, cast
import numpy as np

from predicators.structs import Object, State, EnvironmentTask, GoalDescription, Type, Predicate, GroundAtom
from predicators.envs.mock_spot_env import MockSpotEnv, MockSpotObservation
from predicators.ground_truth_models.mock_spot_env.nsrts import MockSpotGroundTruthNSRTFactory


class MockSpotTask(EnvironmentTask):
    """Task definition for mock Spot environment.
    
    A task consists of:
    - Initial state configuration
    - Goal conditions
    - Available operators and predicates
    - Objects and their types
    """

    def __init__(self, env: MockSpotEnv, goal_description: Optional[GoalDescription] = None) -> None:
        """Initialize the task with a mock Spot environment."""
        super().__init__(env, goal_description)
        self._env = env
        self._objects: Set[Object] = set()
        self._init_state: Optional[State] = None
        self._goal = goal_description
        self._initialize_task()

    def _initialize_task(self) -> None:
        """Initialize task components."""
        # Create basic objects (robot always exists)
        types = {t.name: t for t in self._env.types}
        robot = Object("robot", types["robot"])
        self._objects.add(robot)

        # Get NSRTs for operators
        factory = MockSpotGroundTruthNSRTFactory()
        predicates = {p.name: p for p in self._env.predicates}
        self._nsrts = factory.get_nsrts(
            self._env.get_name(),
            types,
            predicates,
            {}  # Options not needed for task definition
        )

    def get_objects(self) -> Set[Object]:
        """Get all objects in the task."""
        return self._objects

    def update_objects(self, objects_in_view: Set[str]) -> None:
        """Update task objects based on perception."""
        # Add new objects with appropriate types
        types = {t.name: t for t in self._env.types}
        for obj_name in objects_in_view:
            if obj_name == "robot":
                continue
            # Determine object type based on name
            if "table" in obj_name:
                type_name = "immovable_object"
            elif "container" in obj_name or "cup" in obj_name:
                type_name = "container"
            else:
                type_name = "movable_object"
            obj = Object(obj_name, types[type_name])
            self._objects.add(obj)

    def get_init_state(self) -> State:
        """Get initial state."""
        if self._init_state is None:
            # Create initial state from current environment state
            obs = self._env.get_observation()  # Using get_observation instead of get_current_observation
            self._init_state = self.observation_to_state(obs)
        return self._init_state

    def observation_to_state(self, obs: MockSpotObservation) -> State:
        """Convert observation to state."""
        # Update objects based on observation
        self.update_objects(obs.objects_in_view)

        # Create state atoms based on observation
        atoms: Set[GroundAtom] = set()
        predicates = {p.name: p for p in self._env.predicates}
        robot = next(obj for obj in self._objects if obj.type.name == "robot")

        # Add HandEmpty/Holding predicates
        if not obs.objects_in_hand:
            atoms.add(GroundAtom(predicates["HandEmpty"], [robot]))
        else:
            for obj_name in obs.objects_in_hand:
                obj = next(o for o in self._objects if o.name == obj_name)
                atoms.add(GroundAtom(predicates["Holding"], [robot, obj]))

        # Add InView predicates
        for obj_name in obs.objects_in_view:
            obj = next(o for o in self._objects if o.name == obj_name)
            atoms.add(GroundAtom(predicates["InView"], [robot, obj]))

        # Create state with atoms in simulator_state
        # Keep a dummy state dict so we know what objects are in the state
        dummy_state_dict = {o: np.zeros(0, dtype=np.float32) for o in self._objects}
        
        # Create empty VLM atom dict since we don't use VLM in mock env
        vlm_atom_dict = {}
        
        return State(
            data=dummy_state_dict,
            simulator_state=atoms,
            vlm_atom_dict=vlm_atom_dict
        )

    def get_goal(self) -> GoalDescription:
        """Get goal description."""
        if self._goal is None:
            # Create a default goal (can be overridden)
            predicates = {p.name: p for p in self._env.predicates}
            robot = next(obj for obj in self._objects if obj.type.name == "robot")
            goal_atoms = {GroundAtom(predicates["HandEmpty"], [robot])}
            self._goal = goal_atoms
        return self._goal

    def set_goal(self, goal: GoalDescription) -> None:
        """Set a new goal for the task."""
        self._goal = goal 