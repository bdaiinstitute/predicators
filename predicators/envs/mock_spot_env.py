"""Mock environment for Spot robot."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Sequence
import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.structs import Action, State, Object, Type, EnvironmentTask, Video, Image
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.structs import LiftedAtom, STRIPSOperator, Variable, Predicate
from bosdyn.client import math_helpers


def _dummy_classifier(state: State, objects: Sequence[Object]) -> bool:
    """Dummy classifier that always returns True. Used for mock environment."""
    return True


@dataclass
class MockSpotObservation:
    """Observation for mock Spot environment."""
    rgbd: Optional[RGBDImageWithContext]  # RGB-D image with context
    gripper_open: bool  # Whether the gripper is open
    objects_in_view: Set[str]  # Names of objects visible in the image
    objects_in_hand: Set[str]  # Names of objects currently held
    state_id: str  # Unique ID for this observation's latent state


class MockSpotEnv(BaseEnv):
    """Mock environment for Spot robot.
    
    This environment is a POMDP where:
    - States are latent (we don't know actual poses)
    - Observations are RGB-D images + gripper state + object detections
    - Actions can succeed or fail based on available images
    """

    def __init__(self, data_dir: str = "spot_mock_data") -> None:
        super().__init__()
        
        # Create data directories
        self._data_dir = Path(data_dir)
        self._images_dir = self._data_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment state
        self._current_state_id: Optional[str] = None
        self._gripper_open: bool = True
        self._objects_in_hand: Set[str] = set()
        
        # Load or initialize transition graph
        self._transitions: Dict[str, Dict[str, str]] = {}  # state_id -> {action -> next_state_id}
        self._observations: Dict[str, MockSpotObservation] = {}  # state_id -> observation
        self._load_graph_data()

        # Create types
        self._robot_type = Type("robot", ["x", "y", "z"])
        self._base_object_type = Type("base_object", ["x", "y", "z"])
        self._movable_object_type = Type("movable_object", ["x", "y", "z"], parent=self._base_object_type)
        self._container_type = Type("container", ["x", "y", "z"], parent=self._movable_object_type)
        self._immovable_object_type = Type("immovable_object", ["x", "y", "z"], parent=self._base_object_type)

        # Create predicates
        self._NEq = Predicate("NEq", [self._base_object_type, self._base_object_type], _dummy_classifier)
        self._On = Predicate("On", [self._movable_object_type, self._base_object_type], _dummy_classifier)
        self._TopAbove = Predicate("TopAbove", [self._base_object_type, self._base_object_type], _dummy_classifier)
        self._Inside = Predicate("Inside", [self._movable_object_type, self._container_type], _dummy_classifier)
        self._NotInsideAnyContainer = Predicate("NotInsideAnyContainer", [self._movable_object_type], _dummy_classifier)
        self._FitsInXY = Predicate("FitsInXY", [self._movable_object_type, self._base_object_type], _dummy_classifier)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type], _dummy_classifier)
        self._Holding = Predicate("Holding", [self._robot_type, self._movable_object_type], _dummy_classifier)
        self._NotHolding = Predicate("NotHolding", [self._robot_type, self._base_object_type], _dummy_classifier)
        self._InHandView = Predicate("InHandView", [self._robot_type, self._movable_object_type], _dummy_classifier)
        self._InView = Predicate("InView", [self._robot_type, self._movable_object_type], _dummy_classifier)
        self._Reachable = Predicate("Reachable", [self._robot_type, self._base_object_type], _dummy_classifier)
        self._Blocking = Predicate("Blocking", [self._base_object_type, self._base_object_type], _dummy_classifier)
        self._NotBlocked = Predicate("NotBlocked", [self._base_object_type], _dummy_classifier)
        self._ContainerReadyForSweeping = Predicate("ContainerReadyForSweeping", [self._container_type, self._immovable_object_type], _dummy_classifier)
        self._IsPlaceable = Predicate("IsPlaceable", [self._movable_object_type], _dummy_classifier)
        self._IsNotPlaceable = Predicate("IsNotPlaceable", [self._movable_object_type], _dummy_classifier)
        self._IsSweeper = Predicate("IsSweeper", [self._movable_object_type], _dummy_classifier)
        self._HasFlatTopSurface = Predicate("HasFlatTopSurface", [self._immovable_object_type], _dummy_classifier)
        self._RobotReadyForSweeping = Predicate("RobotReadyForSweeping", [self._robot_type, self._movable_object_type], _dummy_classifier)
        self._ContainingWaterUnknown = Predicate("ContainingWaterUnknown", [self._container_type], _dummy_classifier)
        self._ContainingWaterKnown = Predicate("ContainingWaterKnown", [self._container_type], _dummy_classifier)
        self._ContainingWater = Predicate("ContainingWater", [self._container_type], _dummy_classifier)
        self._NotContainingWater = Predicate("NotContainingWater", [self._container_type], _dummy_classifier)
        self._InHandViewFromTop = Predicate("InHandViewFromTop", [self._robot_type, self._movable_object_type], _dummy_classifier)

        # Create operators
        self._operators = list(self._create_operators())

    def _create_operators(self) -> Iterator[STRIPSOperator]:
        """Create operators for the mock Spot environment."""
        # PickObjectFromTop
        robot = Variable("?robot", self._robot_type)
        obj = Variable("?obj", self._movable_object_type)
        surface = Variable("?surface", self._immovable_object_type)
        parameters = [robot, obj, surface]
        preconds = {
            LiftedAtom(self._On, [obj, surface]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._InHandView, [robot, obj]),
            LiftedAtom(self._NotInsideAnyContainer, [obj]),
            LiftedAtom(self._IsPlaceable, [obj]),
            LiftedAtom(self._HasFlatTopSurface, [surface]),
        }
        add_effs = {
            LiftedAtom(self._Holding, [robot, obj]),
        }
        del_effs = {
            LiftedAtom(self._On, [obj, surface]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._InHandView, [robot, obj]),
            LiftedAtom(self._NotHolding, [robot, obj]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PickObjectFromTop", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # PickObjectToDrag
        robot = Variable("?robot", self._robot_type)
        obj = Variable("?obj", self._movable_object_type)
        parameters = [robot, obj]
        preconds = {
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._InHandView, [robot, obj]),
            LiftedAtom(self._IsNotPlaceable, [obj]),
        }
        add_effs = {
            LiftedAtom(self._Holding, [robot, obj]),
        }
        del_effs = {
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._InHandView, [robot, obj]),
            LiftedAtom(self._NotHolding, [robot, obj]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PickObjectToDrag", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # PlaceObjectOnTop
        robot = Variable("?robot", self._robot_type)
        held = Variable("?held", self._movable_object_type)
        surface = Variable("?surface", self._immovable_object_type)
        parameters = [robot, held, surface]
        preconds = {
            LiftedAtom(self._Holding, [robot, held]),
            LiftedAtom(self._Reachable, [robot, surface]),
            LiftedAtom(self._NEq, [held, surface]),
            LiftedAtom(self._IsPlaceable, [held]),
            LiftedAtom(self._HasFlatTopSurface, [surface]),
            LiftedAtom(self._FitsInXY, [held, surface]),
        }
        add_effs = {
            LiftedAtom(self._On, [held, surface]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._NotHolding, [robot, held]),
        }
        del_effs = {
            LiftedAtom(self._Holding, [robot, held]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PlaceObjectOnTop", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # DropNotPlaceableObject
        robot = Variable("?robot", self._robot_type)
        held = Variable("?held", self._movable_object_type)
        parameters = [robot, held]
        preconds = {
            LiftedAtom(self._Holding, [robot, held]),
            LiftedAtom(self._IsNotPlaceable, [held]),
        }
        add_effs = {
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._NotHolding, [robot, held]),
        }
        del_effs = {
            LiftedAtom(self._Holding, [robot, held]),
        }
        ignore_effs = set()
        yield STRIPSOperator("DropNotPlaceableObject", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

        # DropObjectInside
        robot = Variable("?robot", self._robot_type)
        held = Variable("?held", self._movable_object_type)
        container = Variable("?container", self._container_type)
        parameters = [robot, held, container]
        preconds = {
            LiftedAtom(self._Holding, [robot, held]),
            LiftedAtom(self._Reachable, [robot, container]),
            LiftedAtom(self._IsPlaceable, [held]),
            LiftedAtom(self._FitsInXY, [held, container]),
        }
        add_effs = {
            LiftedAtom(self._Inside, [held, container]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._NotHolding, [robot, held]),
        }
        del_effs = {
            LiftedAtom(self._Holding, [robot, held]),
            LiftedAtom(self._NotInsideAnyContainer, [held])
        }
        ignore_effs = set()
        yield STRIPSOperator("DropObjectInside", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # DropObjectInsideContainerOnTop
        robot = Variable("?robot", self._robot_type)
        held = Variable("?held", self._movable_object_type)
        container = Variable("?container", self._container_type)
        surface = Variable("?surface", self._immovable_object_type)
        parameters = [robot, held, container, surface]
        preconds = {
            LiftedAtom(self._Holding, [robot, held]),
            LiftedAtom(self._InView, [robot, container]),
            LiftedAtom(self._On, [container, surface]),
            LiftedAtom(self._IsPlaceable, [held]),
        }
        add_effs = {
            LiftedAtom(self._Inside, [held, container]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._On, [held, surface]),
            LiftedAtom(self._NotHolding, [robot, held]),
            LiftedAtom(self._FitsInXY, [held, container]),
        }
        del_effs = {
            LiftedAtom(self._Holding, [robot, held]),
            LiftedAtom(self._NotInsideAnyContainer, [held])
        }
        ignore_effs = set()
        yield STRIPSOperator("DropObjectInsideContainerOnTop", parameters,
                            preconds, add_effs, del_effs, ignore_effs)

        # DragToUnblockObject
        robot = Variable("?robot", self._robot_type)
        blocked = Variable("?blocked", self._base_object_type)
        blocker = Variable("?blocker", self._movable_object_type)
        parameters = [robot, blocker, blocked]
        preconds = {
            LiftedAtom(self._Blocking, [blocker, blocked]),
            LiftedAtom(self._Holding, [robot, blocker]),
        }
        add_effs = {
            LiftedAtom(self._NotBlocked, [blocked]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._NotHolding, [robot, blocker]),
        }
        del_effs = {
            LiftedAtom(self._Holding, [robot, blocker]),
        }
        ignore_effs = {self._InHandView, self._Reachable, self._RobotReadyForSweeping, self._Blocking}
        yield STRIPSOperator("DragToUnblockObject", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # DragToBlockObject
        robot = Variable("?robot", self._robot_type)
        blocked = Variable("?blocked", self._base_object_type)
        blocker = Variable("?blocker", self._movable_object_type)
        parameters = [robot, blocker, blocked]
        preconds = {
            LiftedAtom(self._NotBlocked, [blocked]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._Holding, [robot, blocker]),
        }
        add_effs = {
            LiftedAtom(self._Blocking, [blocker, blocked]),
            LiftedAtom(self._NotHolding, [robot, blocker]),
        }
        del_effs = {
            LiftedAtom(self._Holding, [robot, blocker]),
        }
        ignore_effs = {self._InHandView, self._Reachable, self._RobotReadyForSweeping, self._Blocking}
        yield STRIPSOperator("DragToBlockObject", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # SweepIntoContainer
        robot = Variable("?robot", self._robot_type)
        sweeper = Variable("?sweeper", self._movable_object_type)
        target = Variable("?target", self._movable_object_type)
        surface = Variable("?surface", self._immovable_object_type)
        container = Variable("?container", self._container_type)
        parameters = [robot, sweeper, target, surface, container]
        preconds = {
            LiftedAtom(self._NotBlocked, [target]),
            LiftedAtom(self._Holding, [robot, sweeper]),
            LiftedAtom(self._On, [target, surface]),
            LiftedAtom(self._RobotReadyForSweeping, [robot, target]),
            LiftedAtom(self._ContainerReadyForSweeping, [container, surface]),
            LiftedAtom(self._IsPlaceable, [target]),
            LiftedAtom(self._HasFlatTopSurface, [surface]),
            LiftedAtom(self._TopAbove, [surface, container]),
            LiftedAtom(self._IsSweeper, [sweeper]),
            LiftedAtom(self._FitsInXY, [target, container]),
        }
        add_effs = {
            LiftedAtom(self._Inside, [target, container]),
        }
        del_effs = {
            LiftedAtom(self._On, [target, surface]),
            LiftedAtom(self._ContainerReadyForSweeping, [container, surface]),
            LiftedAtom(self._RobotReadyForSweeping, [robot, target]),
            LiftedAtom(self._Reachable, [robot, target]),
            LiftedAtom(self._NotInsideAnyContainer, [target])
        }
        ignore_effs = set()
        yield STRIPSOperator("SweepIntoContainer", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # PrepareContainerForSweeping
        robot = Variable("?robot", self._robot_type)
        container = Variable("?container", self._container_type)
        target = Variable("?target", self._movable_object_type)
        surface = Variable("?surface", self._immovable_object_type)
        parameters = [robot, container, target, surface]
        preconds = {
            LiftedAtom(self._Holding, [robot, container]),
            LiftedAtom(self._On, [target, surface]),
            LiftedAtom(self._TopAbove, [surface, container]),
            LiftedAtom(self._NEq, [surface, container]),
        }
        add_effs = {
            LiftedAtom(self._ContainerReadyForSweeping, [container, surface]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._NotHolding, [robot, container]),
        }
        del_effs = {
            LiftedAtom(self._Holding, [robot, container]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PrepareContainerForSweeping", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

        # PickAndDumpContainer
        robot = Variable("?robot", self._robot_type)
        container = Variable("?container", self._container_type)
        surface = Variable("?surface", self._base_object_type)
        obj_inside = Variable("?object", self._movable_object_type)
        parameters = [robot, container, surface, obj_inside]
        preconds = {
            LiftedAtom(self._On, [container, surface]),
            LiftedAtom(self._Inside, [obj_inside, container]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._InHandView, [robot, container])
        }
        add_effs = {LiftedAtom(self._NotInsideAnyContainer, [obj_inside])}
        del_effs = {
            LiftedAtom(self._Inside, [obj_inside, container]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PickAndDumpContainer", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

        # PickAndDumpTwoFromContainer
        robot = Variable("?robot", self._robot_type)
        container = Variable("?container", self._container_type)
        surface = Variable("?surface", self._base_object_type)
        obj_inside1 = Variable("?object1", self._movable_object_type)
        obj_inside2 = Variable("?object2", self._movable_object_type)
        parameters = [robot, container, surface, obj_inside1, obj_inside2]
        preconds = {
            LiftedAtom(self._On, [container, surface]),
            LiftedAtom(self._Inside, [obj_inside1, container]),
            LiftedAtom(self._Inside, [obj_inside2, container]),
            LiftedAtom(self._HandEmpty, [robot]),
            LiftedAtom(self._InHandView, [robot, container])
        }
        add_effs = {
            LiftedAtom(self._NotInsideAnyContainer, [obj_inside1]),
            LiftedAtom(self._NotInsideAnyContainer, [obj_inside2]),
        }
        del_effs = {
            LiftedAtom(self._Inside, [obj_inside1, container]),
            LiftedAtom(self._Inside, [obj_inside2, container]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PickAndDumpTwoFromContainer", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

    @property
    def action_space(self) -> Box:
        """Return action space."""
        return Box(low=0, high=1, shape=(1,))

    def simulate(self, state: State, action: Action) -> State:
        """Simulate a state transition."""
        raise NotImplementedError("Mock environment does not support simulation")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Generate training tasks."""
        return []

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Generate test tasks."""
        return []

    @property
    def predicates(self) -> Set[Any]:
        """Return predicates."""
        return {
            self._NotBlocked,
            self._NotHolding,
            self._Reachable,
            self._HandEmpty,
            self._InHandView,
            self._InHandViewFromTop,
            self._On,
            self._ContainingWaterUnknown,
            self._ContainingWaterKnown,
            self._ContainingWater,
            self._NotContainingWater,
        }

    @property
    def goal_predicates(self) -> Set[Any]:
        """Return goal predicates."""
        return {
            self._Reachable,
            self._InHandView,
            self._InHandViewFromTop,
            self._ContainingWaterKnown,
            self._ContainingWater,
            self._NotContainingWater,
        }

    @property
    def types(self) -> Set[Type]:
        """Return types."""
        return {
            self._robot_type,
            self._base_object_type,
            self._movable_object_type,
            self._container_type,
            self._immovable_object_type,
        }

    @property
    def percept_predicates(self) -> Set[Any]:
        """Return percept predicates."""
        return set()

    def _load_graph_data(self) -> None:
        """Load transition graph and observations from disk."""
        graph_file = self._data_dir / "graph.json"
        if not graph_file.exists():
            return
            
        import json
        with open(graph_file) as f:
            data = json.load(f)
            
        # Load transitions
        self._transitions = data["transitions"]
        
        # Load observations
        for state_id, obs_data in data["observations"].items():
            # Load RGBD image if it exists
            rgbd = None
            rgb_path = self._images_dir / f"rgb_{state_id}.npy"
            depth_path = self._images_dir / f"depth_{state_id}.npy"
            
            if rgb_path.exists() and depth_path.exists():
                rgb = np.load(rgb_path)
                depth = np.load(depth_path)
                # Create RGBDImageWithContext with minimal context
                rgbd = RGBDImageWithContext(
                    rgb=rgb,
                    depth=depth,
                    image_rot=0.0,  # No rotation needed for mock env
                    camera_name="mock_camera",
                    world_tform_camera=math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat()),  # Identity pose
                    depth_scale=1.0,
                    transforms_snapshot=None,  # Not tracking transforms
                    frame_name_image_sensor="mock_camera",
                    camera_model=None  # Not tracking camera model
                )
            
            self._observations[state_id] = MockSpotObservation(
                rgbd=rgbd,
                gripper_open=obs_data["gripper_open"],
                objects_in_view=set(obs_data["objects_in_view"]),
                objects_in_hand=set(obs_data["objects_in_hand"]),
                state_id=state_id
            )

    def _save_graph_data(self) -> None:
        """Save transition graph and observations to disk."""
        # Convert data to serializable format
        observations_data = {}
        for state_id, obs in self._observations.items():
            # Save RGBD images separately
            if obs.rgbd is not None:
                np.save(self._images_dir / f"rgb_{state_id}.npy", obs.rgbd.rgb)
                np.save(self._images_dir / f"depth_{state_id}.npy", obs.rgbd.depth)
            
            observations_data[state_id] = {
                "gripper_open": obs.gripper_open,
                "objects_in_view": list(obs.objects_in_view),
                "objects_in_hand": list(obs.objects_in_hand)
            }
        
        data = {
            "transitions": self._transitions,
            "observations": observations_data
        }
        
        # Save to file
        import json
        with open(self._data_dir / "graph.json", "w") as f:
            json.dump(data, f, indent=2)

    def add_state(self, 
                 rgbd: Optional[RGBDImageWithContext] = None,
                 gripper_open: bool = True,
                 objects_in_view: Optional[Set[str]] = None,
                 objects_in_hand: Optional[Set[str]] = None) -> str:
        """Add a new state to the environment.
        
        Returns:
            state_id: Unique ID for the new state
        """
        # Generate unique state ID
        state_id = str(len(self._observations))
        
        # Create observation
        self._observations[state_id] = MockSpotObservation(
            rgbd=rgbd,
            gripper_open=gripper_open,
            objects_in_view=objects_in_view or set(),
            objects_in_hand=objects_in_hand or set(),
            state_id=state_id
        )
        
        # Initialize empty transitions
        if state_id not in self._transitions:
            self._transitions[state_id] = {}
            
        # Save updated data
        self._save_graph_data()
        
        return state_id

    def add_transition(self, 
                      from_state_id: str,
                      action_name: str,
                      to_state_id: str) -> None:
        """Add a transition between states."""
        if from_state_id not in self._observations:
            raise ValueError(f"Unknown state ID: {from_state_id}")
        if to_state_id not in self._observations:
            raise ValueError(f"Unknown state ID: {to_state_id}")
            
        # Verify action is a valid operator
        if not any(op.name == action_name for op in self._operators):
            raise ValueError(f"Unknown operator: {action_name}")
            
        self._transitions[from_state_id][action_name] = to_state_id
        self._save_graph_data()

    def get_observation(self) -> MockSpotObservation:
        """Get current observation."""
        if self._current_state_id is None:
            raise ValueError("Environment not initialized")
        return self._observations[self._current_state_id]

    def reset(self, train_or_test: str, task_idx: int) -> MockSpotObservation:
        """Reset environment to initial state."""
        if not self._observations:
            raise ValueError("No states added to environment")
            
        # For now, just pick the first state as initial
        state_id = list(self._observations.keys())[0]
        self._current_state_id = state_id
        obs = self._observations[state_id]
        self._gripper_open = obs.gripper_open
        self._objects_in_hand = obs.objects_in_hand.copy()
        
        return obs

    def step(self, action: Action) -> Tuple[MockSpotObservation, float, bool]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Always 0 (no reward function defined)
            done: Always False (no termination condition)
        """
        if self._current_state_id is None:
            raise ValueError("Environment not initialized")
            
        # Get action name from extra info
        if action.extra_info is None:
            raise ValueError("Action must have extra_info with 'name' field")
        action_name = action.extra_info.get("name")
        if action_name is None:
            raise ValueError("Action extra_info must have 'name' field")
            
        # Check if transition exists
        if action_name not in self._transitions[self._current_state_id]:
            # No transition available - action fails
            return self.get_observation(), 0.0, False
            
        # Update state
        self._current_state_id = self._transitions[self._current_state_id][action_name]
        obs = self._observations[self._current_state_id]
        self._gripper_open = obs.gripper_open
        self._objects_in_hand = obs.objects_in_hand.copy()
        
        return obs, 0.0, False

    def render_state_plt(self, *args, **kwargs) -> None:
        """Render the current state using matplotlib."""
        if self._current_state_id is None:
            return
            
        obs = self._observations[self._current_state_id]
        if obs.rgbd is not None:
            import matplotlib.pyplot as plt
            plt.imshow(obs.rgbd.rgb)
            plt.axis('off')
            plt.show()

    def render_state(self, *args, **kwargs) -> Video:
        """Render the current state."""
        if self._current_state_id is None:
            return []
            
        obs = self._observations[self._current_state_id]
        if obs.rgbd is not None:
            return [obs.rgbd.rgb]
        return []

    @classmethod
    def get_name(cls) -> str:
        """Return environment name."""
        return "mock_spot"
