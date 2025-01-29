"""An execution monitor that leverages knowledge of the high-level plan to only
suggest replanning when the expected atoms check is not met."""

import logging

from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.settings import CFG
from predicators.structs import State
from predicators.structs import GroundAtom, Object
from predicators.spot_utils.utils import _container_type


class ExpectedAtomsExecutionMonitor(BaseExecutionMonitor):
    """An execution monitor that only suggests replanning when we're doing
    bilevel planning and the expected atoms check fails."""

    @classmethod
    def get_name(cls) -> str:
        return "expected_atoms"

    def step(self, state: State) -> bool:
        # This monitor only makes sense to use with an oracle
        # bilevel planning approach.
        assert "oracle" in CFG.approach or "active_sampler" in CFG.approach \
            or "maple_q" in CFG.approach
        # If the approach info is empty, don't replan.
        if not self._approach_info:  # pragma: no cover
            return False
        next_expected_atoms = self._approach_info[0]
        assert isinstance(next_expected_atoms, set)
        self._curr_plan_timestep += 1
        # If the expected atoms are a subset of the current atoms, then
        # we don't have to replan.
        unsat_atoms = {a for a in next_expected_atoms if not a.holds(state)}

        # NOTE: this is to update goal and objects
        # Check goal
        assert self.perceiver is not None and self.env_task is not None
        new_goal = self.perceiver._create_goal(state, self.env_task.goal_description)
        if new_goal != self._curr_goal:
            logging.info(
                "Expected atoms execution monitor triggered replanning "
                "because the goal has changed.")
            logging.info(f"Old goal: {self._curr_goal}")
            logging.info(f"New goal: {new_goal}")
            self._curr_goal = new_goal
            # Map th objects in the new goal to the objects in the state
            import jellyfish
            # NOTE: this is goal atoms
            def map_goal_to_state(goal_predicates, state_data):
                goal_to_state_mapping = {}
                state_usage_count = {state_obj: 0 for state_obj in state_data.keys()}
                mapped_objects = set()
                for pred in goal_predicates:
                    for goal_obj in pred.objects:
                        if goal_obj in mapped_objects:
                            continue
                        goal_obj_name = str(goal_obj)
                        closest_state_obj = None
                        min_distance = float('inf')
                        for state_obj in state_data.keys():
                            state_obj_name = str(state_obj)
                            distance = jellyfish.levenshtein_distance(goal_obj_name, state_obj_name)
                            if distance < min_distance:
                                min_distance = distance
                                closest_state_obj = state_obj
                        if state_usage_count[closest_state_obj] > 0:
                            virtual_obj = Object(f"{closest_state_obj.name}{state_usage_count[closest_state_obj]}", closest_state_obj.type)
                            goal_to_state_mapping[goal_obj] = virtual_obj
                        else:
                            goal_to_state_mapping[goal_obj] = closest_state_obj
                        state_usage_count[closest_state_obj] += 1
                        mapped_objects.add(goal_obj)
                return goal_to_state_mapping
            mapping = map_goal_to_state(self._curr_goal, state.data)
            new_goal = set()
            for pred in self._curr_goal:
                new_goal.add(GroundAtom(pred.predicate, [mapping[obj] for obj in pred.objects]))
            self._curr_goal = new_goal
            self.perceiver._novel_objects_with_query.add((Object("cup1", _container_type), "red toy cup/red cup/small red cylinder/red plastic circle"))
            #self.perceiver._novel_objects_with_query.add((Object("cup2", _container_type), "blue cup/blue cylinder/blue plastic cup/blue toy cup"))
            # self.perceiver._novel_objects_with_query.add((Object("cup3", _container_type), "green cup/green mug/uncovered green cup"))
            if "cup1" in str(self._curr_goal):
               return True
        if not unsat_atoms:
            return False
        logging.info(
            "Expected atoms execution monitor triggered replanning "
            f"because of these atoms: {unsat_atoms}")  # pragma: no cover
        return True  # pragma: no cover
