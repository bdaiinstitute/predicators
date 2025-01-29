"""Vision-language model (VLM) based policy-guided bilevel planning approach.

This approach combines the vision-language capabilities of VLMOpenLoopApproach
with the bilevel planning strategy of LLMBilevelPlanningApproach.

Example command line:
    python predicators/main.py --approach vlm_bilevel_planning --seed 0 \
        --env mock_spot_pick_place_two_cup \
        --num_train_tasks 0 \
        --num_test_tasks 1 \
        --bilevel_plan_without_sim True \
        --execution_monitor expected_atoms \
        --vlm_model_name gpt-4o \
        --vlm_temperature 0.2
"""
from __future__ import annotations

import logging
import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Callable, Sequence

import numpy as np

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.vlm_open_loop_approach import VLMOpenLoopApproach
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, ParameterizedOption, \
    State, Task, _GroundNSRT, _Option
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts


class VLMBilevelPlanningApproach(VLMOpenLoopApproach):
    """VLMBilevelPlanningApproach definition.
    
    This approach uses a VLM to generate option plans that are used to guide
    bilevel planning. The VLM's suggestions are used to initialize the search
    queue in A*, similar to how LLMBilevelPlanningApproach works.
    """

    @classmethod
    def get_name(cls) -> str:
        # NOTE: this one is not using VLM
        return "vlm_oracle_bilevel_planning_deprecated"

    def _get_current_nsrts(self) -> Set[utils.NSRT]:
        """Get the current set of NSRTs."""
        # Get NSRTs from the environment's ground truth NSRT factory
        return get_gt_nsrts(CFG.env, self._initial_predicates, self._initial_options)

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Override solve to use task-level planning with VLM guidance."""
        try:
            # Get VLM-based plan
            option_plan = self._query_vlm_for_option_plan(task)
            if not option_plan:
                raise ApproachFailure("VLM failed to produce valid plan")

            # Convert option plan to ground NSRTs
            atoms = utils.abstract(task.init, self._initial_predicates)
            objects = set(task.init)
            option_plan_tuples = [(o.parent, o.objects) for o in option_plan]
            ground_nsrt_plan = self._option_plan_to_nsrt_plan(
                option_plan_tuples, atoms, objects, task.goal)
            if ground_nsrt_plan is None:
                raise ApproachFailure("Failed to convert option plan to NSRT plan")

            # Create a policy that executes the plan
            if CFG.bilevel_plan_without_sim:
                # Skip low-level planning, just execute the high-level plan
                policy = utils.nsrt_plan_to_greedy_policy(ground_nsrt_plan, task.goal, self._rng)
            else:
                # Use the plan to guide low-level planning
                abstract_policy = lambda a, o, g: ground_nsrt_plan[0] if ground_nsrt_plan else None
                max_policy_guided_rollout = CFG.horizon
                nsrts = self._get_current_nsrts()
                preds = self._get_current_predicates()
                options, _, metrics = self._run_sesame_plan(
                    task,
                    nsrts,
                    preds,
                    timeout,
                    CFG.seed,
                    abstract_policy=abstract_policy,
                    max_policy_guided_rollout=max_policy_guided_rollout)
                self._save_metrics(metrics, nsrts, preds)
                policy = utils.option_plan_to_policy(options)

            def _policy(s: State) -> Action:
                try:
                    return policy(s)
                except utils.OptionExecutionFailure as e:
                    raise ApproachFailure(e.args[0], e.info)

            return _policy

        except Exception as e:
            logging.exception("VLM planning failed:")
            raise ApproachFailure(f"VLM planning failed. Reason: {e}")

    def _option_plan_to_nsrt_plan(
            self, option_plan: List[Tuple[ParameterizedOption,
                                        Sequence[Object]]],
            atoms: Set[GroundAtom], objects: Set[Object],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        """Convert an option plan to an NSRT plan.
        
        Like LLMBilevelPlanningApproach, we assume a one-to-one mapping between
        NSRTs and options, which holds in PDDL-only environments.
        """
        nsrts = self._get_current_nsrts()
        options = self._initial_options
        assert all(sum(n.option == o for n in nsrts) == 1 for o in options)
        option_to_nsrt = {n.option: n for n in nsrts}
        return [option_to_nsrt[o].ground(objs) for (o, objs) in option_plan] 