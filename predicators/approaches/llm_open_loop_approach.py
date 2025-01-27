"""Open-loop large language model (LLM) planner approach.

Example command:
    export OPENAI_API_KEY=<your API key>
    python predicators/main.py --approach llm_open_loop --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Example using Gemini:
    export GOOGLE_API_KEY=<your API key>
    python predicators/main.py --approach llm_open_loop --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug --llm_model_name gemini-1.5-flash
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set
import logging
import numpy as np

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.bilevel_planning_approach import BilevelPlanningApproach
from predicators.ground_truth_models import get_gt_nsrts
from predicators.settings import CFG
from predicators.structs import Action, Box, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _Option
from predicators.pretrained_model_interface import create_llm_by_name


class LLMOpenLoopApproach(BilevelPlanningApproach):
    """LLMOpenLoopApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the LLM and base prompt
        self._llm = create_llm_by_name(CFG.llm_model_name)
        # Load the base prompt from file
        filepath_to_llm_prompt = utils.get_path_to_predicators_root() + \
            "/predicators/approaches/llm_planning_prompts/zero_shot.txt"
        with open(filepath_to_llm_prompt, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()

    @classmethod
    def get_name(cls) -> str:
        return "llm_open_loop"

    @property
    def is_learning_based(self) -> bool:
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Learn NSRTs for planning."""
        super().learn_from_offline_dataset(dataset)

    def _get_current_nsrts(self) -> Set[utils.NSRT]:
        """Get NSRTs for planning. If CFG.fm_planning_with_oracle_nsrts is True,
        use oracle NSRTs from the factory. Otherwise, return an empty set."""
        if not CFG.fm_planning_with_oracle_nsrts:
            return set()
        # Get oracle NSRTs from the factory
        return get_gt_nsrts(CFG.env, set(self._initial_predicates), 
                          set(self._initial_options))

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        try:
            option_plan = self._query_llm_for_option_plan(task)
        except Exception as e:
            logging.exception("LLM failed to produce coherent option plan:")  # This will log the full traceback
            raise ApproachFailure(f"LLM failed to produce coherent option plan. Reason: {e}")

        policy = utils.option_plan_to_policy(option_plan)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _query_llm_for_option_plan(self, task: Task) -> List[_Option]:
        """Query the LLM to get a plan of options."""
        # Get NSRTs to access preconditions and effects
        nsrts = self._get_current_nsrts()
        nsrt_by_option = {nsrt.option.name: nsrt for nsrt in nsrts}
        curr_options = sorted(self._initial_options)
        
        # Format options with their parameter types, preconditions, and effects
        options_str = []
        for opt in curr_options:
            params_str = f"params_types={[(f'?x{i}', t.name) for i, t in enumerate(opt.types)]}"
            if opt.name in nsrt_by_option:
                nsrt = nsrt_by_option[opt.name]
                precond_str = f"preconditions={[str(p) for p in nsrt.op.preconditions]}"
                add_effects_str = f"add_effects={[str(e) for e in nsrt.op.add_effects]}"
                delete_effects_str = f"delete_effects={[str(e) for e in nsrt.op.delete_effects]}"
                options_str.append(f"{opt.name}, {params_str}, {precond_str}, {add_effects_str}, {delete_effects_str}")
            else:
                # If no NSRT available, just show parameter types
                options_str.append(f"{opt.name}, {params_str}")
        options_str = "\n  ".join(options_str)
        
        # Format objects and goals
        objects_list = sorted(set(task.init))
        objects_str = "\n".join(str(obj) for obj in objects_list)
        goal_expr_list = sorted(set(task.goal))
        type_hierarchy_str = utils.create_pddl_types_str(self._types)
        goal_str = "\n".join(str(obj) for obj in goal_expr_list)
        
        # Create the prompt using the template
        prompt = self.base_prompt.format(
            options=options_str,
            typed_objects=objects_str,
            type_hierarchy=type_hierarchy_str,
            goal_str=goal_str)
            
        if CFG.fm_planning_verbose:
            logging.info("\n=== LLM Query ===")
            logging.info(f"Prompt:\n{prompt}")
            
        # Query the LLM
        llm_output = self._llm.sample_completions(
            prompt=prompt,
            imgs=None,
            temperature=CFG.llm_temperature,
            seed=CFG.seed,
            num_completions=1)
            
        if CFG.fm_planning_verbose:
            logging.info("\n=== LLM Response ===")
            logging.info(llm_output[0])
            
        # Parse the output into a plan
        plan_prediction_txt = llm_output[0]
        option_plan: List[_Option] = []
        try:
            start_index = plan_prediction_txt.index("Plan:\n") + len("Plan:\n")
            parsable_plan_prediction = plan_prediction_txt[start_index:]
        except ValueError:
            raise ValueError("LLM output is badly formatted; cannot parse plan!")
            
        # Parse the plan using the same method as VLM approach
        parsed_option_plan = utils.parse_model_output_into_option_plan(
            parsable_plan_prediction, objects_list, self._types,
            self._initial_options, parse_continuous_params=False)
            
        # Create grounded options
        for option_tuple in parsed_option_plan:
            option_plan.append(option_tuple[0].ground(
                option_tuple[1], np.array(option_tuple[2])))
                
        if CFG.fm_planning_verbose:
            logging.info("\n=== Parsed Plan ===")
            for opt in option_plan:
                logging.info(f"{opt.name}({', '.join(obj.name for obj in opt.objects)})")
                
        return option_plan
