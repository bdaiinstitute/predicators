"""Open-loop large language model (LLM) meta-controller approach.

Example command line:
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

Easier setting:
    python predicators/main.py --approach llm_open_loop --seed 0 \
        --strips_learner oracle \
        --env pddl_easy_delivery_procedural_tasks \
        --pddl_easy_delivery_procedural_train_min_num_locs 2 \
        --pddl_easy_delivery_procedural_train_max_num_locs 2 \
        --pddl_easy_delivery_procedural_train_min_want_locs 1 \
        --pddl_easy_delivery_procedural_train_max_want_locs 1 \
        --pddl_easy_delivery_procedural_train_min_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_train_max_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_test_min_num_locs 2 \
        --pddl_easy_delivery_procedural_test_max_num_locs 2 \
        --pddl_easy_delivery_procedural_test_min_want_locs 1 \
        --pddl_easy_delivery_procedural_test_max_want_locs 1 \
        --pddl_easy_delivery_procedural_test_min_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_test_max_extra_newspapers 0 \
        --num_train_tasks 5 \
        --num_test_tasks 10 \
        --debug
"""
from __future__ import annotations

from typing import Collection, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple, Any
import logging
import os
from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.nsrt_metacontroller_approach import \
    NSRTMetacontrollerApproach
from predicators.ground_truth_models import get_gt_nsrts
from predicators.planning import task_plan_with_option_plan_constraint
from predicators.settings import CFG
from predicators.structs import Box, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option
from predicators.pretrained_model_interface import create_llm_by_name


class LLMOpenLoopApproach(NSRTMetacontrollerApproach):
    """LLMOpenLoopApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the LLM.
        self._llm = create_llm_by_name(CFG.llm_model_name)
        # Load the base prompt from file
        filepath_to_llm_prompt = utils.get_path_to_predicators_root() + \
            "/predicators/approaches/llm_planning_prompts/zero_shot.txt"
        with open(filepath_to_llm_prompt, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()
        # Store the current plan for monitoring
        self._current_plan: Optional[List[_GroundNSRT]] = None
        # Store the current state for sampling options
        self._current_state: Optional[State] = None
        # Store the current goal for sampling options
        self._current_goal: Optional[Set[GroundAtom]] = None

    @classmethod
    def get_name(cls) -> str:
        return "llm_open_loop"

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict) -> _GroundNSRT:
        # Store current state and goal for sampling options
        self._current_state = state
        self._current_goal = goal
        # If we already have an abstract plan, execute the next step.
        if "abstract_plan" in memory and memory["abstract_plan"]:
            return memory["abstract_plan"].pop(0)
        # Otherwise, we need to make a new abstract plan.
        action_seq = self._get_llm_based_plan(state, atoms, goal)
        if action_seq is not None:
            # If valid plan, add plan to memory so it can be refined!
            memory["abstract_plan"] = action_seq
            return memory["abstract_plan"].pop(0)
        raise ApproachFailure("No LLM predicted plan achieves the goal.")

    def get_execution_monitoring_info(self) -> List[Any]:
        """Return the current plan in a format expected by the execution monitor."""
        if self._current_plan is None or self._current_state is None or self._current_goal is None:
            return []
        # Sample options for each ground NSRT in the plan
        options = []
        for nsrt in self._current_plan:
            option = nsrt.sample_option(self._current_state, self._current_goal, self._rng)
            options.append(option)
        return [{"current_option_plan": options}]

    def _get_llm_based_plan(
            self, state: State, atoms: Set[GroundAtom],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        if CFG.fm_planning_verbose:
            logging.info("\n=== LLM Planning ===")
            logging.info(f"Initial atoms: {sorted(atoms)}")
            logging.info(f"Goal atoms: {sorted(goal)}")
            
        # Try to convert each output into an abstract plan.
        # Return the first abstract plan that is found this way.
        objects = set(state)
        for option_plan in self._get_llm_based_option_plans(
                atoms, objects, goal):
            if CFG.fm_planning_verbose:
                logging.info("\nTrying option plan:")
                for option, objs in option_plan:
                    logging.info(f"  {option.name}({[obj.name for obj in objs]})")
                    
            ground_nsrt_plan = self._option_plan_to_nsrt_plan(
                option_plan, atoms, objects, goal)
                
            if ground_nsrt_plan is not None:
                if CFG.fm_planning_verbose:
                    logging.info("\nFound valid NSRT plan:")
                    for nsrt in ground_nsrt_plan:
                        logging.info(f"  {nsrt}")
                # Store the current plan for monitoring
                self._current_plan = ground_nsrt_plan
                return ground_nsrt_plan
            elif CFG.fm_planning_verbose:
                logging.info("Plan validation failed")
                
        if CFG.fm_planning_verbose:
            logging.warning("No valid plans found")
        return None
    
    def _get_current_nsrts(self) -> Set[utils.NSRT]:
        """Get NSRTs for planning. If CFG.fm_planning_with_oracle_nsrts is True,
        use oracle NSRTs from the factory. Otherwise, return an empty set."""
        if not CFG.fm_planning_with_oracle_nsrts:
            return set()
        # Get oracle NSRTs from the factory
        return get_gt_nsrts(CFG.env, set(self._initial_predicates), 
                          set(self._initial_options))

    def _get_llm_based_option_plans(
        self, atoms: Set[GroundAtom], objects: Set[Object],
        goal: Set[GroundAtom]
    ) -> Iterator[List[Tuple[ParameterizedOption, Sequence[Object]]]]:
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
            
        # Format objects with their types
        objects_str = "\n  ".join(f"{obj.name}: {obj.type.name}" for obj in sorted(objects))
        
        # Create type hierarchy string
        type_hierarchy_str = utils.create_pddl_types_str(self._types)
        
        # Format goal atoms
        goal_str = "\n  ".join(map(str, sorted(goal)))
        
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
        llm_predictions = self._llm.sample_completions(
            prompt=prompt,
            imgs=None,
            temperature=CFG.llm_temperature,
            seed=CFG.seed,
            num_completions=CFG.llm_num_completions)
            
        if CFG.fm_planning_verbose:
            logging.info("\n=== LLM Responses ===")
            for i, pred in enumerate(llm_predictions):
                logging.info(f"\nPrediction {i+1}:")
                logging.info(pred)
                
        for pred in llm_predictions:
            option_plan = self._llm_prediction_to_option_plan(pred, objects)
            if CFG.fm_planning_verbose:
                logging.info("\n=== Formatted Option Plan ===")
                if not option_plan:
                    logging.info("Empty plan (parsing failed)")
                for option, objs in option_plan:
                    logging.info(f"{option.name}({', '.join(obj.name for obj in objs)})")
            yield option_plan

    def _option_plan_to_nsrt_plan(
            self, option_plan: List[Tuple[ParameterizedOption,
                                          Sequence[Object]]],
            atoms: Set[GroundAtom], objects: Set[Object],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        nsrts = self._get_current_nsrts()
        predicates = self._initial_predicates
        strips_ops = [n.op for n in nsrts]
        option_specs = [(n.option, list(n.option_vars)) for n in nsrts]
        return task_plan_with_option_plan_constraint(objects, predicates,
                                                     strips_ops, option_specs,
                                                     atoms, goal, option_plan)

    def _llm_prediction_to_option_plan(
        self, llm_prediction: str, objects: Collection[Object]
    ) -> List[Tuple[ParameterizedOption, Sequence[Object]]]:
        """Convert the output of the LLM into a sequence of
        ParameterizedOptions coupled with a list of objects that will be used
        to ground the ParameterizedOption."""
        option_plan: List[Tuple[ParameterizedOption, Sequence[Object]]] = []
        try:
            # Look for the "Plan:" section
            if "Plan:" not in llm_prediction:
                raise ValueError("No 'Plan:' section found in LLM output")
            plan_section = llm_prediction.split("Plan:")[1].strip()
            
            # Use the same parsing as VLM approach
            parsed_option_plan = utils.parse_model_output_into_option_plan(
                plan_section, objects, self._types,
                self._initial_options, parse_continuous_params=False)
            
            # Convert to expected format
            return [(option, objs) for option, objs, _ in parsed_option_plan]
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {e}")

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, learn NSRTs.
        super().learn_from_offline_dataset(dataset)

    @staticmethod
    def _option_to_str(option: _Option) -> str:
        objects_str = ", ".join(map(str, option.objects))
        return f"{option.name}({objects_str})"
