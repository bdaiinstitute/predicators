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

    def _get_llm_based_option_plans(
        self, atoms: Set[GroundAtom], objects: Set[Object],
        goal: Set[GroundAtom]
    ) -> Iterator[List[Tuple[ParameterizedOption, Sequence[Object]]]]:
        # Format options with their parameter types and spaces
        options_str = "\n  ".join(
            f"{opt.name}, params_types={[(f'?x{i}', t.name) for i, t in enumerate(opt.types)]}, params_space={opt.params_space}"
            for opt in sorted(self._initial_options))
            
        # Format objects with their types
        objects_str = "\n  ".join(f"{obj}: {obj.type.name}" for obj in sorted(objects))
        
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
        option_plan_with_cont_params = utils.\
            parse_model_output_into_option_plan(
            llm_prediction, objects, self._types, self._initial_options, False)
        option_plan = [(option, objs)
                       for option, objs, _ in option_plan_with_cont_params]
        return option_plan

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, learn NSRTs.
        super().learn_from_offline_dataset(dataset)
        # Then, parse the data into the prompting format expected by the LLM.
        self._prompt_prefix = self._data_to_prompt_prefix(dataset)

    def _data_to_prompt_prefix(self, dataset: Dataset) -> str:
        # In this approach, we learned NSRTs, so we just use the segmented
        # trajectories that NSRT learning returned to us.
        prompts = []
        assert len(self._segmented_trajs) == len(dataset.trajectories)
        for segment_traj, ll_traj in zip(self._segmented_trajs,
                                         dataset.trajectories):
            if not ll_traj.is_demo:
                continue
            init = segment_traj[0].init_atoms
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            seg_options = []
            for segment in segment_traj:
                assert segment.has_option()
                seg_options.append(segment.get_option())
            prompt = self._create_prompt(init, goal, seg_options)
            prompts.append(prompt)
        return "\n\n".join(prompts) + "\n\n"

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        init_str = "\n  ".join(map(str, sorted(init)))
        goal_str = "\n  ".join(map(str, sorted(goal)))
        options_str = "\n  ".join(map(self._option_to_str, options))
        prompt = f"""
(:init
  {init_str}
)
(:goal
  {goal_str}
)
Solution:
  {options_str}"""
        return prompt

    def _create_detailed_prompt(self, atoms: Set[GroundAtom], 
                              objects: Set[Object],
                              goal: Set[GroundAtom]) -> str:
        """Create a more detailed prompt including state and operator information."""
        # Format objects with their types
        objects_str = "\n  ".join(f"{obj}: {obj.type.name}" for obj in sorted(objects))
        
        # Format available options with their parameter types and spaces
        options_str = "\n  ".join(
            f"{opt.name}, params_types={[(f'?x{i}', t.name) for i, t in enumerate(opt.types)]}, params_space={opt.params_space}"
            for opt in sorted(self._initial_options))
            
        # Format current state atoms and goal atoms
        atoms_str = "\n  ".join(map(str, sorted(atoms)))
        goal_str = "\n  ".join(map(str, sorted(goal)))
        
        # Create type hierarchy string
        type_hierarchy_str = utils.create_pddl_types_str(self._types)
        
        prompt = f"""
Available objects and their types:
  {objects_str}

Type hierarchy:
  {type_hierarchy_str}

Available operators:
  {options_str}

Current state atoms:
  {atoms_str}

Goal atoms:
  {goal_str}

Provide ONLY a sequence of actions to achieve the goal, one per line, with NO additional text or formatting.
Each action must be in the exact format:
option_name(obj0:type0, obj1:type1, ...)

Example response:
PickObjectFromTop(robot:robot, red_cup:container, table:immovable_object)
DropObjectInside(robot:robot, red_cup:container, target:container)

Solution:"""
        return prompt

    @staticmethod
    def _option_to_str(option: _Option) -> str:
        objects_str = ", ".join(map(str, option.objects))
        return f"{option.name}({objects_str})"
