"""A text-based LLM planning approach that uses natural language state descriptions."""

from typing import List, Set

from predicators import utils
from predicators.approaches.llm_open_loop_approach import LLMOpenLoopApproach
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task
from typing import Callable, List, Set
from predicators.structs import Action, Box, \
    State, Task, _Option
import numpy as np
from predicators.approaches import ApproachFailure
import logging


class LLMTextPlanningApproach(LLMOpenLoopApproach):
    """An approach that uses text descriptions of states for LLM planning."""

    @classmethod
    def get_name(cls) -> str:
        return "vlm_captioning"

  

    
    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        try:
            option_plan = self._create_prompt(task)
        except Exception as e:
            logging.exception("VLM failed to produce coherent option plan:")  # This will log the full traceback
            raise ApproachFailure(f"VLM failed to produce coherent option plan. Reason: {e}")

        policy = utils.option_plan_to_policy(option_plan)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy
    
    def _create_prompt(self, task:Task) -> str:
        """Create a prompt using the text description of the state."""
        state_desc = task.init.text_description


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
        
        
        # Format action history if available
        action_history_str = ""
        if self._action_history:
            action_history_str = "\n".join(f"Action {i}: {action}" for i, action in enumerate(self._action_history))
            
        
        # Create the prompt using the template
        prompt = self.base_prompt.format(
            options=options_str,
            typed_objects=objects_str,
            type_hierarchy=type_hierarchy_str,
            goal_str=goal_str,
            state_str=state_desc,
            action_history = action_history_str)


        if CFG.fm_planning_verbose:
            logging.info("\n=== LLM Query ===")
            logging.info(f"Prompt:\n{prompt}")
            
        objects_list = sorted(set(task.init))

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
    




    def _goal_atoms_to_text(self, goal: Set[GroundAtom]) -> str:
        """Convert goal atoms to text description."""
        return "\n".join(str(atom) for atom in sorted(goal)) 