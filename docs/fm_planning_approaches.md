# Foundation Model Planning Approaches

This document describes the language model-based planning approaches and execution monitoring strategies available in the codebase.

## Planning Approaches

### LLM Bilevel Planning Approach

The `LLMBilevelPlanningApproach` uses a large language model (LLM) to guide bilevel planning. Key features:

- Uses LLM to generate high-level option plans
- Converts option plans into partial policies to guide A* search
- Falls back to regular planning if LLM suggestions fail
- Assumes one-to-one mapping between options and NSRTs

Example usage:
```bash
python predicators/main.py --approach llm_bilevel_planning \
    --seed 0 \
    --strips_learner oracle \
    --env pddl_blocks_procedural_tasks \
    --num_train_tasks 3 \
    --num_test_tasks 1
```

### VLM Open Loop Approach

The `VLMOpenLoopApproach` uses a vision-language model (VLM) to directly generate plans from visual observations. Key features:

- Takes RGB images as input along with text descriptions
- Can use few-shot examples from training demonstrations
- Generates option plans directly from visual input
- Executes plans in open-loop fashion

Example usage:
```bash
python predicators/main.py --approach vlm_open_loop \
    --env burger \
    --seed 0 \
    --num_train_tasks 0 \
    --num_test_tasks 1 \
    --bilevel_plan_without_sim True \
    --vlm_model_name gpt-4o
```

### VLM Bilevel Planning Approach

The `VLMBilevelPlanningApproach` combines vision-language capabilities with bilevel planning. Key features:

- Uses VLM to generate option plans from visual input
- Converts VLM suggestions into search guidance
- Integrates with execution monitoring
- Combines benefits of both VLM and bilevel planning

Example usage:
```bash
python predicators/main.py --approach vlm_bilevel_planning \
    --env mock_spot_pick_place_two_cup \
    --seed 0 \
    --num_train_tasks 0 \
    --num_test_tasks 1 \
    --bilevel_plan_without_sim True \
    --execution_monitor expected_atoms \
    --vlm_model_name gpt-4o
```

## Execution Monitoring

The codebase provides several execution monitoring strategies to handle uncertainty and trigger replanning when needed.

### Expected Atoms Monitor

The `ExpectedAtomsExecutionMonitor` checks if the expected atoms from the plan match reality:

- Compares expected atoms from plan against current state
- Works with both regular and VLM predicates
- Triggers replanning if expectations are violated
- Best suited for bilevel planning approaches

Example usage:
```bash
--execution_monitor expected_atoms
```

### MPC Execution Monitor

The `MpcExecutionMonitor` implements a model-predictive control strategy:

- Always triggers replanning after the first timestep
- Enables continuous plan adaptation
- Useful for handling dynamic environments
- Works with any planning approach

Example usage:
```bash
--execution_monitor mpc
```

### Trivial Execution Monitor

The `TrivialExecutionMonitor` provides a baseline that never triggers replanning:

- Never suggests replanning
- Useful for testing and debugging
- Baseline for comparing monitoring strategies

Example usage:
```bash
--execution_monitor trivial
```

## Choosing an Approach

Here are some guidelines for choosing between approaches:

1. **LLM Bilevel Planning**:
   - Best for tasks with clear symbolic descriptions
   - When you have good language prompts
   - Tasks that benefit from guided search

2. **VLM Open Loop**:
   - Best for visually-guided tasks
   - When you have good visual demonstrations
   - Simple tasks with reliable execution

3. **VLM Bilevel Planning**:
   - Best for complex visual tasks
   - When you need robust execution
   - Tasks that benefit from both visual and symbolic reasoning

## Choosing a Monitor

Guidelines for choosing execution monitors:

1. **Expected Atoms Monitor**:
   - Use with bilevel planning approaches
   - When you have clear expected states
   - Tasks with discrete state transitions

2. **MPC Monitor**:
   - Use in dynamic environments
   - When continuous replanning is beneficial
   - Tasks with uncertainty

3. **Trivial Monitor**:
   - Use for debugging
   - Simple environments
   - When testing base planner behavior 