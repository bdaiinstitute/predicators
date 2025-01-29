#!/usr/bin/env python3
"""Script to run different planners on the mock robot pick and place task with 2 cups.

This script runs the following planners:
1. Oracle (baseline)
2. Random Options (baseline)
3. LLM Open Loop
4. LLM Closed Loop (with MPC)
5. VLM Open Loop
6. VLM Closed Loop (Open Loop + MPC)
7. VLM Closed Loop (Bilevel + MPC)

Example usage:
    python scripts/run_mock_robot_task_pick_place_2_cups.py
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import List, Optional

# Add the predicators directory to the Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from predicators.settings import CFG


def create_base_command(seed: int = 0) -> List[str]:
    """Create the base command with common arguments."""
    return [
        "python", "predicators/main.py",
        "--env", "mock_spot_drawer_cleaning",
        "--seed", str(seed),
        "--perceiver", "mock_spot_perceiver",
        # "--mock_env_vlm_eval_predicate", "True",
        "--num_train_tasks", "0",
        "--num_test_tasks", "1",
        # "--log_rich", "True",
        "--bilevel_plan_without_sim", "True",
        "--horizon", "20",
        "--load_approach",
    ]

def run_command(cmd: List[str], name: str) -> None:
    """Run a command and handle its output."""
    logging.info(f"\n=== Running {name} ===")
    logging.info(f"Command: {' '.join(cmd)}")
    
    process = None
    try:
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=os.environ.copy()  # Pass current environment variables
        )
        
        # Stream output in real-time
        if process.stdout is not None:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                
        return_code = process.poll()
        if return_code != 0:
            logging.error(f"{name} failed with return code {return_code}")
        else:
            logging.info(f"{name} completed successfully")
            
    except subprocess.CalledProcessError as e:
        logging.error(f"{name} failed with error: {e}")
    except KeyboardInterrupt:
        logging.info(f"\nInterrupted {name}")
        if process is not None:
            process.terminate()
        sys.exit(1)

def main(args: argparse.Namespace) -> None:
    """Run all planners on the mock robot pick and place task."""
    # Create base command
    base_cmd = create_base_command(args.seed)
    
    # Define all planner configurations
    planners = [
        # {
        #     "name": "Oracle",
        #     "args": ["--approach", "oracle"]
        # },
        # {
        #     "name": "Random Options",
        #     "args": [
        #         "--approach", "random_options",
        #         "--random_options_max_tries", "1000",
        #         "--max_num_steps_option_rollout", "100",
        #         "--timeout", "60",
        #         "--horizon", "20",  # NOTE: should need more steps, but need to decide one
        #     ]
        # },
        # {
        #     "name": "LLM Open Loop",
        #     "args": [
        #         "--approach", "llm_open_loop",
        #         "--llm_model_name", "gpt-4o",
        #         "--llm_temperature", "0.2"
        #     ]
        # },
        {
            "name": "LLM Closed Loop (MPC)",
            "args": [
                "--approach", "llm_open_loop",
                "--llm_model_name", "gpt-4o",
                "--llm_temperature", "0.2",
                # "--execution_monitor", "mpc"
                "--execution_monitor", "expected_atoms"
            ]
        },
        # {
        #     "name": "VLM Open Loop",
        #     "args": [
        #         "--approach", "vlm_open_loop",
        #         "--vlm_model_name", "gpt-4o",
        #         "--llm_temperature", "0.2"
        #     ]
        # },
        # {
        #     "name": "VLM Closed Loop (Open Loop + MPC)",
        #     "args": [
        #         "--approach", "vlm_open_loop",
        #         "--vlm_model_name", "gpt-4o",
        #         "--llm_temperature", "0.2",
        #         # "--execution_monitor", "mpc"
        #         "--execution_monitor", "expected_atoms"
        #     ]
        # },
        # {
        #     "name": "VLM Closed Loop (Bilevel + MPC)",
        #     "args": [
        #         "--approach", "vlm_oracle_bilevel_planning",
        #         "--vlm_model_name", "gpt-4o",
        #         "--vlm_temperature", "0.2",
        #         "--execution_monitor", "expected_atoms"
        #     ]
        # }
    ]
    
    # Run each planner
    for planner in planners:
        if args.planner and planner["name"].lower() != args.planner.lower():
            continue
            
        cmd = base_cmd + planner["args"]
        # if args.load_approach:
        #     cmd.append("--load_approach")
        run_command(cmd, planner["name"])

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run different planners on the mock robot pick and place task.")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--planner", type=str,
                       help="Run only this planner (by name)")
    # parser.add_argument("--load_approach", action="store_true",
    #                    help="Load saved approach")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s %(levelname)s: %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_args()
    main(args) 
