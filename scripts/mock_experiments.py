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
    python scripts/mock_experiments.py --env mock_spot_drawer_cleaning
    python scripts/mock_experiments.py --envs "mock_spot_drawer_cleaning mock_spot_sort_weight"
    python scripts/mock_experiments.py --envs "mock_spot_drawer_cleaning mock_spot_sort_weight" --planner oracle
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

# Available environments from mock_spot_env.py
AVAILABLE_ENVS = [
    "mock_spot_drawer_cleaning",
    "mock_spot_pick_place_two_cup", 
    "mock_spot_sort_weight",
    "mock_spot_cup_emptiness"
]

# Available planners
AVAILABLE_PLANNERS = [
    "oracle",
    "oracle_closed_loop",
    "oracle_open_loop",
    "random",
    "llm_closed_loop",
    "vlm_closed_loop",
    "vlm_captioning",
    "vlm_captioning_open_loop"
]

def create_base_command(env: str, seed: int = 0, results_dir: str = "results") -> List[str]:
    """Create the base command with common arguments."""
    return [
        "python", "predicators/main.py",
        "--env", env,
        "--seed", str(seed),
        "--num_train_tasks", "0",
        "--num_test_tasks", "1",
        "--bilevel_plan_without_sim", "True",
        "--horizon", "20",  # NOTE: this is the max horizon for the mock env; may need to adjust for different environments
        "--load_approach",
        "--results_dir", results_dir,  # Add results_dir argument
    ]

def run_command(cmd: List[str], name: str) -> None:
    """Run a command and handle its output."""
    logging.info(f"\n=== Running {name} ===")
    logging.info(f"Command: {' '.join(cmd)}")
    
    process = None
    try:
        # Run the command and stream output in real-time
        print(cmd)
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
    
    # # Create base command
    # base_cmd = create_base_command(args.env, args.seed)
    
    # Get environments to run
    envs = []
    if args.env:
        envs = [args.env]
    elif args.envs:
        envs = args.envs.split()
    
    # Validate environments
    for env in envs:
        if env not in AVAILABLE_ENVS:
            raise ValueError(f"Unknown environment: {env}. Available environments: {AVAILABLE_ENVS}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Define all planner configurations
    planners = [
        {
            "name": "oracle",
            "args": ["--approach", 
                     "oracle",
                     "--perceiver", "mock_spot_perceiver",
                     "--method_name", "oracle",
                    #  # NOTE: just added execution_monitor; why didn't put it before?
                    #  "--execution_monitor", "expected_atoms"
                     ]
        },
        {
            "name": "oracle_closed_loop",
            "args": ["--approach", 
                     "oracle",
                     "--perceiver", "mock_spot_perceiver",
                     "--method_name", "oracle_closed_loop",
                     # NOTE: just added execution_monitor; why didn't put it before?
                     "--execution_monitor", "expected_atoms"]
        },
        {
            "name": "oracle_open_loop",
            "args": ["--approach", 
                     "oracle",
                     "--perceiver", "mock_spot_perceiver",
                     "--method_name", "oracle_open_loop"]
        },
        # {
        #     "name": "random",
        #     "args": [
        #         "--approach", "random_options",
        #         "--random_options_max_tries", "1000",
        #         "--max_num_steps_option_rollout", "100",
        #         "--perceiver", "mock_spot_perceiver",
        #         "--timeout", "60",
        #     ]
        # },
        {
            "name": "llm_closed_loop",
            "args": [
                "--approach", "llm_open_loop",
                "--perceiver", "mock_spot_perceiver",
                "--llm_model_name", "gpt-4o",
                "--llm_temperature", "0.2",
                "--execution_monitor", "mpc"
                # "--execution_monitor", "expected_atoms"
            ]
        },
        {
            "name": "vlm_closed_loop",
            "args": [
                "--approach", "vlm_open_loop",
                "--perceiver", "mock_spot_perceiver",
                "--vlm_model_name", "gpt-4o",
                "--llm_temperature", "0.2",
                "--execution_monitor", "mpc"
                # "--execution_monitor", "expected_atoms"
            ]
        },
        {
            "name": "vlm_captioning",
            "args": [
                "--approach", "vlm_captioning",
                "--perceiver", "vlm_perceiver",
                "--vlm_model_name", "gpt-4o",
                "--vlm_temperature", "0.2",
                "--execution_monitor", "mpc",
                "--method_name", "vlm_captioning"
            ]
        },
        {
            # NOTE: no execution monitor, for replicating Sun et al. 2024
            "name": "vlm_captioning_open_loop",
            "args": [
                "--approach", "vlm_captioning",
                "--perceiver", "vlm_perceiver",
                "--vlm_model_name", "gpt-4o",
                "--vlm_temperature", "0.2",
                "--method_name", "vlm_captioning_open_loop"
            ]
        }
    ]
    
    # # Run each planner
    # for planner in planners:
    #     if args.planner and planner["name"].lower() != args.planner.lower():
    #         continue
            
    #     cmd = base_cmd + planner["args"]
    #     # if args.load_approach:
    #     #     cmd.append("--load_approach")
    #     print(cmd)
    #     run_command(cmd, planner["name"])
    # Validate planner if specified
    
    if args.planner and args.planner not in AVAILABLE_PLANNERS:
        raise ValueError(f"Unknown planner: {args.planner}. Available planners: {AVAILABLE_PLANNERS}")
    
    # Run each environment with specified or all planners
    for env in envs:
        logging.info(f"\n=== Running Environment: {env} ===")
        for planner in planners:
            if args.planner and planner["name"].lower() != args.planner.lower():
                continue
                
            base_cmd = create_base_command(env, args.seed, args.results_dir)
            cmd = base_cmd + planner["args"]
            run_command(cmd, f"{env}_{planner['name']}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run different planners on the mock robot pick and place task.")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--planner", type=str, choices=AVAILABLE_PLANNERS,
                       help="Run only this planner (by name)")
    parser.add_argument("--env", type=str, choices=AVAILABLE_ENVS,
                       help="Single environment to run")
    parser.add_argument("--envs", type=str,
                       help="Space-separated list of environments to run")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory to save results")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s %(levelname)s: %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_args()
    
    # Validate that either --env or --envs is provided
    if not args.env and not args.envs:
        raise ValueError("Must provide either --env or --envs argument")
    if args.env and args.envs:
        raise ValueError("Cannot provide both --env and --envs arguments")
        
    main(args) 
