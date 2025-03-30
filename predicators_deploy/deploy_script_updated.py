"""Launch script for running experiments on remote machines.

Usage example:
    python deploy_script.py --machines machines.txt --username your_username --envs "mock_spot_sort_weight mock_spot_cup_emptiness" --planner oracle
    
Or use machines from your .ssh/config:
    python deploy_script.py --machines machines.txt --use_ssh_config --envs "mock_spot_sort_weight mock_spot_cup_emptiness" --planner oracle --num_seeds 10 --parallel
"""

import argparse
import os
from typing import List, Optional, Tuple
import subprocess
import pathlib
import math

try:
    import dotenv
    # Load environment variables if needed
    env_file = os.path.join(pathlib.Path(__file__).parent, ".env")
    dotenv.load_dotenv(env_file, override=True)
except ImportError:
    pass  # dotenv is optional

SAVE_DIRS = ["results_deploy", "runlogs"]
DEFAULT_BRANCH = "super_duper_agi"

def get_cmds_to_prep_repo(branch: str) -> List[str]:
    """Get the commands to run in the repository."""
    old_dir_pattern = " ".join(f"{d}/" for d in SAVE_DIRS)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    return [
        "cd predicators",
        "conda activate predicators2",  # Change this if your env name is different
        "killall python",
        "git stash",
        "git fetch --all",
        f"git checkout {branch}",
        "git clean -fd",
        "git pull",
        f"rm -rf {old_dir_pattern}",
        "mkdir runlogs",
        "mkdir results_deploy",
        "export PYTHONHASHSEED=0",
        f"export OPENAI_API_KEY={openai_api_key}",
    ]

def run_cmds_on_machine(
    cmds: List[str],
    user: str,
    machine: str,
    use_ssh_config: bool = False,
    ssh_key: Optional[str] = None,
) -> None:
    """SSH into the machine and run commands."""
    host = machine if use_ssh_config else f"{user}@{machine}"
    
    # Base SSH command
    if use_ssh_config:
        ssh_cmd = f"ssh -tt {host}"  # Use settings from .ssh/config
    else:
        ssh_cmd = f"ssh -tt -o StrictHostKeyChecking=no {host}"
        if ssh_key is not None:
            ssh_cmd += f" -i {ssh_key}"
    
    server_cmd_str = "\n".join(cmds + ["exit"])
    final_cmd = f"{ssh_cmd} << EOF\n{server_cmd_str}\nEOF"
    
    print(f"Running command: {final_cmd}")
    response = subprocess.run(
        final_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        check=False,
        text=True,
    )
    if response.returncode != 0:
        print(f"Warning: Command failed with return code {response.returncode}")
        print(f"stderr: {response.stderr}")

def _launch_batched_experiment(
    cmds: List[str], 
    machine: str, 
    logfiles: List[str], 
    username: str,
    use_ssh_config: bool = False,
    ssh_key: Optional[str] = None,
    branch: str = DEFAULT_BRANCH,
    parallel: bool = False
) -> None:
    print(f"Launching on machine {machine}:\n\t" + "\n\t".join(cmds))
    
    # Prepare and run commands
    server_cmds = get_cmds_to_prep_repo(branch)
    
    # Add command to set results directory to results_deploy
    server_cmds.append("export RESULTS_DIR=results_deploy")
    
    if parallel:
        # Run all commands in parallel using GNU parallel
        server_cmds.append("# Run experiments in parallel")
        for cmd, logfile in zip(cmds, logfiles):
            server_cmds.append(f"{cmd} --results_dir results_deploy &> {logfile} &")
    else:
        # Run commands sequentially
        server_cmds.append("# Run experiments sequentially")
        for cmd, logfile in zip(cmds, logfiles):
            server_cmds.append(f"{cmd} --results_dir results_deploy &> {logfile}")
    
    run_cmds_on_machine(
        server_cmds, 
        username, 
        machine, 
        use_ssh_config=use_ssh_config,
        ssh_key=ssh_key
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machines", default="machines.txt", type=str,
                       help="File containing machine hostnames/IPs, one per line")
    parser.add_argument("--username", type=str, default=os.getenv("USER"),
                       help="SSH username")
    parser.add_argument("--use_ssh_config", action="store_true",
                       help="Use machine names from .ssh/config")
    parser.add_argument("--sshkey", type=str, default=None,
                       help="Path to SSH key (optional if using ssh config)")
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH,
                       help="Git branch to use")
    parser.add_argument("--envs", type=str, required=True,
                       help="Space-separated list of environments to run")
    parser.add_argument("--planner", type=str, default=None,
                       help="Planner to use for all environments")
    parser.add_argument("--num_seeds", type=int, default=1,
                       help="Total number of seeds to run")
    parser.add_argument("--parallel", action="store_true",
                       help="Run seeds in parallel on each machine")
    args = parser.parse_args()

    # Load the machine names/IPs
    with open(args.machines, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    
    # Verify SSH key if provided
    if args.sshkey is not None and not args.use_ssh_config:
        assert os.path.exists(args.sshkey), f"SSH key not found: {args.sshkey}"
    
    # Calculate seeds per machine
    num_machines = len(machines)
    seeds_per_machine = math.ceil(args.num_seeds / num_machines)
    print(f"\nDistributing {args.num_seeds} seeds across {num_machines} machines:")
    print(f"- Each machine will run up to {seeds_per_machine} seeds")
    
    # Launch experiments on each machine
    for mi, machine in enumerate(machines):
        cmds = []
        logfiles = []
        start_seed = mi * seeds_per_machine
        end_seed = min((mi + 1) * seeds_per_machine, args.num_seeds)
        
        print(f"\nMachine {machine} will run seeds {start_seed} to {end_seed-1}")
        
        for env in args.envs.split():
            for seed in range(start_seed, end_seed):
                logfile = f"runlogs/run_{env}_seed_{seed}.txt"
                cmd = f"python -u scripts/mock_experiments.py --env={env} --seed={seed}"
                if args.planner:
                    cmd += f" --planner={args.planner}"
                cmds.append(cmd)
                logfiles.append(logfile)

        _launch_batched_experiment(
            cmds,
            machine,
            logfiles,
            args.username,
            use_ssh_config=args.use_ssh_config,
            ssh_key=args.sshkey,
            branch=args.branch,
            parallel=args.parallel
        )
