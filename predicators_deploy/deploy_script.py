

"""Launch script for openstack experiments.

Requires a file that contains a list of IP addresses for instances that are:
    - Turned on
    - Accessible via ssh for the user of this file
    - Configured with an llm_glib image (current snapshot name: llm_glib-v1)
    - Sufficient in number to run all of the experiments in the config file
Make sure to place this file within the openstack_scripts folder.

Usage example:
    python openstack_scripts/launch.py --openstack_config example.yaml \
        --machines machines.txt

The default branch can be overridden with the --branch flag.
"""

import argparse
import os
from typing import List, Optional, Tuple
import subprocess

import numpy as np
import yaml


import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import dotenv
import yaml

env_file = os.path.join(pathlib.Path(__file__).parent, ".env")
dotenv.load_dotenv(env_file, override=True)

SAVE_DIRS = ["results", "runlogs"]
DEFAULT_BRANCH = "super_duper_agi"


@dataclass(frozen=True)
class RunConfig:
    """Config for a single run."""

    experiment_id: str
    template_fn: str


def parse_configs(config_filename: str) -> Iterator[Dict[str, Any]]:
    """Parse the YAML config file."""
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    configs_dir = os.path.join(scripts_dir, "configs")
    config_filepath = os.path.join(configs_dir, config_filename)
    with open(config_filepath, "r", encoding="utf-8") as f:
        for config in yaml.safe_load_all(f):
            yield config


def get_cmds_to_prep_repo(
    branch: str,
) -> List[str]:
    """Get the commands that should be run while already in the repository but
    before launching the experiments."""
    old_dir_pattern = " ".join(f"{d}/" for d in SAVE_DIRS)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    return [
        "cd predicators",
        "conda activate predicators2",
        "killall python",
        "git stash",
        "git fetch --all",
        f"git checkout {branch}",
        "git clean -fd",
        "git pull",
        # Remove old results.
        f"rm -rf {old_dir_pattern}",
        "mkdir runlogs",
        "export PYTHONHASHSEED=0",
        f"export OPENAI_API_KEY={openai_api_key}",
    ]


def run_cmds_on_machine(
    cmds: List[str],
    user: str,
    machine: str,
    ssh_key: Optional[str] = None,
    allowed_return_codes: Tuple[int, ...] = (0,),
) -> None:
    """SSH into the machine, run the commands, then exit."""
    host = f"{user}@{machine}"
    ssh_cmd = f"ssh -tt -v -o StrictHostKeyChecking=no {host}"
    if ssh_key is not None:
        ssh_cmd += f" -i {ssh_key}"
    server_cmd_str = "\n".join(cmds + ["exit"])
    final_cmd = f"{ssh_cmd} << EOF\n{server_cmd_str}\nEOF"

    response = subprocess.run(
        final_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        check=False,
        text=True,
    )
     

def _launch_batched_experiment(
    cmds: List[str], machine: str, logfiles: List[str], ssh_key: str, branch: str
) -> None:
    print(f"Launching on machine {machine}:\n\t" + "\n\t".join(cmds))
    # Enter the repo and activate conda.
    # Prepare the repo.
    server_cmds = (get_cmds_to_prep_repo(branch))
    # Run the main command.
    for cmd, logfile in zip(cmds, logfiles):
        server_cmds.append(f"{cmd} &> {logfile} &")
    run_cmds_on_machine(server_cmds, "ubuntu", machine, ssh_key=ssh_key)


if __name__=='__main__':
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", default="machines_10.txt", type=str)
    parser.add_argument("--sshkey", required=False, type=str, default="/home/aidan/cloud.key")
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH)
    args = parser.parse_args()

    
    # Load the machine IPs.
    machine_file = args.machines
    with open(machine_file, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    # Make sure that the ssh key exists.
    if args.sshkey is not None:
        assert os.path.exists(args.sshkey)
    
    
    for mi, machine in enumerate(machines):
        logfile = f"runlogs/run_seed_{mi}.txt"
        # cmd = f"python -u scripts/mock_experiments.py --env=mock_spot_pick_place_two_cup --seed={mi}"
        cmd = f"python -u scripts/mock_experiments.py --env=mock_spot_sort_weight --seed={mi}"

        _launch_batched_experiment(
            [cmd], machine, [logfile], args.sshkey, args.branch
        )
