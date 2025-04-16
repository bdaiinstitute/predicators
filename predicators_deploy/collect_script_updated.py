"""Download results from experiments.

This script supports two modes:
1. SSH config mode (default): Uses your SSH config for authentication
2. OpenStack mode: Uses explicit SSH key for OpenStack instances

Usage examples:
    # SSH config mode (default):
    python predicators_deploy/collect_script.py --machines machines.txt

    # OpenStack mode:
    python predicators_deploy/collect_script.py --machines machines.txt --sshkey ~/.ssh/cloud.key --user ubuntu
"""

import argparse
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

def run_cmds_on_machine(
    cmds: List[str],
    machine: str,
    user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    use_ssh_config: bool = True,
    allowed_return_codes: Tuple[int, ...] = (0,),
) -> None:
    """SSH into the machine, run the commands, then exit."""
    if use_ssh_config:
        host = machine
        ssh_cmd = f"ssh -tt -v {host}"
    else:
        assert user is not None, "User must be specified when not using SSH config"
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

def _download_from_machine(
    machine: str,
    user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    use_ssh_config: bool = True,
) -> None:
    print(f"Downloading from machine {machine}")
    local_save_dir = "./results_collected"

    # Create a temporary directory for downloading
    temp_dir = "./temp_download"
    os.makedirs(temp_dir, exist_ok=True)

    # Create tar file on remote machine
    run_cmds_on_machine(
        ["cd predicators", "tar -czvf results.tar.gz results_deploy runlogs"],
        machine,
        user=user,
        ssh_key=ssh_key,
        use_ssh_config=use_ssh_config,
    )

    # Construct scp command based on mode
    if use_ssh_config:
        cmd = f"scp -r {machine}:~/predicators/results.tar.gz {temp_dir}/results.tar.gz"
    else:
        cmd = f"scp -r "
        if ssh_key is not None:
            cmd += f"-i {ssh_key} "
        cmd += (
            "-o StrictHostKeyChecking=no "
            + f"{user}@{machine}:~/predicators/results.tar.gz {temp_dir}/results.tar.gz"
        )

    print("Executing command: " + str(cmd))
    retcode = os.system(cmd)
    if retcode != 0:
        print(f"WARNING: command failed: {cmd}")
        return

    # Extract the tar file in the temporary directory
    tar_command = f"tar -xzvf {temp_dir}/results.tar.gz -C {temp_dir}"
    print(tar_command)
    os.system(tar_command)

    # Create the local_save_dir if it doesn't exist
    os.makedirs(local_save_dir, exist_ok=True)

    # Copy contents from temp_dir to local_save_dir, overwriting existing files and folders
    for item in os.listdir(f"{temp_dir}/results_deploy"):
        s = os.path.join(f"{temp_dir}/results_deploy", item)
        d = os.path.join(local_save_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # Also copy runlogs if they exist
    if os.path.exists(f"{temp_dir}/runlogs"):
        runlogs_dir = "./runlogs"
        os.makedirs(runlogs_dir, exist_ok=True)
        for item in os.listdir(f"{temp_dir}/runlogs"):
            s = os.path.join(f"{temp_dir}/runlogs", item)
            d = os.path.join(runlogs_dir, item)
            shutil.copy2(s, d)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    print("Download and extraction completed successfully.")

def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machines", required=True, type=str,
                      help="File containing list of machines")
    parser.add_argument("--sshkey", type=str,
                      help="SSH key file (for OpenStack mode)")
    parser.add_argument("--user", type=str,
                      help="Remote username (for OpenStack mode)")
    parser.add_argument("--use_ssh_config", action="store_true", default=True,
                      help="Use SSH config for authentication (default)")
    args = parser.parse_args()

    # If sshkey or user is provided, switch to OpenStack mode
    if args.sshkey is not None or args.user is not None:
        args.use_ssh_config = False
        if args.user is None:
            args.user = "ubuntu"  # Default OpenStack user

    # Validate OpenStack mode arguments
    if not args.use_ssh_config:
        if args.sshkey is not None:
            assert os.path.exists(args.sshkey), f"SSH key not found: {args.sshkey}"

    # Load the machine IPs
    with open(args.machines, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()

    # Download from each machine
    for machine in machines:
        _download_from_machine(
            machine,
            user=args.user,
            ssh_key=args.sshkey,
            use_ssh_config=args.use_ssh_config,
        )

if __name__ == "__main__":
    _main()
