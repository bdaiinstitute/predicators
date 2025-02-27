"""Download results from openstack experiments.

Requires a file that contains a list of IP addresses for instances that are:
    - Turned on
    - Accessible via ssh for the user of this file
    - Configured with an llm_glib image (current snapshot name: llm_glib-v1)
    - Sufficient in number to run all of the experiments in the config file
Make sure to place this file within the openstack_scripts folder.

The dir flag should point to a directory where the results, logs, and llm_cache
subdirectories will be downloaded.

Usage example:
    python scripts/openstack/download.py --dir "$PWD" --machines machines.txt \
        --sshkey ~/.ssh/cloud.key
"""

import argparse
import os
import shutil
import subprocess
from typing import List, Optional, Tuple
import shutil

def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", default="machines_10.txt", type=str)
    parser.add_argument("--sshkey", default="/home/aidan/cloud.key", type=str)
    args = parser.parse_args()
    openstack_dir = os.path.dirname(os.path.realpath(__file__))
    # Load the machine IPs.
    machine_file = os.path.join(openstack_dir, args.machines)
    with open(machine_file, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    # Make sure that the ssh key exists.
    if args.sshkey is not None:
        assert os.path.exists(args.sshkey)
    # Create the download directory if it doesn't exist.
    # os.makedirs(args.dir, exist_ok=True)
    # Loop over machines.
    for machine in machines:
        _download_from_machine(machine, args.sshkey)




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
     
     
def _download_from_machine(machine: str, ssh_key: str) -> None:
    print(f"Downloading from machine {machine}")
    local_save_dir = "./results"

    # Create a temporary directory for downloading
    temp_dir = "./temp_download"
    os.makedirs(temp_dir, exist_ok=True)

    run_cmds_on_machine(
        ["cd predicators", "tar -czvf results.tar.gz results"],
        "ubuntu",
        machine,
        ssh_key=ssh_key,
    )

    cmd = f"scp -r "
    if ssh_key is not None:
        cmd += f"-i {ssh_key} "
    cmd += (
        "-o StrictHostKeyChecking=no "
        + f"ubuntu@{machine}:~/predicators/results.tar.gz {temp_dir}/results.tar.gz"
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
    for item in os.listdir(f"{temp_dir}/results"):
        s = os.path.join(f"{temp_dir}/results", item)
        d = os.path.join(local_save_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    print("Download and extraction completed successfully.")


if __name__ == "__main__":
    _main()
