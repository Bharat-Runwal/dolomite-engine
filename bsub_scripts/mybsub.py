"""
Example usage:

> python mybsub.py --interactive --num_gpus 2 --exclusive
> python mybsub.py --command "python train.py"
"""
import tyro
from dataclasses import dataclass
from typing import Optional, List
import subprocess

@dataclass
class BsubArgs:
    """Arguments for bsub command"""
    num_gpus: int = 1  # Number of GPUs to request
    interactive: bool = False  # Whether to run in interactive mode
    exclusive: bool = True  # Whether to request exclusive GPU access
    nodes: int = 1  # Number of nodes to request
    command: Optional[str] = None  # The command to run. Can only be `none` if also interactive
    run: bool = False  # By default, just print the generated bsub command.

    def __post_init__(self):
        if self.command is None and not self.interactive:
            raise ValueError("command must be provided if not running in interactive mode")

def main(args: BsubArgs) -> None:
    # Build the bsub command
    cmd = ["bsub"]
    
    # Add core request
    cmd.extend(["-n", str(args.nodes)])
    
    # Add interactive flag if requested
    if args.interactive:
        cmd.append("-Is")

    if args.command is None and args.interactive:
        final_cmd = "/bin/bash"
    elif args.command is not None:
        final_cmd = args.command
    else:
        raise ValueError("command must be provided if not running in interactive mode")
    
    # Add GPU configuration if requested
    # """bsub -Is -n 1 -gpu \"num=2/task:mode=exclusive_process\" /bin/bash"""
    if args.num_gpus > 0:
        gpu_string = f"num={args.num_gpus}"
        if args.exclusive:
            gpu_string += "/task:mode=exclusive_process"
        cmd.extend(["-gpu", f"\"{gpu_string}\""])
    
    # Add the actual command
    cmd.extend([final_cmd])
    
    # Execute the command
    final_cmd = " ".join(cmd)
    print(final_cmd)

    if args.run:
        subprocess.run(final_cmd, shell=True)

if __name__ == "__main__":
    args = tyro.cli(BsubArgs)
    main(args)