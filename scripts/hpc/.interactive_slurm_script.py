import os, argparse

try:
    from termcolor import colored
except Exception:
    colored = lambda s, c: f"!!! {s} !!!"

# DEFAULTS
DEF_TASKS = 1
DEF_CPUS_PER_TASK = 4
DEF_NODES = 1
DEF_TIME = 180
DEF_ACCT = "ppi_concerto"
DEF_SHELL = "/bin/zsh"
DEF_PARTN = "a100_shared"


def main():
    verbose = False
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--verbose", action="store_true", required=False, default=False)
        verbose = ap.parse_args().verbose
    except Exception as e:
        print("Error with arguments of `interactive_slurm_script.py:`", e)

    try:
        # TASKS
        tasks = input(f"# Tasks? (RETURN for default = {DEF_TASKS}): ")
        if tasks == "":
            tasks = DEF_TASKS
        else:
            tasks = int(tasks)
        if tasks > 4:
            print(colored(f"WARNING: You are requesting a high number of tasks ({tasks})", "yellow"))

        # CPUS PER TASK
        cpus_per_task = input(
            f"# Cores per task for multiprocessing/threading? (RETURN for default = {DEF_CPUS_PER_TASK}): "
        )
        if cpus_per_task == "":
            cpus_per_task = DEF_CPUS_PER_TASK
        else:
            cpus_per_task = int(cpus_per_task)
        if cpus_per_task > 64:
            print(colored(f"WARNING: You are requesting a high number of cpus per task ({cpus_per_task})", "yellow"))

        # NODES
        nodes = input(f"# Nodes? (RETURN for defualt = {DEF_NODES}): ")
        if nodes == "":
            nodes = DEF_NODES
        else:
            nodes = int(nodes)
        if nodes > 1:
            print(
                colored(
                    f"WARNING: You are requesting more than one node ({nodes}). This is usually not required.", "yellow"
                )
            )

        # PARTITION & NODE EXCLUSIONS
        exclude = ""
        partition = input(
            f"Partition? (RETURN for default = {DEF_PARTN} OR `your_partition --E node1,node2,node3`) to exclude"
            " certain nodes: "
        )
        if partition == "":
            partition = DEF_PARTN
        else:
            partition = partition
        if " --E " in partition:
            exclude = partition.split(" --E ")[1]

        # TIME
        runtime = input(f"Allocated Time (minutes)? (RETURN for default = {DEF_TIME}): ")
        if runtime == "":
            runtime = DEF_TIME
        else:
            runtime = int(runtime)
        if runtime > 60 * 6:
            print(colored(f"WARNING: You are requesting a high runtime ({cpus_per_task})", "yellow"))

        # ACCOUNT
        account = input(f"Account? (RETURN for default = {DEF_ACCT}): ")
        if account == "":
            account = DEF_ACCT

        # SHELL
        shell = input(f"Shell? (RETURN for default = {DEF_SHELL}): ")
        if shell == "":
            shell = DEF_SHELL

        # PUTTING EVERYTHING TOGETHER
        cmds = [
            f"srun",
            f"--account {account}",
            f"--partition {partition}",
            f"{f'--exclude {exclude}' if exclude else ''}",
            f"--time {runtime}",
            f"--ntasks {tasks}",
            f"--nodes {nodes}",
            f"--cpus-per-task {cpus_per_task}",
            "--pty",
            "--unbuffered",
            f"{shell}"
        ]
        cmd = " ".join(cmds)
        if verbose:
            input(f"Slurm command about to be run:\n{cmd}\nPress RETURN to execute (ctrl+c to quit).\n")
        os.system(cmd)

    except Exception as e:
        print("Error with `interactive_slurm_script.py`:", e)
        raise RuntimeError()
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
