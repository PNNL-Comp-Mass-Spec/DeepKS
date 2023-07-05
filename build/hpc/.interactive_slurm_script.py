import os
import argparse
import traceback

# Defaults
DEF_CPU = 4
DEF_TIME = 180
DEF_ACCT = "ppi_concerto"
DEF_SHELL = "/bin/zsh"
DEF_PARTN = "a100_shared"

def main():
	try:
		ap = argparse.ArgumentParser()
		ap.add_argument('-v', '--verbose', action='store_true', required=False, default=False)
		verbose = ap.parse_args().verbose
	except Exception as e:
		print("Error with arguments of `interactive_slurm_script.py:`", e)
	
	try:
		cpus = input(f"# CPUs? (RETURN for default = {DEF_CPU}): ")
		if cpus == "": 
			cpus = DEF_CPU
		else:
			cpus = int(cpus)
		exclude = ""
		partition = input(f"Partition? (RETURN for default = {DEF_PARTN} OR '<partition> --E <exlude nodelist>' to exclude certain nodes: ")
		if partition == "":
			partition = f"-p {DEF_PARTN}"
		elif " --E " in partition:
			partition_ = f"-p {partition.split(' --E ')[0]}"
			exclude = f"--exclude={partition.split(' --E ')[1]}" # input("Node List? (e.g., `a100-05` or `a100-[04-06]` -- without the backticks): ")
			partition = partition_
		else:
			partition = f"-p {partition}"
			
		
		runtime = input(f"Allocated Time (minutes)? (RETURN for default = {DEF_TIME}): " )
		if runtime == "":
			runtime = DEF_TIME
		else:
			runtime = int(runtime)

		account = input(f"Account? (RETURN for default = {DEF_ACCT}): ")
		if account == "":
			account = DEF_ACCT

		shell = input(f"Shell? (RETURN for default = {DEF_SHELL}): ")
		if shell == "":
			shell = DEF_SHELL
		
		cmd = f"srun -A {account} {partition} {exclude} --time={runtime} -n {cpus} -N 1 --pty -u {shell}"
		if verbose:
			input(f"Slurm command about to be run:\n{cmd}\nPress RETURN to execute (ctrl+c to quit).\n") 
		os.system(cmd)
	
	except Exception as e:
		print("Error with `interactive_slurm_script.py`:", e)
		raise RuntimeError()
		
if __name__ == "__main__":
	main()
