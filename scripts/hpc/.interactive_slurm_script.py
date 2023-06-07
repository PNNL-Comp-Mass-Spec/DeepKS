import os
import argparse
import traceback

def main():
	try:
		ap = argparse.ArgumentParser()
		ap.add_argument('-v', '--verbose', action='store_true', required=False, default=False)
		verbose = ap.parse_args().verbose
	except Exception as e:
		print("Error with arguments of `interactive_slurm_script.py:`", e)
	
	try:
		cpus = input("# CPUs? (RETURN for default = 4): ")
		if cpus == "": 
			cpus = 4
		else:
			cpus = int(cpus)
		exclude = ""
		partition = input("Partition? (RETURN for default = a100_shared OR '<partition> --E <exlude nodelist>' to exclude certain nodes: ")
		if partition == "":
			partition = "-p a100_shared"
		elif " --E " in partition:
			partition_ = f"-p {partition.split(' --E ')[0]}"
			exclude = f"--exclude={partition.split(' --E ')[1]}" # input("Node List? (e.g., `a100-05` or `a100-[04-06]` -- without the backticks): ")
			partition = partition_
		else:
			partition = f"-p {partition}"
			
		
		runtime = input("Allocated Time (minutes)? (RETURN for default = 90): " )
		if runtime == "":
			runtime = 90
		else:
			runtime = int(runtime)

		account = input("Account? (RETURN for default = mq_dance): ")
		if account == "":
			account = "mq_dance"

		shell = input("Shell? (RETURN for default = /bin/zsh): ")
		if shell == "":
			shell = "/bin/zsh"
		
		cmd = f"srun -A {account} {partition} {exclude} --time={runtime} -n {cpus} -N 1 --pty -u {shell}"
		if verbose:
			input(f"Slurm command about to be run:\n{cmd}\nPress RETURN to execute (ctrl+c to quit).\n") 
		os.system(cmd)
	
	except Exception as e:
		print("Error with `interactive_slurm_script.py`:", e)
		raise RuntimeError()
		
if __name__ == "__main__": # pragma: no cover
	main()
