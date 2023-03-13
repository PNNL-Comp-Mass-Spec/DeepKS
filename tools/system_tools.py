import tempfile, subprocess, time, re

def os_system_and_get_stdout(cmd, prepend='', shell = 'zsh'):
    """
    cmd: command to run

    NOTE: output line must contain `[@python_capture_output]` to be presented in output
    """
    TAG = "[@python_capture_output]"

    with tempfile.TemporaryFile() as tf:
        shell_cmd = f"{shell} -c 'source ~/.zshrc && {cmd}'"
        with subprocess.Popen(shell_cmd, shell=True, stdout=tf, stderr=tf) as p:
            where = 0
            res_out = []
            while (status := p.poll()) is None:
                time.sleep(0.01)
                tf.seek(where)
                o = tf.read()
                where += len(o)
                ol = o.decode("UTF-8").split("\n")
                for line in ol:
                    if line != "" and TAG not in line:
                        print(prepend+line, flush=True)
                    elif TAG in line:
                        res_out.append(re.sub(f"({TAG})".replace("[", r"\[").replace("]", r"\]"), "", line))
    
    if status:
        print("Status:", status)
        raise Exception(f"Error running system command: {shell_cmd}")
    return res_out
