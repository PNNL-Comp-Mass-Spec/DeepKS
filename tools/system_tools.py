import tempfile, subprocess, time, re

def os_system_and_get_stdout(cmd):
    """
    cmd: command to run

    NOTE: output line must contain `[@python_capture_output]` to be presented in output
    """
    TAG = "[@python_capture_output]"

    tf = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, shell=True, stdout=tf, stderr=tf)

    where = 0
    res_out = []
    while p.poll() is None:
        time.sleep(0.01)
        tf.seek(where)
        o = tf.read()
        where += len(o)
        ol = o.decode("UTF-8").split("\n")
        for line in ol:
            if line != "" and TAG not in line:
                print(line)
            elif TAG in line:
                res_out.append(re.sub(f"({TAG})".replace("[", r"\[").replace("]", r"\]"), "", line))
    
    return res_out
