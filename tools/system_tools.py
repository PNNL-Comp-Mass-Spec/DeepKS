"""Utilities that interact with the system, e.g. run shell commands and get output."""

import tempfile, subprocess, time, re

from ..config.logging import get_logger

logger = get_logger()
"""Logger for this module"""


def os_system_and_get_stdout(cmd: str, prepend: str = "", shell: str = "zsh") -> list[str]:
    """Function to run a shell command and fetch the output.

    Parameters
    ----------
    cmd :
        The shell command to run.
    prepend : optional
        A string to prepend to each line of the output so we know where it came from.
    shell : optional
        The shell to use, `zsh` by default.

    Notes
    -----
    Output line must contain `[@python_capture_output]` to be presented in output


    Returns
    -------
        Lines of output from the shell command.
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
                        logger.info(prepend + line)
                    elif TAG in line:
                        res_out.append(re.sub(f"({TAG})".replace("[", r"\[").replace("]", r"\]"), "", line))

    if status:
        logger.uerror("Status:", status)
        logger.uerror(f"Error running system command: {shell_cmd}")
    return res_out
