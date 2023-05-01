"""Get a file name with a timestamp, with Windows/Onedrive name compatibility
"""

import re, datetime, dateutil.tz, os


def get(prefix="", suffix="", prefix_sep="_", suffix_sep=".", win_compat=True, directory="") -> str:
    """Get a file name with a timestamp, with Windows/Onedrive name compatibility

    Parameters
    ----------
    prefix : str, optional
        The prefix to add to the file name, by default ""
    suffix : str, optional
        The suffix (file extension) to add to the file name, by default ""
    prefix_sep : str, optional
        The separator between the prefix and the timestamp, by default "_"
    suffix_sep : str, optional
        The separator between the timestamp and the suffix (file extension), by default "."
    win_compat : bool, optional
        Whether to make the file name compatible with Windows/Onedrive, by default True
    directory : str, optional
        The directory to put the file in, by default ""

    Returns
    -------
    str
        The resultant file name

    """
    if prefix == "":
        prefix_sep = ""
    if suffix == "":
        suffix_sep = ""
    now = datetime.datetime.now()
    tz = datetime.datetime.now(tz=dateutil.tz.tzlocal()).strftime("%z")
    base = now.isoformat(timespec="milliseconds", sep="@")[:-2] + "@" + tz[:3] + ":" + tz[3:]
    full = f"{prefix}{prefix_sep}{base}{suffix_sep}{suffix}"
    if win_compat:
        # Make compatible with Windows
        full = re.sub(r"[\*<>\?\/\\\|~\"#%&\{\}]", "_", full)
    else:
        full = re.sub(r"[\\\/]", "_", full)
    full = re.sub(r":", "`", full)
    full = os.path.join(directory, full)
    return full
