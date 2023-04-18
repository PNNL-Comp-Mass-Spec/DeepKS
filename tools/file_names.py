import re, datetime, dateutil.tz, os


def get(directory="", prefix="", suffix="", prefix_sep="_", suffix_sep=".", win_compat=True):
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
