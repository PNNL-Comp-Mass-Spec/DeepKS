import os

def get_mode():
    wd = _prerun()
    r = open("mode.cfg", "r").read().strip()
    assert r in ["no_alin", "alin"], f"Invalid mode: Mode must be either 'no_alin' or 'alin'. What I see is {r}"
    _postrun(wd)
    return r

def set_mode(mode):
    wd = _prerun()
    assert mode in ["no_alin", "alin"], f"Invalid mode: Mode must be either 'no_alin' or 'alin'. What I see is {mode}"
    open("mode.cfg", "w").write(mode+"\n")
    _postrun(wd)

def _prerun():
    old_wd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return old_wd

def _postrun(old_wd):
    os.chdir(old_wd)