"""Provides the ability to change the model mode between aligned and non-aligned. Currently, this should not be touched."""

import os
from typing import Literal


def get_mode() -> Literal["alin", "no_alin"]:
    """Get the alignment mode of the model.

    Returns
    -------
        'alin' if the aligned version should be used, 'no_alin' if the aligned version should not be used.
    """
    wd = _prerun()
    with open("mode.cfg", "r") as r:
        r = r.read().strip()
    assert r == "no_alin" or r == "alin", f"Invalid mode: Mode must be either 'no_alin' or 'alin'. What I see is {r}"
    _postrun(wd)
    return r


def set_mode(mode: Literal["alin", "no_alin"]):
    """Set the alignment mode of the model.

    Parameters
    ----------
    mode :
        'alin' if the aligned version should be used, 'no_alin' if the aligned version should not be used.
    """
    wd = _prerun()
    assert mode in ["no_alin", "alin"], f"Invalid mode: Mode must be either 'no_alin' or 'alin'. What I see is {mode}"
    with open("mode.cfg", "w") as f:
        f.write(mode + "\n")
    _postrun(wd)


def _prerun() -> str:
    """Change the working directory to the directory of this file.

    Returns
    -------
        The old working directory.
    """
    old_wd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return old_wd


def _postrun(old_wd: str):
    """Change the working directory back to the old working directory.

    Parameters
    ----------
    old_wd : str
        The old working directory.
    """

    os.chdir(old_wd)
