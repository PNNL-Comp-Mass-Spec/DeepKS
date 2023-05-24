"""Utility that several other packages use that """

import os, pathlib
import warnings

"""The logger for this module."""


def join_first(path: str | None, levels: int, calling_from_file: str) -> str:
    """Helper function to join a target path to a pseudo-root path derived from the location of this file.

    Parameters
    ----------
    path :
        The target path. Should not be ``None``, but that possibility is included for API compatibility.
    levels : 
        How many directories out of the directory of ``calling_from_file`` the "new root" should start?
    calling_from_file :
        The file that is calling this function. Usually, this should be ``__file__``.

    Returns
    -------
        The joined path

    Raises
    ------
    ValueError
        If ``path`` is ``None``.

    Examples
    --------
    Suppose the file structure looks like the following:

    .. code:: plain

        /
        |---dir_A
        |   |---dir_A1
        |   |   |---file_A1.py
        |   |---dir_A2
        |       |---file_A2.py
        |---dir_B
            |---file_B.py
    
    Suppose ``file_A1.py`` needs an absolute path to ``file_B.py``. Then, the following code would work:

    .. code:: python

        from DeepKS.config.join_first import join_first
        path = join_first("dir_B/file_B.py", 2, __file__)
        print(path)
        /dir_A/dir_A1/../../dir_B/file_B.py

    The ``2`` represents the number of directories to go up from ``file_A1.py``'s directory. This puts us in ``/``. \
        We know ``dir_B`` is a child of ``/``, so we can just join it to the path. \
        The result is an absolute path to ``file_B.py``.
        
    """
    if path is None:
        raise ValueError("Path cannot be None.")
    if os.path.isabs(path):
        # warnings.warn(f"Path {path} is already absolute. Returning it as is.")
        return path
    else:
        return os.path.join(pathlib.Path(calling_from_file).parent.resolve(), *[".."] * levels, path)
