"""Implementation of a custom tqdm progress bar that works with logging."""

from typing import Iterable
import tqdm
from termcolor import colored

from ..config.logging import get_logger

logger = get_logger()
"""The logger for this module."""


class CustomTqdm(tqdm.tqdm):
    """A custom tqdm progress bar that works with logging."""

    def __init__(self, iterable: Iterable, **kwargs):
        """Initializes the custom tqdm progress bar.

        Parameters
        ----------
        iterable :
            The iterable to pass to the ``tqdm.tqdm`` progress bar original class.
        kwargs : optional
            The keyword arguments to pass to the ``tqdm.tqdm`` progress bar original class.
        """
        kwargs_to_use: dict = dict()
        kwargs_to_use = kwargs_to_use | kwargs
        kwargs_to_use = kwargs_to_use | dict(
            colour="green",
            ascii="░▒█",
            bar_format=colored("{l_bar}", "green") + "{bar:15}" + colored("{r_bar}", "green") + "\r",
        )
        if "position" not in kwargs:
            kwargs_to_use = kwargs_to_use | dict(position=6)

        # logger.debug(f"{kwargs_to_use=}")
        if logger._upper_level >= logger.PROGRESS >= logger._level:
            super().__init__(iterable, **kwargs_to_use)
        else:
            super().__init__(iterable, **kwargs_to_use, disable=True)


if __name__ == "__main__":  # pragma: no cover
    import time

    for i in CustomTqdm(range(100), desc="Test Description"):
        tqdm.tqdm.write(f"i is {i}")
        time.sleep(0.05)
