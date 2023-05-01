import tqdm
from termcolor import colored

from ..config.logging import get_logger

logger = get_logger()


class CustomTqdm(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
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

        super().__init__(*args, **kwargs_to_use)


if __name__ == "__main__":
    import time

    for i in CustomTqdm(range(100), desc="Test Description"):
        tqdm.tqdm.write(f"i is {i}")
        time.sleep(0.05)
