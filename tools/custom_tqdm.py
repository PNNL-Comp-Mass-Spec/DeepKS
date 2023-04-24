import tqdm
from termcolor import colored


class CustomTqdm(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                colour="green",
                ascii="░▒█",
                # ncols=os.get_terminal_size().columns // 2,
                bar_format=colored("{l_bar}", "green") + "{bar:15}" + colored("{r_bar}", "green") + "\r",
                position=4,
            )
        )
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    import time

    for i in CustomTqdm(range(100), desc="Test Description"):
        tqdm.tqdm.write(f"i is {i}")
        time.sleep(0.05)
