import pathlib

parent = pathlib.Path(__file__).parent.resolve()


def write_splash(splash_file):
    splash_screen = open(f"{parent}/{splash_file}.splash", "r").read()
    print(splash_screen)
    print()
