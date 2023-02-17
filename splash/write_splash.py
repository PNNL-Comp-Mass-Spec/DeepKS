import os, pathlib
old_dir = os.getcwd()

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

def write_splash(splash_file):
    splash_screen = open(f"{splash_file}.splash", "r").read()
    print(splash_screen)
    print()
    os.chdir(old_dir)