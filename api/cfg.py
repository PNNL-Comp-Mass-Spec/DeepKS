import json, os, pathlib

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

with open("default_paths.cfg.json") as f:
    default_paths = json.load(f)

PRE_TRAINED_NN = default_paths['PRE_TRAINED_NN']
PRE_TRAINED_GC = default_paths['PRE_TRAINED_GC']