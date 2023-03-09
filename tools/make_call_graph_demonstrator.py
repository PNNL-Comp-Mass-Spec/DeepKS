def dummy_fn():
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, os, sys, pathlib, json, pickle, tqdm, itertools, collections, random, warnings, pprint, plotly, plotly.express
    from matplotlib import rcParams
    rcParams['font.family'] = "P052"

    def dummy_inner():
        def dummy_inner_II():
            print("Dummy Inner II")
            [x for x in range(1, 10, 2)]
            {a:b for a, b in zip(range(10), range(10))}
        print("Dummy Inner -- Outer Module", re.sub(r"<module '(.*)' from.*", r"\1", str(sys.modules[__name__])).split(".")[0])
        dummy_inner_II()
        {x for x in range(20, 30, 5)}
    
    dummy_inner()