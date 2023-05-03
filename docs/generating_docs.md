Doc links: [Good Gist](https://gist.github.com/bskinn/0e164963428d4b51017cebdb6cda5209)

- python: https://docs.python.org/3/objects.inv
- numpy: https://numpy.org/doc/stable/objects.inv
- torch: https://pytorch.org/docs/stable/objects.inv
- pandas: https://pandas.pydata.org/docs/objects.inv
- matplotlib: https://matplotlib.org/objects.inv
- tqdm: ?
- dill: https://dill.readthedocs.io/en/latest/objects.inv
- cloudpickle: ?
- scipy: https://docs.scipy.org/doc/scipy/reference/objects.inv
- termcolor: ?
- scikit_learn: https://scikit-learn.org/stable/objects.inv
- aiohttp: https://docs.aiohttp.org/en/stable/objects.inv
- plotly: https://plotly.com/python-api-reference/objects.inv
- parameterized: ?
- selenium: ?
- seaborn: https://seaborn.pydata.org/objects.inv
- jsonschema: ?
- coverage: https://coverage.readthedocs.io/en/coverage-5.5/objects.inv


Recipie Script: 
```bash
pydoctor --docformat numpy --project-name DeepKS --intersphinx https://docs.python.org/3/objects.inv --intersphinx https://numpy.org/doc/stable/objects.inv --intersphinx https://pytorch.org/docs/stable/objects.inv --intersphinx https://pandas.pydata.org/docs/objects.inv --intersphinx https://matplotlib.org/objects.inv --intersphinx https://dill.readthedocs.io/en/latest/objects.inv --intersphinx https://docs.scipy.org/doc/scipy/reference/objects.inv --intersphinx https://scikit-learn.org/stable/objects.inv --intersphinx https://docs.aiohttp.org/en/stable/objects.inv --intersphinx https://plotly.com/python-api-reference/objects.inv --intersphinx https://seaborn.pydata.org/objects.inv --intersphinx https://json-schema.org/draft/2020-12/schema/objects.inv --intersphinx https://coverage.readthedocs.io/en/coverage-5.5/objects.inv --html-output DeepKS/docs/api_pydoctor_docs --template-dir DeepKS/docs/no_timestamp_templates DeepKS
```

beautifultable
brokenaxes
cloudpickle ?
git+https://github.com/Ben-Drucker/psankey-modified@master#egg=psankey_modified
git+https://github.com/Ben-Drucker/roc_comparison-modified@main#egg=roc_comparison_modified
git+https://github.com/Ben-Drucker/torchinfo-modified@main#egg=torchinfo_modified
graphviz
html2text
jsonschema ?
kaleido
lxml
mlrose_hiive
more_itertools
nbformat
numpyencoder
openpyxl
parameterized
prettytable
psutil
pycallgraph2
python_dateutil
selenium ?
sigfig
termcolor
tqdm ?