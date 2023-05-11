#!/bin/zsh
rm -rf docs/api_pydoctor_docs/
cd ..
pydoctor \
    --docformat numpy \
    --project-name DeepKS \
    --intersphinx https://docs.python.org/3/objects.inv \
    --intersphinx https://numpy.org/doc/stable/objects.inv \
    --intersphinx https://pytorch.org/docs/stable/objects.inv \
    --intersphinx https://pandas.pydata.org/docs/objects.inv \
    --intersphinx https://matplotlib.org/objects.inv \
    --intersphinx https://dill.readthedocs.io/en/latest/objects.inv \
    --intersphinx https://docs.scipy.org/doc/scipy/reference/objects.inv \
    --intersphinx https://scikit-learn.org/stable/objects.inv \
    --intersphinx https://docs.aiohttp.org/en/stable/objects.inv \
    --intersphinx https://plotly.com/python-api-reference/objects.inv \
    --intersphinx https://seaborn.pydata.org/objects.inv \
    --intersphinx https://coverage.readthedocs.io/en/coverage-5.5/objects.inv \
    --html-output DeepKS/docs/api_pydoctor_docs \
    --template-dir DeepKS/docs/no_timestamp_templates DeepKS
