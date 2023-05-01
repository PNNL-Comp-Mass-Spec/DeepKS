Doc links: [Good Gist](https://gist.github.com/bskinn/0e164963428d4b51017cebdb6cda5209)

- python: https://docs.python.org/3/objects.inv
- numpy: https://numpy.org/doc/stable/objects.inv
- torch: https://pytorch.org/docs/stable/objects.inv
- pandas: https://pandas.pydata.org/docs/objects.inv
- matplotlib: https://matplotlib.org/objects.inv
- tqdm: https://tqdm.github.io/docs/objects.inv
- dill: https://dill.readthedocs.io/en/latest/objects.inv
- cloudpickle: ?
- scipy: https://docs.scipy.org/doc/scipy/reference/objects.inv
- termcolor: ?
- scikit_learn: https://scikit-learn.org/stable/objects.inv

Recipie Script: 
```bash
pydoctor --docformat numpy --project-name <project name> --intersphinx https://docs.python.org/3/objects.inv --intersphinx https://numpy.org/doc/stable/objects.inv --intersphinx https://pytorch.org/docs/stable/objects.inv --intersphinx https://pandas.pydata.org/docs/objects.inv --intersphinx https://matplotlib.org/objects.inv --intersphinx https://tqdm.github.io/docs/objects.inv --intersphinx https://dill.readthedocs.io/en/latest/objects.inv --intersphinx https://docs.scipy.org/doc/scipy/reference/objects.inv --intersphinx https://scikit-learn.org/stable/objects.inv <module>
```