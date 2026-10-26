"""Compatibility import; edit the calculations in notebooks/09_covariance_evaluation.ipynb."""
from quiet_index_500.notebook_source import load_notebook_module as _load_notebook_module

_load_notebook_module(globals(), 'quiet_index_500.publication')

if __name__ == "__main__":
    print(build_publication())

