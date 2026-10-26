"""Compatibility import; edit the calculations in notebooks/04_factor_exposures.ipynb."""
from quiet_index_500.notebook_source import load_notebook_module as _load_notebook_module

_load_notebook_module(globals(), 'quiet_index_500.factor_exposures')

if __name__ == "__main__":
    build_factor_exposures()

