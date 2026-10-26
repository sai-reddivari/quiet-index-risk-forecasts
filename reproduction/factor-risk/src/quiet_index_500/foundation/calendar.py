"""Compatibility import; edit the calculations in notebooks/03_factors.ipynb."""
from quiet_index_500.notebook_source import load_notebook_module as _load_notebook_module

_load_notebook_module(globals(), 'quiet_index_500.foundation.calendar')
