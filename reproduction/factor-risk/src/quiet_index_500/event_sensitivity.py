"""Compatibility import; edit the calculations in notebooks/03b_foundation_validation.ipynb."""
from quiet_index_500.notebook_source import load_notebook_module as _load_notebook_module

_load_notebook_module(globals(), 'quiet_index_500.event_sensitivity')

if __name__ == '__main__':
    print(build_event_sensitivity())

