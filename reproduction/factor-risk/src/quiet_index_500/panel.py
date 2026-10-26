"""Compatibility import; edit the calculations in notebooks/07_characteristic_panel.ipynb."""
from quiet_index_500.notebook_source import load_notebook_module as _load_notebook_module

_load_notebook_module(globals(), 'quiet_index_500.panel')

if __name__ == "__main__":
    print(build_characteristic_panel())

