"""Compatibility import; edit the calculations in notebooks/08_forward_risk_tests.ipynb."""
from quiet_index_500.notebook_source import load_notebook_module as _load_notebook_module

_load_notebook_module(globals(), 'quiet_index_500.forward_tests')

if __name__ == "__main__":
    print(build_forward_tests(), flush=True)

