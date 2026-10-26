"""Compatibility import; edit the calculations in notebooks/06_size_liquidity.ipynb."""
from quiet_index_500.notebook_source import load_notebook_module as _load_notebook_module

_load_notebook_module(globals(), 'quiet_index_500.size_liquidity')

if __name__ == '__main__':
    built = build_and_save()
    create_notebook()
    print({name: len(table) for name, table in built.items()})

