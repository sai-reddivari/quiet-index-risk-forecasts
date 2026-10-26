"""Shared provenance and notebook helpers for the research stages."""

from pathlib import Path
import sys

import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import loader
from quiet_index_500.foundation.manifests import results_table_path, write_manifest


def study_input_paths(include_market_cap=False):
    """Exact frozen study dependencies, including corrections and security rules."""
    paths = [loader.universe_tickers_path(), loader.universe_weights_path(),
             loader.ivv_equity_holdings_path(), loader.adjusted_close_path(),
             loader.split_adjusted_close_path(), loader.volume_path(), loader.supplement_path(),
             loader.spy_prices_path(), loader.smh_prices_path(), loader.factors_path(),
             settings.processed_data_folder / f"same_security_rules_{settings.holdings_date}.csv",
             settings.corrections_folder / f"extreme_event_ledger_{settings.holdings_date}.csv"]
    if include_market_cap:
        paths.append(loader.market_cap_raw_path())
    return paths


def save_table(table, area, name, *, licensed=False, inputs=(), source_modules=(), note=""):
    """Save a deterministic CSV and full input/code provenance; licensed values stay ignored."""
    path = results_table_path(area, name, derived_from_licensed_data=licensed)
    table.to_csv(path, index=False, float_format="%.12g")
    foundation_sources = sorted((settings.repository_root_folder / "src/quiet_index_500/foundation").glob("*.py"))
    sources = [settings, sys.modules[__name__], *foundation_sources, *source_modules]
    unique_sources = list(dict.fromkeys(str(Path(item.__file__ if hasattr(item, "__file__") else item)) for item in sources))
    dependencies = list(dict.fromkeys([*study_input_paths(licensed), *inputs]))
    write_manifest(path, dependencies, "quiet_index_500 research pipeline", note=note,
                   source_modules=unique_sources,
                   extra_fields={"licensed_market_cap_dependency": licensed,
                                 "price_vintage": settings.price_vintage_last_session,
                                 "research_price_cutoff": settings.research_price_cutoff,
                                 "factor_vintage": settings.ff5_vintage})
    return path


def make_notebook(path, title, narrative, code_cells):
    """Create a notebook whose cells call package functions or display saved results."""
    import nbformat
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.cells = [nbformat.v4.new_markdown_cell(f"# {title}\n\n{narrative}")]
    notebook.cells.extend(nbformat.v4.new_code_cell(cell) for cell in code_cells)
    nbformat.write(notebook, Path(path))
    return Path(path)
