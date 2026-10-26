"""Import only explicitly marked calculation cells from the authored notebooks.

Collection, display and example cells are never executed by package imports.
Notebook code is the single source; these adapters preserve the existing API/CLI.
"""
import hashlib
import json
import linecache
from pathlib import Path
from quiet_index_500 import settings

NOTEBOOK_MODULES = {'quiet_index_500.foundation.same_security': 'notebooks/01_ivv_holdings.ipynb', 'quiet_index_500.foundation.loader': 'notebooks/02_prices.ipynb', 'quiet_index_500.foundation.build': 'notebooks/02_prices.ipynb', 'quiet_index_500.foundation.calendar': 'notebooks/03_factors.ipynb', 'quiet_index_500.foundation.corrections': 'notebooks/03b_foundation_validation.ipynb', 'quiet_index_500.foundation.eligibility': 'notebooks/03b_foundation_validation.ipynb', 'quiet_index_500.foundation.inventory': 'notebooks/03b_foundation_validation.ipynb', 'quiet_index_500.foundation.validation': 'notebooks/03b_foundation_validation.ipynb', 'quiet_index_500.event_sensitivity': 'notebooks/03b_foundation_validation.ipynb', 'quiet_index_500.factor_exposures': 'notebooks/04_factor_exposures.ipynb', 'quiet_index_500.downside_momentum': 'notebooks/05_downside_momentum.ipynb', 'quiet_index_500.size_liquidity': 'notebooks/06_size_liquidity.ipynb', 'quiet_index_500.panel': 'notebooks/07_characteristic_panel.ipynb', 'quiet_index_500.forward_tests': 'notebooks/08_forward_risk_tests.ipynb', 'quiet_index_500.covariance_evaluation': 'notebooks/09_covariance_evaluation.ipynb', 'quiet_index_500.reporting': 'notebooks/09_covariance_evaluation.ipynb', 'quiet_index_500.publication': 'notebooks/09_covariance_evaluation.ipynb'}


def calculation_cells(notebook_path, module_name):
    """Return ordered source cells for one module; missing or invalid source fails."""
    notebook = json.loads(Path(notebook_path).read_text())
    cells = [cell for cell in notebook['cells']
             if cell.get('cell_type') == 'code'
             and cell.get('metadata', {}).get('research_module') == module_name]
    if not cells:
        raise ValueError(f'No calculation cells for {module_name} in {notebook_path}')
    return [(cell.get('id', str(i)), ''.join(cell['source']) if isinstance(cell['source'], list)
             else cell['source']) for i, cell in enumerate(cells)]


def notebook_source_hash(notebook_path, module_name):
    """Hash ordered calculation code only; prose, outputs and counts are not code."""
    sources = [source for _, source in calculation_cells(notebook_path, module_name)]
    payload = json.dumps(sources, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def load_notebook_module(namespace, module_name):
    """Execute the saved definitions in the calling adapter's normal module namespace."""
    notebook_path = settings.repository_root_folder / NOTEBOOK_MODULES[module_name]
    for cell_id, source in calculation_cells(notebook_path, module_name):
        filename = str(notebook_path) + '#cell-' + cell_id
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        exec(compile(source, filename, 'exec'), namespace)
    namespace['__notebook_source__'] = notebook_path
