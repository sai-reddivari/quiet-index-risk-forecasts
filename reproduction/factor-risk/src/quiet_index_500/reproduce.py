"""One-command offline reproduction of the study from the frozen local inputs."""

import argparse
import sys
import time

import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation.manifests import sha256_of_file
from quiet_index_500.research_io import save_table

research_areas = {"factor_exposures", "downside_momentum", "size_liquidity", "panel",
                  "forward_tests", "covariance_evaluation", "review", "report"}


def research_table_hashes():
    """Scientific CSV hashes, excluding audit records that describe a run rather than data."""
    paths = []
    for path in settings.results_folder.rglob("*.csv"):
        relative = path.relative_to(settings.results_folder)
        area = relative.parts[1] if relative.parts[0] == "licensed" else relative.parts[0]
        if area in research_areas:
            paths.append(path)
    return {path.relative_to(settings.repository_root_folder).as_posix(): sha256_of_file(path) for path in sorted(paths)}


def reproduce_study(execute_notebooks=False, verify_existing=False):
    """Recompute every measurement/evaluation and compare exact scientific CSV bytes."""
    from quiet_index_500.foundation import build_all_processed_tables
    from quiet_index_500.factor_exposures import build_factor_exposures
    from quiet_index_500.downside_momentum import write_downside_momentum_outputs
    from quiet_index_500.size_liquidity import build_and_save
    from quiet_index_500.panel import build_characteristic_panel
    from quiet_index_500.forward_tests import build_forward_tests
    from quiet_index_500.covariance_evaluation import build_covariance_evaluation
    from quiet_index_500.event_sensitivity import build_event_sensitivity
    from quiet_index_500.audit import audit_research_outputs
    before = research_table_hashes()
    if verify_existing and not before:
        raise ValueError("verification requires preexisting research outputs")
    stages = [("foundation", build_all_processed_tables), ("factors", build_factor_exposures),
              ("downside/momentum", write_downside_momentum_outputs), ("size/liquidity", build_and_save),
              ("integration", build_characteristic_panel), ("forward tests", build_forward_tests),
              ("covariance", build_covariance_evaluation), ("event sensitivity", build_event_sensitivity)]
    for label, function in stages:
        started = time.monotonic()
        print(f"Reproducing {label}", flush=True)
        function()
        print(f"Completed {label} in {time.monotonic() - started:.1f}s", flush=True)
    from quiet_index_500.reporting import build_report_tables_and_figures
    build_report_tables_and_figures()
    if execute_notebooks:
        from quiet_index_500.notebook_runner import execute_notebook
        for name in ["03b_foundation_validation", "04_factor_exposures", "05_downside_momentum", "06_size_liquidity",
                     "07_characteristic_panel", "08_forward_risk_tests", "09_covariance_evaluation"]:
            execute_notebook(settings.repository_root_folder / "notebooks" / f"{name}.ipynb")
    after = research_table_hashes()
    comparison = pd.DataFrame([{"file": path, "before_sha256": before.get(path, ""), "after_sha256": after.get(path, ""),
                                "identical": before.get(path) == after.get(path)} for path in sorted(set(before) | set(after))])
    save_table(comparison, "audit", "reproduction.csv", source_modules=[sys.modules[__name__]],
               note="Exact scientific CSV bytes before and after a full rebuild. Manifest timestamps and audit records are excluded.")
    if verify_existing and not comparison.identical.all():
        raise AssertionError("scientific CSV reproduction differs: " + ", ".join(comparison.loc[~comparison.identical, "file"]))
    audit_research_outputs()
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    table = reproduce_study(args.notebooks, args.verify)
    print(f"Rebuilt {len(table)} scientific tables; {int(table.identical.sum())} byte-identical to pre-run outputs.")
