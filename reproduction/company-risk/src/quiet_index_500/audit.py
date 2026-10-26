"""Final frozen-input, table-manifest and artifact inventory checks."""

from pathlib import Path
import sys

import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation.manifests import sha256_of_file
from quiet_index_500.foundation.validation import manifest_completeness
from quiet_index_500.research_io import save_table


def artifact_inventory():
    """Fingerprint research artifacts and input manifests; never inspect local credentials."""
    root = settings.repository_root_folder
    paths = set()
    for folder in ["src", "tests", "docs", "notebooks", "data/processed", "data/corrections", "results"]:
        for path in (root / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or ".ipynb_checkpoints" in path.parts:
                continue
            if path.suffix not in {".py", ".md", ".csv", ".json", ".ipynb", ".png", ".svg", ".html"}:
                continue
            if root / "results/audit" in path.parents:
                continue  # the inventory cannot fingerprint itself
            paths.add(path)
    paths.update((root / "data/raw").glob("*2026-10-12.manifest.json"))
    paths.update(root / name for name in ["README.md", "pyproject.toml", "plan.md", "opus-handoff.md"]
                 if (root / name).is_file())
    return pd.DataFrame([{"file": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256_of_file(path)} for path in sorted(paths)])


def audit_research_outputs():
    """Require every research table's recorded data and source hashes to match."""
    audit_folder = settings.results_folder / "audit"
    skip = list(audit_folder.glob("*.csv")) if audit_folder.exists() else []
    checks = manifest_completeness(skip_tables=skip)
    if not checks.complete.all():
        failures = checks.loc[~checks.complete, ["table", "table_sha256_matches", "inputs_sha256_all_match", "source_code_sha256_all_match"]]
        raise AssertionError(f"stale or missing research manifests:\n{failures.to_string(index=False)}")
    checks_path = save_table(checks, "audit", "manifest_checks.csv", source_modules=[sys.modules[__name__]],
                            note="Every non-audit CSV under processed data, corrections and results; source, input and output fingerprints required.")
    inventory = artifact_inventory()
    inventory_path = save_table(inventory, "audit", "artifact_inventory.csv", source_modules=[sys.modules[__name__]],
                               note="All package/tests/docs/notebooks/processed/research artifacts excluding audit itself; raw price downloads remain local and are identified by their manifests.")
    import importlib.metadata
    import platform
    versions = [{"package": "python", "version": platform.python_version()}, {"package": "platform", "version": platform.platform()}]
    for name in ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "nbclient", "nbformat", "pytest"]:
        versions.append({"package": name, "version": importlib.metadata.version(name)})
    save_table(pd.DataFrame(versions), "audit", "runtime_environment.csv", source_modules=[sys.modules[__name__]])
    return {"manifest_checks": checks_path, "artifact_inventory": inventory_path}


if __name__ == "__main__":
    print(audit_research_outputs())
