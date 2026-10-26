"""Manifests and fingerprints for every table the study writes.

A manifest sits beside its table as <table>.manifest.json and records what the table is, what built it, from which
input files (by sha256) and with which source code (by sha256), so any reported number can be traced to exact bytes.
Tables derived from licensed Financial Modeling Prep data go under results/licensed/<area>/, where the CSVs are
git-ignored and only the manifests are committed.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quiet_index_500 import settings


def sha256_of_file(file_path):
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def repository_relative_path(file_path):
    """Path relative to the repository root in posix form; a path outside the repository is returned absolute."""
    resolved_path = Path(file_path).resolve()
    try:
        return resolved_path.relative_to(settings.repository_root_folder).as_posix()
    except ValueError:  # a scratch or test location outside the repository
        return resolved_path.as_posix()


def manifest_path_for(table_path):
    """The manifest file that belongs beside a table: same name, .manifest.json in place of the extension."""
    return Path(table_path).with_suffix(".manifest.json")


def verify_file_against_manifest(file_path, manifest_path=None):
    """Check a file's sha256 against its manifest and return the fingerprint; raise if they differ.

    The manifest either records one file ("sha256" at the top level) or several ("files", each with "file" and "sha256"),
    as the raw price pull does.
    """
    file_path = Path(file_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else manifest_path_for(file_path)
    manifest = json.loads(manifest_path.read_text())
    if "files" in manifest:  # one manifest for several files: find this file's entry by name
        matching_entries = [entry for entry in manifest["files"] if Path(entry["file"]).name == file_path.name]
        if len(matching_entries) != 1:
            raise ValueError(f"{manifest_path.name} has no single entry for {file_path.name}")
        expected_sha256 = matching_entries[0]["sha256"]
    else:
        expected_sha256 = manifest["sha256"]
    actual_sha256 = sha256_of_file(file_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{file_path.name} does not match its manifest {manifest_path.name}: the file changed after it was recorded")
    return actual_sha256


def _source_file_path(source_module):
    """A module object or a path, as the path of its source file."""
    return Path(source_module.__file__) if hasattr(source_module, "__file__") else Path(source_module)


def write_manifest(table_path, input_paths, built_by, note="", source_modules=(), extra_fields=None):
    """Write <table>.manifest.json beside a CSV table and return the manifest path.

    Fields, in order: file, built_by, built_at_utc, inputs (path relative to the repository and sha256 of each),
    source_code (path and sha256 of each module used), sha256, bytes, rows, columns, note, then any extra_fields.
    rows and columns are read back from the written file, so they describe what is on disk.
    """
    table_path = Path(table_path)
    table_read_back = pd.read_csv(table_path)
    manifest = {
        "file": repository_relative_path(table_path),
        "built_by": built_by,
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": [{"file": repository_relative_path(input_path), "sha256": sha256_of_file(input_path)} for input_path in input_paths],
        "source_code": [{"file": repository_relative_path(_source_file_path(source_module)),
                         "sha256": sha256_of_file(_source_file_path(source_module))} for source_module in source_modules],
        "sha256": sha256_of_file(table_path),
        "bytes": table_path.stat().st_size,
        "rows": len(table_read_back),
        "columns": table_read_back.columns.tolist(),
        "note": note,
    }
    manifest.update(extra_fields or {})
    manifest_path = manifest_path_for(table_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def results_table_path(area, file_name, derived_from_licensed_data):
    """Where a study table goes: results/<area>/ for committable tables, results/licensed/<area>/ for FMP-derived ones.

    The folder is created if needed. Anything holding values derived from FMP market cap (cap levels, log cap, size
    percentile or quartile, quartile dollar boundaries, cap-weighted statistics) must pass derived_from_licensed_data=True.
    """
    area_folder = (settings.licensed_results_folder if derived_from_licensed_data else settings.results_folder) / area
    area_folder.mkdir(parents=True, exist_ok=True)
    return area_folder / file_name


def licensed_results_path(area, file_name):
    """results/licensed/<area>/<file_name>, for a table derived from licensed FMP data (CSV git-ignored, manifest committed)."""
    return results_table_path(area, file_name, derived_from_licensed_data=True)
