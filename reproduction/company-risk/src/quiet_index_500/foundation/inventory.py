"""Inventory of the files left outside the study: the 2026-10-13 price pull in data/raw.

They are listed for the validation report (path, size, sha256) and left in place. Nothing else in the package opens
them; hashing reads their bytes but never parses them.
"""

import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation.manifests import repository_relative_path, sha256_of_file

leftover_vintage_marker = "2026-10-13"  # the pull after the frozen 2026-10-12 vintage


def inventory_leftover_files(folder=None, marker=leftover_vintage_marker):
    """One row per file in folder (default data/raw) whose name contains the marker: path, bytes, sha256."""
    folder = settings.raw_data_folder if folder is None else folder
    leftover_paths = sorted(file_path for file_path in folder.iterdir() if file_path.is_file() and marker in file_path.name)
    return pd.DataFrame(
        [{"path": repository_relative_path(file_path), "bytes": file_path.stat().st_size, "sha256": sha256_of_file(file_path)}
         for file_path in leftover_paths],
        columns=["path", "bytes", "sha256"],
    )
