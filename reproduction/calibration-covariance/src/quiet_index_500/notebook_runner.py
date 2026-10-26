"""Execute saved notebooks with the project's Python kernel, without network pulls."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from quiet_index_500 import settings


def execute_notebook(path):
    """Execute and save one existing notebook; all research logic stays in package calls."""
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(settings.repository_root_folder / "notebooks")}}).execute()
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    arguments = parser.parse_args()
    for notebook_path in arguments.paths:
        print(execute_notebook(notebook_path), flush=True)
