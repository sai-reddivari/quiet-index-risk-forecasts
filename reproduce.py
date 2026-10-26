#!/usr/bin/env python3
"""Verify frozen inputs and execute selected notebooks in a new disposable copy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


PACKAGE = Path(__file__).resolve().parent
DEFAULTS = {"factor-risk": ["08", "09"], "calibration-covariance": ["06", "08"], "company-risk": ["03", "04", "05"]}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def within(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    result = root / part
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes input root: {relative}")
    return result


def verify_shipped(study: str, root: Path) -> list[dict]:
    manifest = PACKAGE / "reproduction/manifest.json"
    if not manifest.is_file():
        raise ValueError("Missing reproduction/manifest.json; the package is incomplete.")
    entries = [item for item in json.loads(manifest.read_text())["files"] if item["study"] == study]
    if not entries:
        raise ValueError(f"No shipped hashes for {study}")
    for item in entries:
        path = within(root, item["path"])
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError(f"Shipped file missing or changed: {study}/{item['path']}")
        if "bytes" in item and path.stat().st_size != item["bytes"]:
            raise ValueError(f"Shipped size mismatch: {study}/{item['path']}")
    return entries


def verify_backup(backup: Path, study: str) -> list[tuple[Path, dict]]:
    profile = "company-extension" if study == "company-risk" else "original"
    manifest = json.loads((backup / "manifest.json").read_text())
    if manifest.get("snapshot") != "equity-2026-10-12_treasury-2026-10-16":
        raise ValueError("Unexpected backup snapshot; use the documented pinned frozen bundle.")
    selected = [item for item in manifest["files"] if profile in item["profiles"]]
    if len(selected) != (10 if study == "company-risk" else 9):
        raise ValueError(f"Incomplete {profile} backup profile")
    checked = []
    for item in selected:
        source = within(backup / "frozen", item["path"])
        if not source.is_file() or digest(source) != item["sha256"]:
            raise ValueError(f"Frozen input missing or changed: {item['path']}")
        if source.stat().st_size != item["bytes"]:
            raise ValueError(f"Frozen input size mismatch: {item['path']}")
        checked.append((source, item))
    return checked


def select_notebooks(root: Path, study: str, prefixes: list[str]) -> list[Path]:
    directory = root / ("extensions/company_risk/notebooks" if study == "company-risk" else "notebooks")
    selected = []
    for prefix in prefixes:
        if not re.fullmatch(r"\d{2}[a-z]?", prefix):
            raise ValueError(f"Notebook selectors must be numeric prefixes such as 03 or 03b: {prefix}")
        if (study == "factor-risk" and prefix in {"01", "02", "03"}) or (study == "company-risk" and prefix in {"00", "01", "02"}):
            raise ValueError(f"Acquisition/measurement notebooks are not enabled by this runner: {study}/{prefix}")
        matches = sorted(directory.glob(prefix + "_*.ipynb"))
        if len(matches) != 1:
            raise ValueError(f"Expected one shipped {study} notebook for {prefix}; found {len(matches)}")
        if matches[0] in selected:
            raise ValueError(f"Repeated notebook selector: {prefix}")
        selected.append(matches[0])
    return selected


OFFLINE_GUARD = '''# Added by reproduce.py in the disposable copy only.
import sys as _runner_sys
def _runner_offline(event, args):
    if event == 'socket.getaddrinfo':
        host = args[0]
        if host not in ('localhost', '127.0.0.1', '::1', None):
            raise RuntimeError('Reproduction runner blocks network acquisition')
    if event == 'socket.connect':
        address = args[1]
        if isinstance(address, tuple) and address[0] not in ('localhost', '127.0.0.1', '::1'):
            raise RuntimeError('Reproduction runner blocks network acquisition')
_runner_sys.addaudithook(_runner_offline)
'''


def execute(root: Path, notebooks: list[Path], timeout: int, report: dict) -> None:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), MPLBACKEND="Agg", MPLCONFIGDIR=str(root / ".matplotlib"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    (root / ".matplotlib").mkdir(exist_ok=True)
    (root / "notebooks").mkdir(exist_ok=True)
    if (root / "extensions/company_risk").is_dir():
        (root / "extensions/company_risk/results").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reproduce-kernel-") as temp:
        kernels = Path(temp)
        (kernels / "python3").mkdir()
        (kernels / "python3/kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Frozen research reproduction", "language": "python"}))
        for path in notebooks:
            relative = path.relative_to(root)
            if report["study"] == "factor-risk" and path.name[:2] in {"08", "09"}:
                module, function = {
                    "08": ("quiet_index_500.forward_tests", "build_forward_tests"),
                    "09": ("quiet_index_500.covariance_evaluation", "build_covariance_evaluation"),
                }[path.name[:2]]
                print(f"Recomputing {module}.{function} before notebook display", flush=True)
                program = OFFLINE_GUARD + f"\nfrom {module} import {function}\n{function}()\n"
                subprocess.run([sys.executable, "-c", program], cwd=str(root), env=env, check=True, timeout=timeout)
                report.setdefault("completed_steps", []).append(f"{module}.{function}")
                (root / "reproduction-run.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Executing {relative}", flush=True)
            notebook = nbformat.read(path, as_version=4)
            notebook.cells.insert(0, nbformat.v4.new_code_cell(OFFLINE_GUARD))
            notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Frozen research reproduction", "language": "python"}
            manager = KernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
            working_directory = root / "notebooks"
            client = NotebookClient(notebook, km=manager, timeout=timeout, resources={"metadata": {"path": str(working_directory)}})
            try:
                client.execute(cwd=str(working_directory), env=env)
            finally:
                nbformat.write(notebook, path)
            report["completed_notebooks"].append(str(relative))
            (root / "reproduction-run.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(DEFAULTS), default="company-risk")
    parser.add_argument("--notebooks", nargs="+", help="Ordered notebook prefixes; defaults depend on study")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--input-backup", type=Path, help="Authorized private bundle containing manifest.json and frozen/")
    parser.add_argument("--check", "--verify-only", dest="check", action="store_true", help="Verify only; do not copy or execute")
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per cell (default: 1800)")
    args = parser.parse_args()
    destination = None
    report = None
    try:
        root = PACKAGE / "reproduction" / args.study
        shipped = verify_shipped(args.study, root)
        notebooks = select_notebooks(root, args.study, args.notebooks or DEFAULTS[args.study])
        overlay = verify_backup(args.input_backup.resolve(), args.study) if args.input_backup else []
        needs_private = args.study != "company-risk" or any(path.name.startswith("06_") for path in notebooks)
        if needs_private and not overlay:
            raise ValueError("These notebooks require the private frozen overlay; supply --input-backup PATH.")
        print(f"Verified {len(shipped)} shipped files and {len(overlay)} private frozen inputs.", flush=True)
        if args.check:
            print("Verification only; nothing copied or executed.")
            return 0
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if not args.output:
            raise ValueError("Execution requires --output pointing to a new directory")
        destination = args.output.expanduser().resolve()
        if destination.exists() or destination.is_relative_to(PACKAGE):
            raise ValueError("--output must be a new directory outside the shipped package")
        for module in ["nbformat", "nbclient", "jupyter_client", "ipykernel"]:
            if importlib.util.find_spec(module) is None:
                raise ValueError(f"Missing {module}; install the dependencies using REPRODUCE.md")
        for _, item in overlay:
            target = within(root, item["path"])
            if target.exists() and digest(target) != item["sha256"]:
                raise ValueError(f"Private overlay conflicts with shipped input: {item['path']}")
        shutil.copytree(root, destination)
        for source, item in overlay:
            target = within(destination, item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with source.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            if digest(target) != item["sha256"]:
                raise ValueError(f"Copied input checksum mismatch: {item['path']}")
        report = {"study": args.study, "status": "running", "python": sys.executable,
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "completed_notebooks": [], "completed_steps": [], "shipped_files_verified": len(shipped),
                  "private_inputs": [{"path": item["path"], "sha256": item["sha256"]} for _, item in overlay]}
        execute(destination, [destination / path.relative_to(root) for path in notebooks], args.timeout, report)
        report["status"] = "completed"
        (destination / "reproduction-run.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Completed in {destination}")
        return 0
    except Exception as exc:
        if destination is not None and report is not None:
            report.update(status="failed", error=str(exc))
            (destination / "reproduction-run.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Reproduction stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
