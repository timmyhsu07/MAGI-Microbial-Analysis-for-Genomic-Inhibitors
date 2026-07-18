"""Capture tool / database / dependency versions for the run manifest.

Reproducibility requires recording *exactly* what produced a feature matrix.
This module gathers, defensively, the AMRFinderPlus software version, the
AMRFinderPlus database version, the BLAST+ and HMMER versions AMRFinderPlus
depends on, and the Python library versions used to build the outputs.

Every probe is best-effort: a missing tool yields ``"unknown"`` (or a short
error string) rather than crashing, so version capture never blocks a run.
The one exception is enforced elsewhere -- the pipeline can be configured to
*require* a resolvable AMRFinderPlus version and fail loudly if absent.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

_TIMEOUT = 60


class VersionError(RuntimeError):
    """Raised when a required version could not be determined."""


def _run(args: list[str]) -> str | None:
    """Run a command, return combined stdout/stderr text, or None on failure."""
    if shutil.which(args[0]) is None and not Path(args[0]).exists():
        return None
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "") + (proc.stderr or "")
    return out.strip() or None


def _first_version_token(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"\d+\.\d+(?:\.\d+)*", text)
    return m.group(0) if m else text.splitlines()[0].strip()


def amrfinder_software_version(amrfinder_bin: str) -> str | None:
    """AMRFinderPlus software version via ``amrfinder --version``."""
    return _first_version_token(_run([amrfinder_bin, "--version"]))


def amrfinder_db_version(amrfinder_bin: str, database_dir: str | None) -> str | None:
    """AMRFinderPlus database version.

    Preference order:
    1. A ``version.txt`` inside the database directory (authoritative and does
       not require running the tool).
    2. ``amrfinder --database_version`` if the flag is supported.
    3. ``None`` if neither is available.
    """
    if database_dir:
        vfile = Path(database_dir) / "version.txt"
        if vfile.exists():
            try:
                return vfile.read_text(encoding="utf-8").strip() or None
            except OSError:
                pass
    out = _run([amrfinder_bin, "--database_version"])
    return _first_version_token(out)


def _blast_version(binary: str) -> str | None:
    return _first_version_token(_run([binary, "-version"]))


def _hmmer_version() -> str | None:
    out = _run(["hmmsearch", "-h"])
    if not out:
        return None
    for line in out.splitlines():
        if line.startswith("# HMMER"):
            return _first_version_token(line)
    return _first_version_token(out)


def _py_pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def collect_versions(
    amrfinder_bin: str,
    database_dir: str | None,
    require_amrfinder: bool = False,
) -> dict[str, Any]:
    """Collect all tool/dep versions into a manifest-ready dict.

    If ``require_amrfinder`` is True and the software version cannot be
    resolved, raise :class:`VersionError` (used by the real pipeline; tests that
    mock annotation leave it False).
    """
    amr_ver = amrfinder_software_version(amrfinder_bin)
    if require_amrfinder and amr_ver is None:
        raise VersionError(
            f"Could not determine AMRFinderPlus version from '{amrfinder_bin} "
            "--version'. Is AMRFinderPlus installed and on PATH? See "
            "scripts/setup_amrfinder.sh."
        )

    db_ver = amrfinder_db_version(amrfinder_bin, database_dir)
    if require_amrfinder and db_ver is None:
        raise VersionError(
            "Could not determine AMRFinderPlus database version. Run "
            "'amrfinder --update' (or the setup script) to download the "
            "database, and/or set database_dir in the config."
        )

    return {
        "amrfinderplus": {
            "software_version": amr_ver or "unknown",
            "database_version": db_ver or "unknown",
            "binary": shutil.which(amrfinder_bin) or amrfinder_bin,
            "database_dir": database_dir,
        },
        "dependencies": {
            "blastn": _blast_version("blastn") or "unknown",
            "blastx": _blast_version("blastx") or "unknown",
            "hmmer": _hmmer_version() or "unknown",
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "pandas": _py_pkg_version("pandas"),
            "pyarrow": _py_pkg_version("pyarrow"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
