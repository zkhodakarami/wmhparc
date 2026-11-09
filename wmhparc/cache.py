"""Download cache with SHA-256 pinning.

All external resources (atlases, templates) are fetched on demand into
``$WMHPARC_HOME`` (default ``~/.cache/wmhparc``).  Every file we know is pinned
by its SHA-256 so that a changed upstream file is detected rather than silently
used.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


def cache_dir() -> Path:
    """Return (and create) the cache directory."""
    d = Path(os.environ.get("WMHPARC_HOME", Path.home() / ".cache" / "wmhparc"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_file(url: str, relpath: str, sha256: Optional[str] = None, force: bool = False,
               quiet: bool = False) -> Path:
    """Download ``url`` into the cache under ``relpath`` unless already present.

    Parameters
    ----------
    url : str
        Source URL.
    relpath : str
        Destination relative to :func:`cache_dir`.
    sha256 : str, optional
        Expected digest.  A mismatch raises ``RuntimeError`` and the file is
        removed, so a silently updated upstream file never gets used.
    force : bool
        Re-download even if present.
    """
    dest = cache_dir() / relpath
    if dest.exists() and not force:
        if sha256 and sha256sum(dest) != sha256:
            raise RuntimeError(
                f"{dest} exists but its SHA-256 does not match the pinned value. "
                "Delete it (or pass force=True) to re-download.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[wmhparc] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "wmhparc/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, \
            tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        shutil.copyfileobj(r, tmp)
        tmp_path = Path(tmp.name)
    if sha256:
        got = sha256sum(tmp_path)
        if got != sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {url}: expected {sha256}, got {got}")
    os.replace(tmp_path, dest)
    return dest
