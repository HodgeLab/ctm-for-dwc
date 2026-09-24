"""Resolve PeMS clearinghouse credentials.

Explicit arguments win; otherwise ``PEMS_USERNAME`` / ``PEMS_PASSWORD`` are
read from the environment, loading the repo-root ``.env`` (via
``python-dotenv``) if they aren't already set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_credentials(
    username: Optional[str] = None, password: Optional[str] = None,
) -> tuple[str, str]:
    """Return ``(username, password)`` for the PeMS clearinghouse.

    Parameters
    ----------
    username, password : str, optional
        Explicit credentials; each falls back to its environment variable.

    Returns
    -------
    tuple[str, str]
        The resolved credentials.

    Raises
    ------
    RuntimeError
        If either credential is still missing after checking ``.env``.
    """
    username = username or os.environ.get("PEMS_USERNAME", "").strip() or None
    password = password or os.environ.get("PEMS_PASSWORD", "").strip() or None
    if not username or not password:
        # python-dotenv is in the project deps; try a one-shot .env read so
        # users running from the repo root get the credentials they put there.
        try:
            from dotenv import load_dotenv
            load_dotenv(_REPO_ROOT / ".env")
        except ImportError:
            pass
        username = username or os.environ.get("PEMS_USERNAME", "").strip() or None
        password = password or os.environ.get("PEMS_PASSWORD", "").strip() or None
    if not username or not password:
        raise RuntimeError(
            "PeMS credentials not found. Set PEMS_USERNAME / PEMS_PASSWORD in "
            "your environment (a .env file works) or pass them explicitly "
            "(--username / --password on the command line)."
        )
    return username, password
