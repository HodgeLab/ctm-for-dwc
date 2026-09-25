"""Tests for :func:`ctm_for_dwc.pems.resolve_credentials`.

Every test points the ``.env`` lookup at ``tmp_path`` so the developer's real
repo-root ``.env`` is never read.
"""

from __future__ import annotations

import pytest

from ctm_for_dwc.pems import credentials
from ctm_for_dwc.pems.credentials import resolve_credentials


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No PEMS_* vars and an empty fake repo root; both restored on teardown."""
    for var in ("PEMS_USERNAME", "PEMS_PASSWORD"):
        monkeypatch.setenv(var, "placeholder")  # records the original state
        monkeypatch.delenv(var)
    monkeypatch.setattr(credentials, "_REPO_ROOT", tmp_path)
    return tmp_path


def test_explicit_arguments_win_over_environment(clean_env, monkeypatch):
    monkeypatch.setenv("PEMS_USERNAME", "env_user")
    monkeypatch.setenv("PEMS_PASSWORD", "env_pass")
    assert resolve_credentials("arg_user", "arg_pass") == ("arg_user", "arg_pass")


def test_environment_used_when_arguments_absent(clean_env, monkeypatch):
    monkeypatch.setenv("PEMS_USERNAME", "env_user")
    monkeypatch.setenv("PEMS_PASSWORD", "env_pass")
    assert resolve_credentials() == ("env_user", "env_pass")


def test_dotenv_loaded_when_environment_missing(clean_env):
    (clean_env / ".env").write_text("PEMS_USERNAME=file_user\nPEMS_PASSWORD=file_pass\n")
    assert resolve_credentials() == ("file_user", "file_pass")


def test_missing_credentials_raise(clean_env):
    with pytest.raises(RuntimeError, match="PeMS credentials not found"):
        resolve_credentials()
