"""Smoke tests for the cell2location command-line entry points.

`_cli.py` is loaded directly from its source path so the test can run in a CI matrix
where the surrounding cell2location package import would pull in heavy ML deps. The
CLI itself has no ML imports.
"""

from __future__ import annotations

import importlib.util
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

_CLI_PATH = Path(__file__).resolve().parents[1] / "cell2location" / "_cli.py"
_spec = importlib.util.spec_from_file_location("cell2location_cli_under_test", _CLI_PATH)
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = _cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_list_skills_finds_bundled_or_dev_fallback() -> None:
    code, stdout, stderr = _run("list-skills")
    if code != 0:
        pytest.skip(f"No bundled skills available: {stderr}")
    assert "spatial-mapping" in stdout
    assert "cell2location-context" in stdout
    assert "cell2location-troubleshooting" in stdout


def test_install_skills_dry_run(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        code, stdout, stderr = _run("install-skills", "--dry-run")
    if code != 0:
        pytest.skip(f"No bundled skills available: {stderr}")
    assert "cell2location-spatial-mapping" in stdout
    # Nothing should actually be written in dry-run mode.
    assert not (fake_home / ".claude" / "skills").exists() or not any((fake_home / ".claude" / "skills").iterdir())


def test_install_skills_copy_and_uninstall(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        code, _, err = _run("install-skills")
        if code != 0:
            pytest.skip(f"No bundled skills available: {err}")
        installed_root = fake_home / ".claude" / "skills"
        assert installed_root.is_dir()
        installed_dirs = sorted(p.name for p in installed_root.iterdir() if p.is_dir())
        assert any(name.startswith("cell2location-") for name in installed_dirs)

        # Second install without --force should be a no-op (every entry already there).
        code2, stdout2, _ = _run("install-skills")
        assert code2 == 0
        assert "Skipping" in stdout2 or "Nothing installed" in stdout2

        code3, _, _ = _run("uninstall-skills")
        assert code3 == 0
        remaining = [p for p in installed_root.iterdir() if p.name.startswith("cell2location-")]
        assert remaining == []
