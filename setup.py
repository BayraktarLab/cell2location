"""setup.py — mirrors the repo's .claude/skills/ tree into cell2location/_bundled_skills/
at build time so the skills travel inside the wheel.

The mirror runs unconditionally before setuptools.setup() so it covers every install
path: `pip install .`, `pip install -e .`, `python -m build`, `python setup.py
bdist_wheel`, `python setup.py sdist`. It is idempotent and a no-op when the source
.claude/skills/ tree is absent (e.g. when the user is installing from a published
wheel, in which case _bundled_skills/ already exists).
"""

import shutil
from pathlib import Path

import setuptools

_REPO_ROOT = Path(__file__).resolve().parent
_SOURCE_SKILLS = _REPO_ROOT / ".claude" / "skills"
_BUNDLED_TARGET = _REPO_ROOT / "cell2location" / "_bundled_skills"


def _mirror_bundled_skills() -> None:
    if not _SOURCE_SKILLS.is_dir():
        return
    if _BUNDLED_TARGET.exists():
        shutil.rmtree(_BUNDLED_TARGET)
    shutil.copytree(_SOURCE_SKILLS, _BUNDLED_TARGET)


if __name__ == "__main__":
    _mirror_bundled_skills()
    setuptools.setup()
