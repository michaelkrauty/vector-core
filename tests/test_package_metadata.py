from __future__ import annotations

import tomllib
from pathlib import Path

import vector_core

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_version_matches_project_metadata() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert vector_core.__version__ == pyproject["project"]["version"]

