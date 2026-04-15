"""Helpers for source-tree versus installed-package runtime layouts."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def package_root_from(module_file: str | Path, *, levels_up: int) -> Path:
    """Return the package root reached by walking ``levels_up`` parents."""
    path = Path(module_file).resolve()
    for _ in range(levels_up):
        path = path.parent
    return path


def in_source_checkout(package_root: Path) -> bool:
    """Return ``True`` when running from a source checkout."""
    return (package_root / "pyproject.toml").exists()


def workspace_root(package_root: Path) -> Path:
    """Use the checkout root locally and the current working directory when installed."""
    if in_source_checkout(package_root):
        return package_root
    return Path.cwd().resolve()


def benchmark_results_root(package_root: Path) -> Path:
    """Return the benchmark-results root for the active runtime layout."""
    return workspace_root(package_root) / "benchmark_results"


def paper_root(package_root: Path) -> Path:
    """Return the paper-output root for the active runtime layout."""
    return workspace_root(package_root) / "paper"


def resolve_input_path(package_root: Path, path_str: str) -> Path:
    """Resolve a runtime input path against the active workspace or package tree."""
    raw = Path(path_str)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, package_root / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve input path {path_str!r}. Looked in: {attempted}")


def import_from_layout(active_package: str | None, *, source_name: str, installed_name: str):
    """Import a module from the source-tree or installed-package layout."""
    if (active_package or "").startswith("LNPBO."):
        return import_module(installed_name)
    return import_module(source_name)
