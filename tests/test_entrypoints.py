"""Regression coverage for CLI import paths and package manifest entries."""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from setuptools import build_meta


REPO = Path(__file__).resolve().parents[1]


def _run_module_help(module_name: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _run_source_module(args: list[str], *, cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO.parent)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def _build_wheel(tmpdir: str) -> Path:
    source_copy = Path(tmpdir) / "source"
    source_copy.mkdir()

    for name in [
        "README.md",
        "LICENSE",
        "__init__.py",
        "pipeline.py",
        "runtime_paths.py",
        "pyproject.toml",
    ]:
        shutil.copy2(REPO / name, source_copy / name)
    for name in [
        "benchmarks",
        "cli",
        "data",
        "diagnostics",
        "experiments",
        "models",
        "optimization",
        "scripts",
        "space",
    ]:
        shutil.copytree(REPO / name, source_copy / name)

    with contextlib.chdir(source_copy):
        wheel_name = build_meta.build_wheel(tmpdir)
    return Path(tmpdir) / wheel_name


def _built_wheel_members() -> set[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = _build_wheel(tmpdir)
        with zipfile.ZipFile(wheel_path) as wheel:
            return set(wheel.namelist())


def _run_from_installed_wheel(code: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        site = root / "site"
        work = root / "work"
        work.mkdir()

        wheel_path = _build_wheel(tmpdir)
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel.extractall(site)

        pkg = site / "LNPBO"
        for path in sorted(pkg.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o444)
        pkg.chmod(0o555)
        site.chmod(0o555)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(site)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=work,
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        return completed.stdout.strip()


def test_benchmark_entrypoints_support_help() -> None:
    _run_module_help("benchmarks.benchmark")
    _run_module_help("LNPBO.benchmarks.benchmark")


def test_ablation_entrypoints_support_help() -> None:
    _run_module_help("experiments.run_ablation")
    _run_module_help("LNPBO.experiments.run_ablation")


def test_setuptools_manifest_includes_new_subpackages_and_excludes_root_tests() -> None:
    members = _built_wheel_members()
    assert "LNPBO/experiments/__init__.py" in members
    assert "LNPBO/models/experimental/__init__.py" in members
    assert "LNPBO/experiments/ablations/encoding/config.json" in members
    assert "LNPBO/experiments/data_integrity/studies_with_ids.json" in members
    assert "LNPBO/setup.py" not in members
    assert not any(name.startswith("LNPBO/test_") for name in members)


def test_installed_entrypoints_default_results_to_current_working_directory() -> None:
    output = _run_from_installed_wheel(
        """
import json
from pathlib import Path
import LNPBO.benchmarks.benchmark as bench
import LNPBO.experiments.run_ablation as abl

bench.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(abl.RESULTS_BASE / "demo").mkdir(parents=True, exist_ok=True)
print(json.dumps({
    "cwd": str(Path.cwd().resolve()),
    "bench": str(bench.RESULTS_DIR),
    "ablation": str(abl.RESULTS_BASE),
}))
"""
    )
    paths = json.loads(output)
    cwd = Path(paths["cwd"])
    assert Path(paths["bench"]) == cwd / "benchmark_results" / "within_study"
    assert Path(paths["ablation"]) == cwd / "benchmark_results" / "ablations"


def test_installed_entrypoints_resolve_packaged_config_and_study_assets() -> None:
    output = _run_from_installed_wheel(
        """
import json
from pathlib import Path
import LNPBO.benchmarks.benchmark as bench
import LNPBO.experiments.run_ablation as abl
from LNPBO.runtime_paths import resolve_input_path

package_root = Path(bench.__file__).resolve().parent.parent
config_path = resolve_input_path(package_root, "experiments/ablations/encoding/config.json")
studies_with_ids = resolve_input_path(package_root, "experiments/data_integrity/studies_with_ids.json")
studies = resolve_input_path(package_root, "experiments/data_integrity/studies.json")

print(json.dumps({
    "config": str(config_path),
    "config_exists": config_path.exists(),
    "studies_with_ids": str(studies_with_ids),
    "studies_with_ids_exists": studies_with_ids.exists(),
    "studies": str(studies),
    "studies_exists": studies.exists(),
}))
"""
    )
    payload = json.loads(output)
    assert payload["config_exists"] is True
    assert payload["studies_with_ids_exists"] is True
    assert payload["studies_exists"] is True
    assert "site/LNPBO/experiments/ablations/encoding/config.json" in payload["config"]


def test_installed_runner_uses_exact_online_conformal_strategy_path() -> None:
    output = _run_from_installed_wheel(
        """
import json
from LNPBO.benchmarks.runner import STRATEGY_CONFIGS

print(json.dumps({
    "exact_type": STRATEGY_CONFIGS["discrete_xgb_online_conformal"]["type"],
    "baseline_type": STRATEGY_CONFIGS["discrete_xgb_cumulative_split_conformal_ucb_baseline"]["type"],
}))
"""
    )
    payload = json.loads(output)
    assert payload["exact_type"] == "discrete_online_conformal_exact"
    assert payload["baseline_type"] == "discrete_online_conformal_baseline"


def test_runner_cli_smoke_supports_exact_online_conformal_strategy() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = str(Path(tmpdir) / "runner_smoke")
        completed = _run_source_module(
            [
                "-m",
                "LNPBO.benchmarks.runner",
                "--strategies",
                "discrete_xgb_online_conformal",
                "--rounds",
                "1",
                "--batch-size",
                "2",
                "--n-seeds",
                "20",
                "--subset",
                "80",
                "--output",
                output_prefix,
                "--no-plot",
            ],
            cwd=Path(tmpdir),
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout


def test_fsbo_exact_module_entrypoint_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        completed = _run_source_module(
            ["-m", "LNPBO.models.experimental.fsbo"],
            cwd=Path(tmpdir),
            extra_env={"LNPBO_FSBO_SMOKE": "1"},
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
