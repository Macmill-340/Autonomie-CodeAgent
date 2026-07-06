"""
Unit tests for tools.py

Covers: file I/O helpers, ecosystem detection, Windows cmd resolution,
pytest/jest test running (with subprocess mocked), branch-coverage parsing,
and the checkpoint/rollback functions added for stabilization.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import tools


# ---------- file helpers ----------

def test_write_then_read_file(tmp_path):
    target = tmp_path / "nested" / "file.txt"
    tools.write_file(str(target), "hello world")
    assert tools.read_file(str(target)) == "hello world"


def test_write_file_creates_parent_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c.py"
    tools.write_file(str(target), "print(1)")
    assert target.exists()


def test_list_project_files_filters_extensions_and_ignored_dirs(tmp_path):
    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / "readme.md").write_text("# hi")  # not in extensions set
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    (ignored / "lib.js").write_text("ignored")

    files = tools.list_project_files(str(tmp_path))
    names = {Path(f).name for f in files}
    assert "app.py" in names
    assert "lib.js" not in names
    assert "readme.md" not in names


def test_list_project_files_empty_dir(tmp_path):
    assert tools.list_project_files(str(tmp_path)) == []


# ---------- ecosystem detection ----------

def test_detect_ecosystem_node(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    assert tools._detect_ecosystem(str(tmp_path)) == "node"


def test_detect_ecosystem_python_default(tmp_path):
    assert tools._detect_ecosystem(str(tmp_path)) == "python"


# ---------- cmd resolution ----------

def test_resolve_cmd_absolute_path_passthrough(tmp_path):
    fake_exe = tmp_path / "pytest"
    fake_exe.write_text("")
    resolved = tools._resolve_cmd([str(fake_exe), "-q"], str(tmp_path))
    assert resolved == [str(fake_exe), "-q"]


def test_resolve_cmd_uses_which(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pytest" if name == "pytest" else None)
    resolved = tools._resolve_cmd(["pytest", "-q"], "somedir")
    assert resolved == ["/usr/bin/pytest", "-q"]


def test_resolve_cmd_falls_back_to_cmd_shim_on_windows(monkeypatch):
    def fake_which(name):
        return "/usr/bin/npm.cmd" if name == "npm.cmd" else None
    monkeypatch.setattr(shutil, "which", fake_which)
    resolved = tools._resolve_cmd(["npm", "install"], "somedir")
    assert resolved == ["/usr/bin/npm.cmd", "install"]


def test_resolve_cmd_unresolvable_returns_original(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    resolved = tools._resolve_cmd(["ghost-binary"], "somedir")
    assert resolved == ["ghost-binary"]


# ---------- run_tests (subprocess mocked) ----------

class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_tests_python_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "1 passed", ""))
    result = tools.run_tests(str(tmp_path))
    assert result["passed"] is True
    assert "1 passed" in result["output"]


def test_run_tests_python_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(1, "1 failed", "AssertionError"))
    result = tools.run_tests(str(tmp_path))
    assert result["passed"] is False


def test_run_tests_detects_no_tests_found(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "collected 0 items", ""))
    result = tools.run_tests(str(tmp_path))
    assert result["passed"] is False
    assert "NO TESTS FOUND" in result["output"]


def test_run_tests_auto_installs_missing_module(tmp_path, monkeypatch):
    calls = {"test_runs": 0, "install_called": False}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "pip":
            calls["install_called"] = True
            return _FakeResult(0, "", "")
        calls["test_runs"] += 1
        if calls["test_runs"] == 1:
            return _FakeResult(1, "", "ModuleNotFoundError: No module named 'requests'")
        return _FakeResult(0, "2 passed", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(tools, "_resolve_cmd", lambda cmd, cwd: cmd)

    result = tools.run_tests(str(tmp_path))
    assert result["passed"] is True
    assert calls["install_called"] is True
    assert calls["test_runs"] == 2  # retried once after installing


def test_run_tests_node_ecosystem(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text("{}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "Tests: 3 passed", ""))
    result = tools.run_tests(str(tmp_path))
    assert result["passed"] is True


# ---------- run_tests_coverage ----------

def test_run_tests_coverage_python_parses_json(tmp_path, monkeypatch):
    cov_file = tmp_path / "coverage.json"
    cov_file.write_text(json.dumps({"totals": {"num_branches": 10, "covered_branches": 9}}))

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "5 passed", ""))
    result = tools.run_tests_coverage(str(tmp_path))

    assert result["passed"] is True
    assert result["branch_coverage"] == 90.0


def test_run_tests_coverage_no_tests_found(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(1, "collected 0 items", ""))
    result = tools.run_tests_coverage(str(tmp_path))
    assert result["passed"] is False
    assert result["branch_coverage"] == 0.0


def test_run_tests_coverage_fails_open_when_no_coverage_file(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "5 passed", ""))
    result = tools.run_tests_coverage(str(tmp_path))
    # No coverage.json written -> fail-open default of 100.0
    assert result["branch_coverage"] == 100.0


def test_run_tests_coverage_node(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text("{}")
    cov_dir = tmp_path / "coverage"
    cov_dir.mkdir()
    (cov_dir / "coverage-summary.json").write_text(
        json.dumps({"total": {"branches": {"total": 4, "covered": 2}}})
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult(0, "PASS", ""))
    result = tools.run_tests_coverage(str(tmp_path))
    assert result["branch_coverage"] == 50.0


# ---------- checkpoint / rollback (Step 5) ----------

def test_snapshot_sandbox_missing_source_returns_false(tmp_path):
    assert tools.snapshot_sandbox(str(tmp_path / "nope"), str(tmp_path / "backup")) is False


def test_snapshot_and_restore_round_trip(tmp_path):
    sandbox = tmp_path / "sandbox"
    backup = tmp_path / "backup"
    sandbox.mkdir()
    (sandbox / "app.py").write_text("VERSION = 1")

    assert tools.snapshot_sandbox(str(sandbox), str(backup)) is True
    assert (backup / "app.py").read_text() == "VERSION = 1"

    # Simulate the agent breaking the sandbox
    (sandbox / "app.py").write_text("VERSION = 2 # broken")

    assert tools.restore_last_good(str(sandbox), str(backup)) is True
    assert (sandbox / "app.py").read_text() == "VERSION = 1"


def test_restore_last_good_no_checkpoint_returns_false(tmp_path):
    assert tools.restore_last_good(str(tmp_path / "sandbox"), str(tmp_path / "backup")) is False


def test_snapshot_overwrites_previous_checkpoint(tmp_path):
    sandbox = tmp_path / "sandbox"
    backup = tmp_path / "backup"
    sandbox.mkdir()
    (sandbox / "app.py").write_text("v1")
    tools.snapshot_sandbox(str(sandbox), str(backup))

    (sandbox / "app.py").write_text("v2")
    tools.snapshot_sandbox(str(sandbox), str(backup))

    assert (backup / "app.py").read_text() == "v2"


def test_write_failure_log(tmp_path):
    fixes = [{"file": "app.py", "fix": "handle zero division"}]
    log_path = tools.write_failure_log(str(tmp_path), fixes, "AssertionError: boom")

    data = json.loads(Path(log_path).read_text())
    assert data["attempted_fixes"] == fixes
    assert "boom" in data["last_test_output"]
