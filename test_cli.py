"""
Unit tests for cli.py's non-interactive helper functions.
`main()` itself isn't covered here since it drives the live LangGraph (integration-level).
"""
import os
import pytest

import cli


def test_blank_state_defaults():
    state = cli._blank_state("check")
    assert state["mode"] == "check"
    assert state["project_dir"] == "sandbox"
    assert state["branch_coverage"] == 0.0
    assert state["pending_fixes"] == []
    assert state["aborted"] is False


def test_blank_state_request_mode():
    state = cli._blank_state("request")
    assert state["mode"] == "request"


def test_load_scenario_copies_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scenario_dir = tmp_path / "scenarios" / "demo"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "app.py").write_text("x = 1")

    cli.load_scenario("demo")

    assert (tmp_path / "sandbox" / "app.py").exists()


def test_load_scenario_missing_scenario_prints_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli.load_scenario("does_not_exist")
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_load_scenario_overwrites_existing_sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scenario_dir = tmp_path / "scenarios" / "demo"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "app.py").write_text("NEW")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "old_file.py").write_text("OLD")

    cli.load_scenario("demo")

    assert (tmp_path / "sandbox" / "app.py").read_text() == "NEW"


def test_ensure_sandbox_exits_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.ensure_sandbox()


def test_ensure_sandbox_passes_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sandbox").mkdir()
    cli.ensure_sandbox()  # should not raise
