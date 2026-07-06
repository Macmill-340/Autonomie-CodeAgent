"""
Unit tests for agent.py's node/routing logic.

The Gemini LLM is mocked everywhere (via monkeypatching agent.llm / agent.json_llm),
since these are unit tests for the state machine, not integration tests against Gemini.
"""
import json
from types import SimpleNamespace

import pytest

import agent
import tools


class FakeLLM:
    """Stand-in for the langchain chat model. Returns whatever text you queue up."""
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        text = self._responses.pop(0) if self._responses else "{}"
        return SimpleNamespace(text=text)


def _base_state(**overrides):
    state = {
        "mode": "check", "project_dir": "sandbox", "request": "", "intent": "",
        "plan": [], "proposed_changes": {}, "heal_instructions": [], "test_output": "",
        "test_passed": False, "retry_count": 0, "needs_human": False, "human_feedback": "",
        "aborted": False, "branch_coverage": 0.0, "needs_coverage_fix": False,
        "coverage_attempts": 0, "context_doc": "", "pending_fixes": [], "current_fix_index": 0,
        "fix_retry_count": 0,
    }
    state.update(overrides)
    return state


# ---------- test_verifier_node ----------

def test_verifier_node_writes_missing_tests_and_passes_coverage(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def add(a, b): return a + b")
    monkeypatch.setattr(agent, "json_llm", FakeLLM(['{"test_app.py": "def test_add(): assert True"}']))
    monkeypatch.setattr(agent, "run_tests_coverage",
                         lambda d: {"passed": True, "output": "ok", "branch_coverage": 95.0})

    result = agent.test_verifier_node(_base_state(project_dir=str(tmp_path)))

    assert (tmp_path / "test_app.py").exists()
    assert result["needs_coverage_fix"] is False
    assert result["branch_coverage"] == 95.0


def test_verifier_node_triggers_edge_case_generation_below_threshold(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def add(a, b): return a + b")
    responses = ['{}', '{"test_app.py": "def test_edge(): assert True"}']
    monkeypatch.setattr(agent, "json_llm", FakeLLM(responses))
    monkeypatch.setattr(agent, "run_tests_coverage",
                         lambda d: {"passed": True, "output": "ok", "branch_coverage": 40.0})

    result = agent.test_verifier_node(_base_state(project_dir=str(tmp_path), coverage_attempts=0))

    assert result["needs_coverage_fix"] is True
    assert result["coverage_attempts"] == 1


def test_verifier_node_gives_up_after_max_attempts(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def add(a, b): return a + b")
    monkeypatch.setattr(agent, "json_llm", FakeLLM(['{}']))
    monkeypatch.setattr(agent, "run_tests_coverage",
                         lambda d: {"passed": True, "output": "ok", "branch_coverage": 40.0})

    result = agent.test_verifier_node(_base_state(project_dir=str(tmp_path), coverage_attempts=2))
    assert result["needs_coverage_fix"] is False
    assert result["branch_coverage"] == 40.0


# ---------- routing functions (pure, no LLM) ----------

def test_route_after_verifier_loops_on_coverage_fix():
    assert agent.route_after_verifier(_base_state(needs_coverage_fix=True)) == "test_verifier"


def test_route_after_verifier_check_mode_goes_to_test_runner():
    assert agent.route_after_verifier(_base_state(mode="check")) == "test_runner"


def test_route_after_verifier_request_mode_goes_to_analyze():
    assert agent.route_after_verifier(_base_state(mode="request")) == "analyze_codebase"


def test_route_after_test_passed_ends():
    assert agent.route_after_test(_base_state(test_passed=True)) == agent.END


def test_route_after_test_heals_pending_fix_under_retry_limit():
    state = _base_state(test_passed=False, pending_fixes=[{"file": "a.py", "fix": "x"}],
                         current_fix_index=0, fix_retry_count=1)
    assert agent.route_after_test(state) == "self_healer"


def test_route_after_test_moves_to_next_fix_after_exhausting_retries():
    state = _base_state(test_passed=False,
                         pending_fixes=[{"file": "a.py", "fix": "x"}, {"file": "b.py", "fix": "y"}],
                         current_fix_index=0, fix_retry_count=3)
    assert agent.route_after_test(state) == "self_healer"


def test_route_after_test_escalates_after_last_fix_exhausted():
    state = _base_state(test_passed=False, pending_fixes=[{"file": "a.py", "fix": "x"}],
                         current_fix_index=0, fix_retry_count=3)
    assert agent.route_after_test(state) == "escalator"


def test_route_after_test_fallback_no_pending_fixes():
    assert agent.route_after_test(_base_state(test_passed=False, retry_count=3)) == "escalator"
    assert agent.route_after_test(_base_state(test_passed=False, retry_count=1)) == "self_healer"


def test_route_after_hitl_aborted_ends():
    assert agent.route_after_hitl(_base_state(aborted=True)) == agent.END


def test_route_after_hitl_not_aborted_goes_to_code_writer():
    assert agent.route_after_hitl(_base_state(aborted=False)) == "code_writer"


def test_route_after_escalator_aborted_ends():
    assert agent.route_after_escalator(_base_state(aborted=True)) == agent.END


def test_route_after_escalator_not_aborted_goes_to_code_writer():
    assert agent.route_after_escalator(_base_state(aborted=False)) == "code_writer"


# ---------- test_runner_node checkpoints on pass ----------

def test_test_runner_node_snapshots_on_pass(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "app.py").write_text("ok")
    backup = tmp_path / "backup"

    monkeypatch.setattr(agent, "run_tests", lambda d: {"passed": True, "output": "3 passed"})
    monkeypatch.setattr(tools, "LAST_GOOD_DIR", str(backup))
    monkeypatch.setattr(agent, "snapshot_sandbox",
                         lambda d: tools.snapshot_sandbox(d, str(backup)))

    result = agent.test_runner_node(_base_state(project_dir=str(sandbox)))

    assert result["test_passed"] is True
    assert (backup / "app.py").exists()


def test_test_runner_node_no_snapshot_on_failure(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    backup = tmp_path / "backup"

    monkeypatch.setattr(agent, "run_tests", lambda d: {"passed": False, "output": "1 failed"})
    called = {"snapshot": False}
    monkeypatch.setattr(agent, "snapshot_sandbox", lambda d: called.__setitem__("snapshot", True))

    result = agent.test_runner_node(_base_state(project_dir=str(sandbox)))

    assert result["test_passed"] is False
    assert called["snapshot"] is False


# ---------- escalator_node: failure log + abort/rollback ----------

def test_escalator_node_writes_log_and_rolls_back_on_abort(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    backup = tmp_path / "backup"
    sandbox.mkdir()
    backup.mkdir()
    (backup / "app.py").write_text("GOOD VERSION")

    monkeypatch.setattr("builtins.input", lambda prompt="": "abort")
    monkeypatch.setattr(agent, "write_failure_log", lambda d, fixes, out: str(tmp_path / "failure_log.json"))
    monkeypatch.setattr(agent, "restore_last_good", lambda d: tools.restore_last_good(d, str(backup)))

    result = agent.escalator_node(_base_state(project_dir=str(sandbox),
                                               pending_fixes=[{"file": "a.py", "fix": "x"}],
                                               test_output="boom"))

    assert result["aborted"] is True
    assert (sandbox / "app.py").read_text() == "GOOD VERSION"


def test_escalator_node_hint_continues_without_abort(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "try using math.isclose")
    monkeypatch.setattr(agent, "write_failure_log", lambda d, fixes, out: "irrelevant")

    result = agent.escalator_node(_base_state(pending_fixes=[{"file": "a.py", "fix": "x"}]))

    assert "aborted" not in result
    assert result["human_feedback"] == "try using math.isclose"
