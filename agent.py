import json
import os
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tools import (read_file, write_file, list_project_files, run_tests, run_tests_coverage, COVERAGE_THRESHOLD,
                    snapshot_sandbox, restore_last_good, write_failure_log)

load_dotenv()

#state
class AgentState(TypedDict):
    mode: str
    project_dir: str
    request: str
    intent: str
    plan: List[str]
    proposed_changes: dict
    heal_instructions: List[str]
    test_output: str
    test_passed: bool
    retry_count: int
    needs_human: bool
    human_feedback: str
    aborted: bool
    # --- New coverage memory ---
    branch_coverage: float  # Tracks the last measured coverage %
    needs_coverage_fix: bool  # True = tells the graph to loop back to write more tests
    coverage_attempts: int  # Tracks how many times we've tried (to prevent infinite loops)
    context_doc: str
    # --- Multi-fix healing state ---
    pending_fixes: List[dict]
    current_fix_index: int
    fix_retry_count: int

#model
# NOTE (stability fix): the LLM client used to be built at import time, which meant
# `import agent` crashed with a pydantic ValidationError any time GEMINI_API_KEY
# wasn't set (e.g. running tests, or importing agent.py from a script that doesn't
# need the LLM yet). It's now created lazily on first real use.
_llm_instance = None

def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3,
            max_retries=8,
        )
    return _llm_instance


class _LazyLLM:
    """Thin proxy so existing `llm.invoke(...)` / `json_llm.invoke(...)` call sites don't change."""
    def __init__(self, json_mode: bool = False):
        self._json_mode = json_mode

    def invoke(self, *args, **kwargs):
        base = _get_llm()
        if self._json_mode:
            base = base.bind(response_mime_type="application/json")
        return base.invoke(*args, **kwargs)


llm = _LazyLLM(json_mode=False)
json_llm = _LazyLLM(json_mode=True)

def _get_codebase_context(dir: str) -> str:
    files = list_project_files(dir)
    return "\n".join([f"\n--- {f} ---\n{read_file(f)}" for f in files])

#nodes
def router_node(state: AgentState) -> dict:
    #passes execution to test verifier
    return {}

def test_verifier_node(state: AgentState) -> dict:
    """The Gatekeeper: Ensures tests exist before any other action is taken."""
    current_code = _get_codebase_context(state["project_dir"])
    print("\n🔎 Verifying tests and branch coverage...")

    # --- Part 1: Ensure test files exist (Your original logic) ---
    messages = [
        ("system",
         'You are a TDD enforcer. Review the codebase. If functions lack tests or no '
         'test files exist, output a JSON mapping of the new test files to create. '
         'Example (Python): {"test_app.py": "import pytest\\n..."}. '
         'Example (Node/JS): {"app.test.js": "const app = require(...) \\n test(...)"}. '
         'If tests are fully adequate, return exactly {}.'),
        ("user", f"Codebase:\n{current_code}\n\nReturn JSON:")
    ]
    response = json_llm.invoke(messages)
    try:
        proposed = json.loads(response.text)
        for rel_path, content in proposed.items():
            if content:
                print(f"   ↳ Writing missing test file: {rel_path}")
                rel_path = rel_path.removeprefix("sandbox/").removeprefix("/")
                write_file(os.path.join(state["project_dir"], rel_path), content)
    except Exception:
        pass

    # --- Part 2: The New Coverage Gate ---
    attempts = state.get("coverage_attempts", 0)
    result = run_tests_coverage(state["project_dir"])
    branch_cov = result.get("branch_coverage", 0.0)
    tests_passed = result.get("passed", False)

    print(f"   ↳ Branch coverage: {branch_cov}%  (threshold: {COVERAGE_THRESHOLD}%)")

    # If coverage is low, tests are passing, and we haven't hit our retry limit:
    if branch_cov < COVERAGE_THRESHOLD and tests_passed and attempts < 2:
        print(f"   ↳ Coverage too low. Writing edge-case tests (attempt {attempts + 1}/2)...")

        edge_messages = [
            ("system",
             'You are a TDD enforcer. Branch coverage is below the required threshold. '
             'Write ADDITIONAL test cases specifically targeting uncovered branches: '
             'None/null inputs, zero, negative numbers, empty strings, '
             'boundary values, and expected exception paths. '
             'Use Pytest for Python, Jest for JS/TS. '
             'Return ONLY a JSON mapping of test file paths to their COMPLETE updated content. '
             'You MUST merge with existing tests — do NOT remove any existing test cases.'),
            ("user",
             f"Codebase:\n{current_code}\n\n"
             f"Current branch coverage: {branch_cov}%\n"
             f"Return JSON:")
        ]
        try:
            edge_proposed = json.loads(json_llm.invoke(edge_messages).text)
            for rel_path, content in edge_proposed.items():
                if content:
                    rel_path = rel_path.removeprefix("sandbox/").removeprefix("/")
                    write_file(os.path.join(state["project_dir"], rel_path), content)
                    print(f"   ↳ Updated edge-case tests: {rel_path}")
        except Exception as e:
            print(f"   ↳ Edge-case test generation failed: {e}")

        # Tell LangGraph we need to loop back and check again
        return {
            "needs_coverage_fix": True,
            "coverage_attempts": attempts + 1,
            "branch_coverage": branch_cov
        }

    # Give up gracefully after 2 retries — log a warning but don't block forever
    if branch_cov < COVERAGE_THRESHOLD:
        print(f"   ↳ ⚠️ Coverage still at {branch_cov}% after {attempts} attempt(s). Proceeding anyway.")

    return {
        "needs_coverage_fix": False,
        "branch_coverage": branch_cov,
        "coverage_attempts": attempts
    }


def analyze_codebase_node(state: AgentState) -> dict:
    current_code = _get_codebase_context(state["project_dir"])

    # Inject the context document if it exists!
    extra_context = ""
    if state.get("context_doc"):
        extra_context = f"\n\n--- SUPPLIED CONTEXT/SPEC ---\n{state['context_doc']}\n-----------------------------"

    messages = [
        ("system",
         "Summarize relevant codebase context for a feature request. If a spec/context is provided, incorporate its rules into your summary."),
        ("user", f"Codebase:\n{current_code}{extra_context}\n\nRequest: {state['request']}\nSummary:")
    ]
    return {"intent": llm.invoke(messages).text}


def planner_node(state: AgentState) -> dict:
    messages = [
        ("system",
         "Return ONLY a numbered list of implementation steps. You MUST include updating/writing tests for the new feature as a step."),
        ("user", f"Feature: {state['request']}\nContext: {state['intent']}\nPlan:")
    ]
    plan = [l.strip() for l in llm.invoke(messages).text.splitlines() if l.strip()]
    return {"plan": plan}


def hitl_check_node(state: AgentState) -> dict:
    plan_text = " ".join(state.get("plan", [])).lower()
    major_signals = ["new file", "create file", "add dependency", "database"]

    # Pause if it's a feature request OR if it contains major signals
    if state["mode"] != "request" and not any(s in plan_text for s in major_signals):
        return {"needs_human": False}

    print("\n📋 Implementation Plan:")
    for i, step in enumerate(state.get("plan", []), 1):
        print(f"  {i}. {step}")

    answer = input("\n⚠️ Major change detected. Approve? [y / n / type feedback]: ").strip()
    if answer.lower() == "n":
        print("Aborted by user.")
        return {"needs_human": True, "aborted": True}

    return {"needs_human": True, "human_feedback": "" if answer.lower() == "y" else answer}


def code_writer_node(state: AgentState) -> dict:
    current_code = _get_codebase_context(state["project_dir"])

    context = f"Intent: {state.get('intent', '')}\n"
    if state.get("plan"): context += "Plan:\n" + "\n".join(state["plan"]) + "\n"
    if state.get("heal_instructions"):
        fixes = state["heal_instructions"]
        if isinstance(fixes, list):
            context += "Fix these errors:\n" + "\n".join(str(f) for f in fixes) + "\n"
        else:
            context += f"Fix this error:\n{fixes}\n"
    if state.get("human_feedback"): context += f"Human feedback: {state['human_feedback']}\n"

    messages = [
        ("system",
         'Return ONLY a valid JSON object mapping file paths to new content. Example: {"app.py": "code here"}'),
        ("user", f"{context}\n\nCurrent code:\n{current_code}\n\nUpdate files as JSON.")
    ]

    try:
        proposed = json.loads(json_llm.invoke(messages).text)
        for rel_path, content in proposed.items():
            rel_path = rel_path.removeprefix("sandbox/").removeprefix("/")
            write_file(os.path.join(state["project_dir"], rel_path), content)
    except Exception as e:
        print(f"JSON parse failed: {e}")
        proposed = {}

    # After applying fixes, advance to next pending fix
    pending = state.get("pending_fixes", [])
    idx = state.get("current_fix_index", 0)
    if pending and idx < len(pending):
        return {
            "proposed_changes": proposed,
            "heal_instructions": [],
            "human_feedback": "",
            "current_fix_index": idx + 1,
            "fix_retry_count": 0
        }

    return {"proposed_changes": proposed, "heal_instructions": [], "human_feedback": ""}


def test_runner_node(state: AgentState) -> dict:
    result = run_tests(state["project_dir"])
    if result["passed"]:
        # Checkpoint: only overwrite the "last known good" snapshot once tests are green.
        snapshot_sandbox(state["project_dir"])
    return {"test_output": result["output"], "test_passed": result["passed"]}


def self_healer_node(state: AgentState) -> dict:
    pending = state.get("pending_fixes", [])
    idx = state.get("current_fix_index", 0)
    fix_retries = state.get("fix_retry_count", 0)

    # First failure → get the list of fixes
    if not pending:
        current_code = _get_codebase_context(state["project_dir"])
        messages = [
            ("system",
             "You are a debugger. Return a JSON array of independent fixes. "
             "Each object must have: {\"file\": \"path\", \"fix\": \"precise change description\"}. "
             "Only include distinct locations that need fixing. Be exhaustive."),
            ("user", f"Test output:\n{state['test_output']}\n\nCurrent code:\n{current_code}\n\nReturn JSON array:")
        ]
        try:
            fixes = json.loads(llm.invoke(messages).text)
            if not isinstance(fixes, list):
                fixes = [fixes]
        except Exception:
            fixes = []
        return {
            "pending_fixes": fixes,
            "current_fix_index": 0,
            "fix_retry_count": 0,
            "heal_instructions": fixes
        }

    # Subsequent retries on same fix
    print(f"\n[Agent] Healing fix {idx+1}/{len(pending)} (Attempt {fix_retries+1}/3)...")
    return {"fix_retry_count": fix_retries + 1}


def escalator_node(state: AgentState) -> dict:
    pending = state.get("pending_fixes", [])
    idx = state.get("current_fix_index", 0)
    print(f"\n❌ Could not fix after exhausting 3 attempts on fix {idx+1}/{len(pending)}.")
    print(f"\nLast test output:\n{state['test_output']}")

    log_path = write_failure_log(state["project_dir"], pending, state.get("test_output", ""))
    print(f"📝 Wrote failure log: {log_path}")

    hint = input("\n💡 Type 'abort' to roll back to the last checkpoint, or provide a hint for the agent: ").strip()

    if hint.lower() == "abort":
        restored = restore_last_good(state["project_dir"])
        if restored:
            print("⏪ Rolled back sandbox to last known-good checkpoint.")
        else:
            print("⚠️ No checkpoint found yet — nothing to roll back to.")
        return {"aborted": True, "pending_fixes": [], "current_fix_index": 0, "fix_retry_count": 0}

    return {"human_feedback": hint, "retry_count": 0, "pending_fixes": [], "current_fix_index": 0, "fix_retry_count": 0}


#routing
def route_after_verifier(state: AgentState) -> str:
    if state.get("needs_coverage_fix"):
        return "test_verifier"   # Loop back to itself!
    return "test_runner" if state["mode"] == "check" else "analyze_codebase"

def route_after_test(state: AgentState) -> str:
    if state.get("test_passed"):
        print("\n✅ All tests passed!")
        return END

    pending = state.get("pending_fixes", [])
    idx = state.get("current_fix_index", 0)
    fix_retries = state.get("fix_retry_count", 0)

    # If we have pending fixes, keep healing current one until 3 attempts exhausted
    if pending and idx < len(pending):
        if fix_retries < 3:
            return "self_healer"
        # Exhausted 3 tries on this fix → move to next fix
        return "self_healer" if idx + 1 < len(pending) else "escalator"

    # Fallback to old behavior if no pending_fixes
    return "escalator" if state.get("retry_count", 0) >= 3 else "self_healer"


def route_after_hitl(state: AgentState) -> str:
    return END if state.get("aborted") else "code_writer"

def route_after_escalator(state: AgentState) -> str:
    return END if state.get("aborted") else "code_writer"


#graph
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("test_verifier", test_verifier_node)
    g.add_node("analyze_codebase", analyze_codebase_node)
    g.add_node("planner", planner_node)
    g.add_node("hitl_check", hitl_check_node)
    g.add_node("code_writer", code_writer_node)
    g.add_node("test_runner", test_runner_node)
    g.add_node("self_healer", self_healer_node)
    g.add_node("escalator", escalator_node)

    g.set_entry_point("router")

    g.add_edge("router", "test_verifier")
    g.add_conditional_edges("test_verifier", route_after_verifier, {
        "test_verifier": "test_verifier",
        "test_runner": "test_runner",
        "analyze_codebase": "analyze_codebase"
    })

    g.add_edge("analyze_codebase", "planner")
    g.add_edge("planner", "hitl_check")
    g.add_conditional_edges("hitl_check", route_after_hitl, {"code_writer": "code_writer", END: END})
    g.add_edge("code_writer", "test_runner")

    g.add_conditional_edges("test_runner", route_after_test, {
        "self_healer": "self_healer",
        "escalator": "escalator",
        END: END
    })

    g.add_edge("self_healer", "code_writer")
    g.add_conditional_edges("escalator", route_after_escalator, {"code_writer": "code_writer", END: END})

    return g.compile(checkpointer=MemorySaver())


graph = build_graph()