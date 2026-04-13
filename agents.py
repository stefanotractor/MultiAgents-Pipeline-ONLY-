"""
Multi-Agent System.

3 agents: supervisor (routing logic), code_executor (direct LLM → REPL),
validator (file check — no LLM needed).

Key design: NO tool calling. The LLM generates pure Python code,
and the code_executor runs it directly in a REPL.
Includes rate limiting for free-tier APIs (Gemini: 10 RPM).
Includes task caching: if a task's output file already exists, skip it.
"""

import os
import time
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.types import Command
from openai import OpenAI

from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_MODEL,
    OUTPUT_DIR,
)
from tools import PythonREPL
from prompts import TASKS

# ── Ensure output dir exists ─────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Direct LLM client (no LangChain, no tool calling) ───────────────────────
_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
_repl = PythonREPL()

# ── Rate limiting ────────────────────────────────────────────────────────────
_SECONDS_BETWEEN_CALLS = 8  # 10 RPM = 1 every 6s, we use 8s for safety
_last_call_time = 0.0

# ── Verbose logging ──────────────────────────────────────────────────────────
VERBOSE = True  # Set to False once pipeline is stable

# ── Cache control (set to False via --no-cache in main.py) ───────────────────
USE_CACHE = True


# ── State ────────────────────────────────────────────────────────────────────
class AgentState(MessagesState):
    next: str
    current_task_index: int
    task_status: str
    retry_count: int


MAX_RETRIES = 2

# Maps each task name to the file it MUST produce.
EXPECTED_FILES = {
    "data_loading_allarmi":   os.path.join(OUTPUT_DIR, "allarmi_clean.csv"),
    "data_loading_tipologia": os.path.join(OUTPUT_DIR, "tipologia_clean.csv"),
    "merge":                  os.path.join(OUTPUT_DIR, "merged_data.csv"),
    "baseline_grouping":      os.path.join(OUTPUT_DIR, "routes_summary.csv"),
    "baseline_stats":         os.path.join(OUTPUT_DIR, "baseline_data.csv"),
    "outlier_detection":      os.path.join(OUTPUT_DIR, "outlier_results.csv"),
    "risk_profiling":         os.path.join(OUTPUT_DIR, "risk_profiled.csv"),
    "report_generation":      os.path.join(OUTPUT_DIR, "anomaly_report.txt"),
}


# ── Helper: check if task output already exists (cache) ──────────────────────
def _task_is_cached(task_name: str) -> bool:
    """Return True if this task's output file already exists and is non-empty."""
    if not USE_CACHE:
        return False
    filepath = EXPECTED_FILES.get(task_name, "")
    if not filepath:
        return False
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        size = os.path.getsize(filepath)
        print(f"  [cache] '{task_name}' → {os.path.basename(filepath)} already exists ({size:,} bytes), skipping.")
        return True
    return False


# ── Helper: rate-limited LLM call ───────────────────────────────────────────
def _call_llm(messages: list[dict], max_tokens: int = 1024) -> str:
    """Call LLM with rate limiting and 429 backoff."""
    global _last_call_time

    # Rate limit: wait if needed
    elapsed = time.time() - _last_call_time
    if elapsed < _SECONDS_BETWEEN_CALLS:
        wait = _SECONDS_BETWEEN_CALLS - elapsed
        print(f"  [rate_limit] Waiting {wait:.0f}s...")
        time.sleep(wait)

    # Try with exponential backoff on 429
    for attempt in range(3):
        try:
            _last_call_time = time.time()
            response = _client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                backoff = 15 * (2 ** attempt)  # 15s, 30s, 60s
                print(f"  [rate_limit] 429 received, waiting {backoff}s (attempt {attempt+1}/3)...")
                time.sleep(backoff)
            else:
                return f"LLM_ERROR: {e}"

    return "LLM_ERROR: 429 — rate limit exceeded after 3 retries"


# ── Helper: clean code from LLM response ─────────────────────────────────────
def _clean_code(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    code = raw.strip()
    # Handle ```python ... ``` or ``` ... ```
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    code = code.strip()
    return code


# ── Helper: ask LLM for code and run it ──────────────────────────────────────
def _ask_and_run(task_prompt: str, retry_context: str = "") -> str:
    """
    Send task to LLM, get Python code back, execute in REPL.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python data analyst. "
                "Reply with ONLY Python code, nothing else. "
                "No markdown, no backticks, no explanations. Just code."
            ),
        },
        {"role": "user", "content": task_prompt},
    ]

    if retry_context:
        messages.append({
            "role": "user",
            "content": f"Error: {retry_context}\nFix the code. Reply with ONLY the corrected Python code.",
        })

    raw = _call_llm(messages)

    if raw.startswith("LLM_ERROR"):
        return raw

    code = _clean_code(raw)

    if not code:
        return "LLM_ERROR: empty code response"

    if VERBOSE:
        print(f"\n  {'─'*40}")
        print(f"  GENERATED CODE:")
        print(f"  {'─'*40}")
        for i, line in enumerate(code.split("\n"), 1):
            print(f"  {i:3d} | {line}")
        print(f"  {'─'*40}\n")

    # Execute
    result = _repl.run(code)
    return result


# ── Supervisor ───────────────────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> Command[Literal["code_executor", "validator", "__end__"]]:
    idx = state.get("current_task_index", 0)
    status = state.get("task_status", "pending")
    retries = state.get("retry_count", 0)

    if idx >= len(TASKS):
        return Command(goto="__end__", update={"next": "__end__"})

    task_name, task_prompt = TASKS[idx]

    # ── CACHE CHECK: skip if output already exists ──
    if status == "pending" and _task_is_cached(task_name):
        next_idx = idx + 1
        if next_idx >= len(TASKS):
            return Command(goto="__end__", update={
                "current_task_index": next_idx, "task_status": "done",
            })
        return Command(
            goto="supervisor",
            update={
                "messages": [HumanMessage(content=f"Cached: {task_name}", name="supervisor")],
                "current_task_index": next_idx,
                "task_status": "pending",
                "retry_count": 0,
            },
        )

    # ── pending / failed → send to code_executor
    if status in ("pending", "failed"):
        if status == "failed" and retries >= MAX_RETRIES:
            print(f"\n  SKIP '{task_name}' after {retries} retries\n")
            next_idx = idx + 1
            return Command(
                goto="__end__" if next_idx >= len(TASKS) else "supervisor",
                update={
                    "messages": [HumanMessage(content=f"Skipped {task_name}.", name="supervisor")],
                    "current_task_index": next_idx,
                    "task_status": "pending",
                    "retry_count": 0,
                },
            )

        return Command(
            goto="code_executor",
            update={
                "messages": [HumanMessage(content=task_prompt, name="supervisor")],
                "task_status": "executing",
            },
        )

    # ── executing → validate
    if status == "executing":
        expected = EXPECTED_FILES.get(task_name, "")
        return Command(
            goto="validator",
            update={
                "messages": [HumanMessage(content=expected, name="supervisor")],
                "task_status": "validating",
            },
        )

    # ── validating → check result
    if status == "validating":
        last = state["messages"][-1].content if state["messages"] else ""
        if last.startswith("APPROVED"):
            next_idx = idx + 1
            print(f"  ✓ Task '{task_name}' OK")
            if next_idx >= len(TASKS):
                return Command(goto="__end__", update={
                    "current_task_index": next_idx, "task_status": "done",
                })
            return Command(
                goto="supervisor",
                update={
                    "current_task_index": next_idx,
                    "task_status": "pending",
                    "retry_count": 0,
                },
            )
        else:
            return Command(
                goto="supervisor",
                update={"task_status": "failed", "retry_count": retries + 1},
            )

    return Command(goto="__end__", update={})


# ── Code Executor (direct LLM call, no tool calling) ────────────────────────
def code_executor_node(state: AgentState) -> Command[Literal["supervisor"]]:
    idx = state.get("current_task_index", 0)
    retries = state.get("retry_count", 0)

    if idx >= len(TASKS):
        return Command(
            update={"messages": [HumanMessage(content="All tasks done.", name="code_executor")]},
            goto="supervisor",
        )

    task_name, task_prompt = TASKS[idx]

    # On retry, pass the previous error to the LLM
    retry_context = ""
    if retries > 0:
        last_msg = state["messages"][-1].content if state["messages"] else ""
        if "Error" in last_msg or "REJECTED" in last_msg:
            retry_context = last_msg

    print(f"  [code_executor] Running '{task_name}' (attempt {retries + 1})...")
    result = _ask_and_run(task_prompt, retry_context)

    # Truncate for state
    if len(result) > 2000:
        result = result[:2000] + "\n... [truncated]"

    print(f"  [code_executor] Output: {result[:300]}")

    return Command(
        update={"messages": [HumanMessage(content=result, name="code_executor")]},
        goto="supervisor",
    )


# ── Validator (pure Python, no LLM needed) ───────────────────────────────────
def validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    filepath = state["messages"][-1].content if state["messages"] else ""
    filepath = filepath.strip()

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        size = os.path.getsize(filepath)
        msg = f"APPROVED: {filepath} ({size:,} bytes)"
        print(f"  [validator] {msg}")
    else:
        # Debug: list what IS in the output dir
        if VERBOSE and os.path.isdir(OUTPUT_DIR):
            files = os.listdir(OUTPUT_DIR)
            print(f"  [validator] Files in output dir: {files}")
        msg = f"REJECTED: {filepath} not found or empty"
        print(f"  [validator] {msg}")

    return Command(
        update={"messages": [HumanMessage(content=msg, name="validator")]},
        goto="supervisor",
    )


# ── Graph ────────────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("code_executor", code_executor_node)
    g.add_node("validator", validator_node)
    g.add_edge(START, "supervisor")
    return g.compile()
