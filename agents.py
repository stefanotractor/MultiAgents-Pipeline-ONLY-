"""
Multi-Agent System — v2 Light (optimized for small local LLMs).

3 agents: supervisor (routing logic), code_executor (REPL), validator (file check).
The key difference from v1: prompts include the exact code to run,
so the LLM just needs to call python_repl with it — not write code from scratch.
"""

from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL
from tools import python_repl, check_file
from prompts import TASKS


# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=LM_STUDIO_MODEL,
    temperature=0.0,
    max_tokens=2048,
)


# ── State ────────────────────────────────────────────────────────────────────
class AgentState(MessagesState):
    next: str
    current_task_index: int
    task_status: str
    retry_count: int


MAX_RETRIES = 2
EXPECTED_FILES = {
    "data_loading": "merged_data.csv",
    "baseline_building": "baseline_data.csv",
    "outlier_detection": "outlier_results.csv",
    "risk_profiling": "risk_profiled.csv",
    "report_generation": "anomaly_report.txt",
}


# ── Supervisor ───────────────────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> Command[Literal["code_executor", "validator", "__end__"]]:
    idx = state.get("current_task_index", 0)
    status = state.get("task_status", "pending")
    retries = state.get("retry_count", 0)

    if idx >= len(TASKS):
        return Command(goto="__end__", update={"next": "__end__"})

    task_name, task_prompt = TASKS[idx]

    # ── pending / failed → send to code_executor
    if status in ("pending", "failed"):
        if status == "failed" and retries >= MAX_RETRIES:
            print(f"\n⚠️  SKIP '{task_name}' after {retries} retries\n")
            next_idx = idx + 1
            return Command(
                goto="__end__" if next_idx >= len(TASKS) else "code_executor",
                update={
                    "messages": [HumanMessage(content=f"Skipped {task_name}.", name="supervisor")],
                    "current_task_index": next_idx,
                    "task_status": "pending",
                    "retry_count": 0,
                },
            )

        msg = task_prompt
        if status == "failed":
            msg = (
                f"RETRY (attempt {retries+1}): The file was not created. "
                f"You MUST call the python_repl tool. Here is the task:\n\n{task_prompt}"
            )

        return Command(
            goto="code_executor",
            update={
                "messages": [HumanMessage(content=msg, name="supervisor")],
                "task_status": "executing",
            },
        )

    # ── executing → validate
    if status == "executing":
        f = EXPECTED_FILES.get(task_name, "")
        return Command(
            goto="validator",
            update={
                "messages": [HumanMessage(
                    content=f"Check if file '{f}' exists using check_file tool. "
                            f"Reply APPROVED or REJECTED.",
                    name="supervisor",
                )],
                "task_status": "validating",
            },
        )

    # ── validating → check result
    if status == "validating":
        last = state["messages"][-1].content if state["messages"] else ""
        if "APPROVED" in last.upper() or "OK:" in last.upper():
            next_idx = idx + 1
            print(f"  ✓ Task '{task_name}' OK")
            if next_idx >= len(TASKS):
                return Command(goto="__end__", update={
                    "current_task_index": next_idx, "task_status": "done",
                })
            return Command(
                goto="code_executor",
                update={
                    "messages": [HumanMessage(content=f"Task {task_name} done. Next.", name="supervisor")],
                    "current_task_index": next_idx, "task_status": "pending", "retry_count": 0,
                },
            )
        else:
            return Command(
                goto="code_executor",
                update={"task_status": "failed", "retry_count": retries + 1},
            )

    return Command(goto="__end__", update={})


# ── Code Executor ────────────────────────────────────────────────────────────
_code_agent = create_react_agent(
    model=llm,
    tools=[python_repl],
    prompt=(
        "You are a code executor. You MUST call the python_repl tool to run code.\n"
        "The user gives you Python code. Call python_repl with that exact code.\n"
        "Do NOT write code in your reply — call the tool instead.\n"
        "After the tool returns, briefly say what happened.\n"
    ),
    name="code_executor",
)


def code_executor_node(state: AgentState) -> Command[Literal["supervisor"]]:
    msgs = state["messages"][-2:] if len(state["messages"]) > 2 else state["messages"]
    result = _code_agent.invoke({"messages": msgs})
    content = result["messages"][-1].content if result["messages"] else "No output"
    return Command(
        update={"messages": [HumanMessage(content=content, name="code_executor")]},
        goto="supervisor",
    )


# ── Validator ────────────────────────────────────────────────────────────────
_val_agent = create_react_agent(
    model=llm,
    tools=[check_file],
    prompt=(
        "You check files. Call the check_file tool with the filepath.\n"
        "If result starts with OK → reply APPROVED.\n"
        "Otherwise → reply REJECTED.\n"
    ),
    name="validator",
)


def validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    msgs = state["messages"][-1:] if state["messages"] else []
    result = _val_agent.invoke({"messages": msgs})
    content = result["messages"][-1].content if result["messages"] else "REJECTED"
    return Command(
        update={"messages": [HumanMessage(content=content, name="validator")]},
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
