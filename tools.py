"""
Python REPL tool + file checker for agents.
"""
import sys
import io
import traceback
from typing import Annotated
from langchain_core.tools import tool


class PythonREPL:
    """Executes Python code and captures output."""
    def __init__(self):
        self.globals = {}

    def run(self, code: str) -> str:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = mystdout = io.StringIO()
        sys.stderr = mystderr = io.StringIO()
        try:
            exec(code, self.globals)
            output = mystdout.getvalue()
            errors = mystderr.getvalue()
            if errors:
                output += f"\nSTDERR:\n{errors}"
            return output if output else "(OK, no output)"
        except Exception:
            return f"Error:\n{traceback.format_exc()}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


_repl = PythonREPL()


@tool
def python_repl(
    code: Annotated[str, "Python code to execute."],
) -> str:
    """Execute Python code. Use print() to see output. Variables persist between calls."""
    try:
        result = _repl.run(code)
    except BaseException as e:
        return f"Failed: {repr(e)}"
    # Keep response short for small models
    if len(result) > 3000:
        result = result[:3000] + "\n... [truncated]"
    return result


@tool
def check_file(
    filepath: Annotated[str, "File path to check."],
) -> str:
    """Check if a file exists and show its size."""
    import os
    if not os.path.exists(filepath):
        return f"NOT_FOUND: {filepath}"
    size = os.path.getsize(filepath)
    if size == 0:
        return f"EMPTY: {filepath} (0 bytes)"
    return f"OK: {filepath} ({size} bytes)"
