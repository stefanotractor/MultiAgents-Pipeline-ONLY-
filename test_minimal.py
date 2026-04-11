from openai import OpenAI
import sys, io, traceback

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def run_python(code):
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        exec(code, {})
        return mystdout.getvalue()
    except Exception:
        return traceback.format_exc()
    finally:
        sys.stdout = old_stdout

def ask_and_run(task, max_retries=2):
    print(f"\n{'='*50}")
    print(f"TASK: {task}")
    print('='*50)
    
    messages = [
        {"role": "system", "content": (
            "You are a Python data analyst. "
            "Reply with ONLY Python code, nothing else. "
            "No markdown, no backticks, no explanations. Just code."
        )},
        {"role": "user", "content": task}
    ]
    
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b-instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.0
        )
        
        code = response.choices[0].message.content.strip()
        # Rimuovi backticks se presenti
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        code = code.strip()
        
        print(f"\nAttempt {attempt+1} CODE:\n{code}\n")
        
        result = run_python(code)
        print(f"OUTPUT:\n{result}")
        
        if "Error" not in result and "Traceback" not in result:
            print("SUCCESS")
            return result
        
        if attempt < max_retries:
            print(f"-- Retrying ({attempt+1}/{max_retries})...")
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": f"Error: {result}\nFix the code. Reply with ONLY the corrected Python code."})
    
    print("FAILED after retries")
    return None

# ── Test ──
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/data/ALLARMI.csv' with pandas. "
    "Print the shape and the list of column names."
)