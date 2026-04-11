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
            model="google/gemma-3-4b",
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

# ── Task 1: Load e standardize ──py
ask_and_run(
    "Load '/Users/matteo/Desktop/MultiAgents-Pipeline-ONLY-/data/ALLARMI.csv' and "
    "'/Users/matteo/Desktop/MultiAgents-Pipeline-ONLY-/data/TIPOLOGIA_VIAGGIATORE.csv' with pandas. "
    "Standardize all column names to lowercase snake_case. "
    "Print columns dtypes"
    "For each numeric column fill missing values with 0"
    "For each string column fill missing values with Nan"
    "For each object column print value_counts"
    "Print dtypes to confirm."
)

# ── Task 2: Check types e convert ──
ask_and_run(
    "Load '/Users/matteo/Desktop/MultiAgents-Pipeline-ONLY-/data/ALLARMI.csv' with pandas. "
    "Standardize column names to lowercase snake_case. "
    "Print columns dtypes"
    "For each numeric column fill missing values with 0"
    "For each string column fill missing values with Nan"
    "For each object column print value_counts"
    "Convert columns that look numeric but are stored as string to int or float, fill NaN with 0. "
    "Print dtypes to confirm."
)