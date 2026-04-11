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
            max_tokens=1024,
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

# ── Task 1: Load and standardize ALLARMI.csv
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/data/ALLARMI.csv' with pandas. "
    "Print the shape. "
    "Standardize all column names to lowercase snake_case. "
    "Then remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
    "Remove duplicate rows using df.drop_duplicates() and print shape. "
    "For every non-numeric column: strip whitespace, then replace these values with NaN: 'N.D.', 'n.d.', '?', '-', '//', 'NULL', '', 'None'. "
    "Use pd.api.types.is_numeric_dtype() to check column types. "
    "For numeric columns fill NaN with 0. For non-numeric columns fill NaN with 'unknown'. "
    "Print columns names. "
    "Save to '/Users/stefanolosurdo/Desktop/files 2/output/allarmi_clean.csv' without index."
)

# ── Task 2: Load and standardize TIPOLOGIA_VIAGGIATORE.csv
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/data/TIPOLOGIA_VIAGGIATORE.csv' with pandas. "
    "Print the shape"
    "Standardize all column names to lowercase snake_case. "
    "Then remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
    "Remove duplicate rows using df.drop_duplicates() and print shape using: df2.shape. "
    "For every non-numeric column: strip whitespace, then replace these values with NaN: 'N.D.', 'n.d.', '?', '-', '//', 'NULL', '', 'None'. "
    "Use pd.api.types.is_numeric_dtype() to check column types. "
    "For numeric columns fill NaN with 0. For non-numeric columns fill NaN with 'unknown'. "
    "Print columns' names."
    "Save to '/Users/stefanolosurdo/Desktop/files 2/output/tipologia_clean.csv' without index."
)

# ── Task 3: Merge ──
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/output/allarmi_clean.csv' and "
    "'/Users/stefanolosurdo/Desktop/files 2/output/tipologia_clean.csv' with pandas. "
    "Find the common columns between the two dataframes and print them. "
    "Merge on the common columns using outer join. "
    "Remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
    "Print shape of the merged dataframe. "
    "For all string columns: convert all values to lowercase using: df = df.astype(str).apply(lambda x: x.str.lower()). "
    "Save to '/Users/stefanolosurdo/Desktop/files 2/output/merged_data.csv' without index."
)

# ── Task 4: Group by route ──
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/output/merged_data.csv' with pandas. "
    "Create a 'route' column by combining columns 'areoporto_partenza' and 'areoporto_arrivo' with '-'. "
    "Build an aggregation dict: for each column, use 'sum' if pd.api.types.is_numeric_dtype(), else 'first'. "
    "Group by 'route' using df.groupby('route').agg(agg_dict).reset_index(). "
    "Print shape. "
    "Save to '/Users/stefanolosurdo/Desktop/files 2/output/routes_summary.csv' without index."
)

# ── Task 5: Baseline statistics (globali) ──
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/output/routes_summary.csv' with pandas. "
    "Ensure 'allarmati' is numeric using pd.to_numeric(errors='coerce').fillna(0). "
    "Compute global mean and std of 'allarmati' across all routes. "
    "Add column 'rolling_mean_alarms' = global mean. "
    "Add column 'rolling_std_alarms' = global std. If std is 0, set it to 1. "
    "Add column 'z_score': (allarmati - rolling_mean_alarms) / rolling_std_alarms. "
    "Add column 'ratio_to_baseline': allarmati / rolling_mean_alarms. Replace inf with 0. "
    "Print global mean and std. "
    "Print shape. "
    "Print top 10 rows by z_score descending showing route, allarmati, rolling_mean_alarms, z_score. "
    "Save the full dataframe to '/Users/stefanolosurdo/Desktop/files 2/output/baseline_data.csv' without index."
)

# ── Task 6: Outlier Detection ──
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/output/baseline_data.csv' with pandas. "
    "Import IsolationForest from sklearn.ensemble. "
    "Ensure columns 'allarmati', 'z_score', 'ratio_to_baseline' are numeric using pd.to_numeric(errors='coerce').fillna(0). "
    "Replace inf and -inf with 0. "
    "Build feature matrix with columns: allarmati, z_score, ratio_to_baseline. "
    "model = IsolationForest(contamination=0.05, random_state=42). "
    "model.fit(feature_matrix). "
    "df['anomaly'] = model.predict(feature_matrix) == -1. "
    "Do NOT drop any columns. Keep all columns including allarmati, z_score, ratio_to_baseline. "
    "Print number of rows where anomaly is True, and total rows. "
    "Print top 10 rows where anomaly is True sorted by z_score descending: print df[df['anomaly']==True].nlargest(10,'z_score')[['route','allarmati','z_score']]. "
    "Save the full dataframe to '/Users/stefanolosurdo/Desktop/files 2/output/outlier_results.csv' without index."
)

# ── Task 7: Risk Profiling ──
ask_and_run(
    "Load '/Users/stefanolosurdo/Desktop/files 2/output/outlier_results.csv' with pandas. "
    "Import numpy as np. "
    "Filter only rows where anomaly is True. Print how many. "
    "If zero, save empty dataframe to risk_profiled.csv and print 'No anomalies'. "
    "Otherwise: "
    "rule_route = anom['ratio_to_baseline'] > 3.0. "
    "rule_zscore_high = anom['z_score'].abs() > 8. "
    "rule_zscore_med = (anom['z_score'].abs() > 5) & (~rule_zscore_high). "
    "conditions = [rule_route & rule_zscore_high, rule_route | rule_zscore_high, rule_zscore_med]. "
    "choices = ['CRITICAL', 'HIGH', 'MEDIUM']. "
    "anom['risk_level'] = np.select(conditions, choices, default='LOW'). "
    "Print risk_level value_counts. "
    "Save full dataframe to '/Users/stefanolosurdo/Desktop/files 2/output/risk_profiled.csv' without index."
)
