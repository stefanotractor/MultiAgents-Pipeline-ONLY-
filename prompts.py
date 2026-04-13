"""
Natural language prompts for each pipeline task.
The LLM receives these and generates Python code autonomously.
"""

from config import (
    ALLARMI_CSV,
    TIPOLOGIA_CSV,
    DEFAULT_OUTLIER_ALGORITHM,
    ISOLATION_FOREST_CONTAMINATION,
    LOF_N_NEIGHBORS,
    ZSCORE_THRESHOLD,
    ALERT_RATE_MULTIPLIER,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_MODEL,
    OUTPUT_DIR,
)


# ── Task 1: Load and clean ALLARMI.csv ───────────────────────────────────────
def _build_data_prompt():
    return (
        f"Load '{ALLARMI_CSV}' with pandas. "
        f"Print the shape. "
        f"Standardize all column names to lowercase snake_case. "
        f"Normalize all text/string columns to lowercase for consistent matching. Handle NaN and mixed-type values gracefully — only apply lowercase to actual string values, skip nulls. For example, 'ROME', 'Rome', and 'rome' should all become 'rome', while NaN stays NaN."
        f"Then remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
        f"Remove duplicate rows using df.drop_duplicates() and print shape. "
        f"For every non-numeric column: strip whitespace, then replace these values with NaN: 'N.D.', 'n.d.', '?', '-', '//', 'NULL', '', 'None'. "
        f"Use pd.api.types.is_numeric_dtype() to check column types. "
        f"For numeric columns fill NaN with 0. For non-numeric columns fill NaN with 'unknown'. "
        f"Print columns names. "
        f"Save to '{OUTPUT_DIR}/allarmi_clean.csv' without index."
    )


# ── Task 2: Load and clean TIPOLOGIA_VIAGGIATORE.csv ─────────────────────────
def _build_data_prompt_2():
    return (
        f"Load '{TIPOLOGIA_CSV}' with pandas. "
        f"Print the shape. "
        f"Standardize all column names to lowercase snake_case. "
        f"Normalize all text/string columns to lowercase for consistent matching. Handle NaN and mixed-type values gracefully — only apply lowercase to actual string values, skip nulls. For example, 'ROME', 'Rome', and 'rome' should all become 'rome', while NaN stays NaN."
        f"Then remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
        f"Remove duplicate rows using df.drop_duplicates() and print shape. "
        f"For every non-numeric column: strip whitespace, then replace these values with NaN: 'N.D.', 'n.d.', '?', '-', '//', 'NULL', '', 'None'. "
        f"Use pd.api.types.is_numeric_dtype() to check column types. "
        f"For numeric columns fill NaN with 0. For non-numeric columns fill NaN with 'unknown'. "
        f"Print columns names. "
        f"Save to '{OUTPUT_DIR}/tipologia_clean.csv' without index."
    )


# ── Task 3: Merge ────────────────────────────────────────────────────────────
def _build_merge_prompt():
    return (
        f"Load '{OUTPUT_DIR}/allarmi_clean.csv' and "
        f"'{OUTPUT_DIR}/tipologia_clean.csv' with pandas. "
        f"Find the common columns between the two dataframes and print them. "
        f"Merge on the common columns using outer join. "
        f"Remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
        f"Print shape of the merged dataframe. "
        f"Save to '{OUTPUT_DIR}/merged_data.csv' without index."
    )


# ── Task 4: Group by route ──────────────────────────────────────────────────
def _build_baseline_prompt():
    return (
        f"Load '{OUTPUT_DIR}/merged_data.csv' with pandas. "
        f"Create a 'route' column by combining columns 'areoporto_partenza' and 'areoporto_arrivo' with '-'. "
        f"Build an aggregation dict: for each column, use 'sum' if pd.api.types.is_numeric_dtype(), else 'first'. "
        f"Group by 'route' using df.groupby('route').agg(agg_dict).reset_index(). "
        f"Print shape. "
        f"Save to '{OUTPUT_DIR}/routes_summary.csv' without index."
    )


# ── Task 5: Baseline statistics ──────────────────────────────────────────────
def _build_baseline_stats_prompt():
    return (
        f"Load '{OUTPUT_DIR}/routes_summary.csv' with pandas. "
        f"Ensure 'allarmati' is numeric using pd.to_numeric(errors='coerce').fillna(0). "
        f"Compute global mean and std of 'allarmati' across all routes. "
        f"Add column 'rolling_mean_alarms' = global mean. "
        f"Add column 'rolling_std_alarms' = global std. If std is 0, set it to 1. "
        f"Add column 'z_score': (allarmati - rolling_mean_alarms) / rolling_std_alarms. "
        f"Add column 'ratio_to_baseline': allarmati / rolling_mean_alarms. Replace inf with 0. "
        f"Print global mean and std. "
        f"Print shape. "
        f"Print top 10 rows by z_score descending showing route, allarmati, rolling_mean_alarms, z_score. "
        f"Save the full dataframe to '{OUTPUT_DIR}/baseline_data.csv' without index."
    )


# ── Task 6: Outlier Detection ───────────────────────────────────────────────
def _build_outlier_prompt():
    algo = DEFAULT_OUTLIER_ALGORITHM
    contam = ISOLATION_FOREST_CONTAMINATION
    neighbors = LOF_N_NEIGHBORS
    zscore_t = ZSCORE_THRESHOLD
    return (
        f"Load '{OUTPUT_DIR}/baseline_data.csv' with pandas. "
        f"Import IsolationForest from sklearn.ensemble. "
        f"Ensure columns 'allarmati', 'z_score', 'ratio_to_baseline' are numeric using pd.to_numeric(errors='coerce').fillna(0). "
        f"Replace inf and -inf with 0. "
        f"Build feature matrix with columns: allarmati, z_score, ratio_to_baseline. "
        + (
            f"model = IsolationForest(contamination={contam}, random_state=42). "
            f"model.fit(feature_matrix). "
            f"df['anomaly'] = model.predict(feature_matrix) == -1. "
            if algo == "IsolationForest" else
            f"from sklearn.neighbors import LocalOutlierFactor. "
            f"model = LocalOutlierFactor(n_neighbors={neighbors}, contamination={contam}). "
            f"df['anomaly'] = model.fit_predict(feature_matrix) == -1. "
            if algo == "LOF" else
            f"df['anomaly'] = df['z_score'].abs() > {zscore_t}. "
        )
        + f"Do NOT drop any columns. Keep all columns including allarmati, z_score, ratio_to_baseline. "
        f"Print number of rows where anomaly is True, and total rows. "
        f"Print top 10 rows where anomaly is True sorted by z_score descending: print df[df['anomaly']==True].nlargest(10,'z_score')[['route','allarmati','z_score']]. "
        f"Save the full dataframe to '{OUTPUT_DIR}/outlier_results.csv' without index."
    )


# ── Task 7: Risk Profiling ──────────────────────────────────────────────────
def _build_risk_prompt():
    mult = ALERT_RATE_MULTIPLIER
    return (
        f"Load '{OUTPUT_DIR}/outlier_results.csv' with pandas. "
        f"Import numpy as np. "
        f"Filter only rows where anomaly is True. Print how many. "
        f"If zero, save empty dataframe to '{OUTPUT_DIR}/risk_profiled.csv' and print 'No anomalies'. "
        f"Otherwise: "
        f"rule_route = anom['ratio_to_baseline'] > {mult}. "
        f"rule_zscore_high = anom['z_score'].abs() > 8. "
        f"rule_zscore_med = (anom['z_score'].abs() > 5) & (~rule_zscore_high). "
        f"conditions = [rule_route & rule_zscore_high, rule_route | rule_zscore_high, rule_zscore_med]. "
        f"choices = ['CRITICAL', 'HIGH', 'MEDIUM']. "
        f"anom['risk_level'] = np.select(conditions, choices, default='LOW'). "
        f"Print risk_level value_counts. "
        f"Save full dataframe to '{OUTPUT_DIR}/risk_profiled.csv' without index."
    )


# ── Task 8: Report (placeholder — da implementare successivamente) ───────────
def _build_report_prompt():
    return (
        f"Load '{OUTPUT_DIR}/risk_profiled.csv' with pandas. "
        f"Sort by z_score descending and take top 5 rows. "
        f"Import OpenAI from openai. "
        f"Create client: client = OpenAI(base_url='{LM_STUDIO_BASE_URL}', api_key='{LM_STUDIO_API_KEY}'). "
        f"For each row, call client.chat.completions.create("
        f"model='{LM_STUDIO_MODEL}', "
        f"messages=[{{'role':'user','content':'Explain this transit anomaly in 2 sentences: ' + str(row.to_dict())}}], "
        f"max_tokens=150). "
        f"Get response text from response.choices[0].message.content. "
        f"Build a text report with header 'TRANSIT ANOMALY REPORT' and today's date. "
        f"Save report to '{OUTPUT_DIR}/anomaly_report.txt'. "
        f"Also save a JSON version with json.dump to '{OUTPUT_DIR}/anomaly_report.json'. "
        f"Print the report."
    )


# ── Build task list ──────────────────────────────────────────────────────────
TASKS = [
    ("data_loading_allarmi",   _build_data_prompt()),
    ("data_loading_tipologia", _build_data_prompt_2()),
    ("merge",                  _build_merge_prompt()),
    ("baseline_grouping",      _build_baseline_prompt()),
    ("baseline_stats",         _build_baseline_stats_prompt()),
    ("outlier_detection",      _build_outlier_prompt()),
    ("risk_profiling",         _build_risk_prompt()),
    ("report_generation",      _build_report_prompt()),
]
