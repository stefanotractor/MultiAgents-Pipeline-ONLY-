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


def _build_data_prompt():
    return (
        f"Load '{ALLARMI_CSV}' with pandas as df1."
        f"Print shape using df1.shape. "
        f"Standardize all column names to lowercase snake_case. "
        f"Then remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
        f"Remove duplicate rows using df.drop_duplicates() and print shape. "
        f"For every non-numeric column: strip whitespace, then replace these values with NaN: 'N.D.', 'n.d.', '?', '-', '//', 'NULL', '', 'None'. "
        f"Use pd.api.types.is_numeric_dtype() to check column types. "
        f"For numeric columns fill NaN with 0. For non-numeric columns fill NaN with 'unknown'. "
        f"Print columns names. "
        f"Save to '{OUTPUT_DIR}/allarmi_clean.csv' without index."
    )


def _build_data_prompt_2():
    return (
        f"Load '{TIPOLOGIA_CSV}' with pandas. "
        f"Print shape. "
        f"Standardize all column names to lowercase snake_case. "
        f"Then remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
        f"Remove duplicate rows using df.drop_duplicates() and print shape. "
        f"For every non-numeric column: strip whitespace, then replace these values with NaN: 'N.D.', 'n.d.', '?', '-', '//', 'NULL', '', 'None'. "
        f"Use pd.api.types.is_numeric_dtype() to check column types. "
        f"For numeric columns fill NaN with 0. For non-numeric columns fill NaN with 'unknown'. "
        f"Print columns names. "
        f"Save to '{OUTPUT_DIR}/tipologia_clean.csv' without index."
    )


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


def _build_baseline_prompt():
    return (
        f"Load '{OUTPUT_DIR}/merged_data.csv' with pandas. "
        f"Create a 'route' column by combining columns 'areoporto_partenza' and 'areoporto_arrivo' with '-'. "
        f"Build an aggregation dict: for each column, use 'sum' if pd.api.types.is_numeric_dtype(), else 'first'. "
        f"Group by 'route' using df.groupby('route').agg(agg_dict).reset_index(). "
        f"Print shape. "
        f"Save to '{OUTPUT_DIR}/routes_summary.csv' without index."
    )


def _build_outlier_prompt():
    algo = DEFAULT_OUTLIER_ALGORITHM
    contam = ISOLATION_FOREST_CONTAMINATION
    neighbors = LOF_N_NEIGHBORS
    zscore_t = ZSCORE_THRESHOLD
    return (
        f"Load '{OUTPUT_DIR}/routes_summary.csv' with pandas. "
        f"Ensure 'allarmati' column is numeric using pd.to_numeric(errors='coerce').fillna(0). "
        f"Compute z_score for 'allarmati': (value - mean) / std. Replace inf with 0. "
        f"Compute ratio_to_baseline: value / mean. Replace inf with 0. "
        f"Build feature matrix with columns: allarmati, z_score, ratio_to_baseline. "
        f"Apply {algo} for anomaly detection. "
        f"If IsolationForest: use contamination={contam}. "
        f"If LOF: use n_neighbors={neighbors}. "
        f"If zscore: flag rows where abs(z_score) > {zscore_t}. "
        f"Add 'anomaly' boolean column. "
        f"Print number of anomalies and total rows. "
        f"Save to '{OUTPUT_DIR}/outlier_results.csv' without index."
    )


def _build_risk_prompt():
    mult = ALERT_RATE_MULTIPLIER
    return (
        f"Load '{OUTPUT_DIR}/outlier_results.csv' with pandas. "
        f"Filter only rows where anomaly is True. "
        f"If no anomalies, save empty dataframe to '{OUTPUT_DIR}/risk_profiled.csv' and print 'No anomalies'. "
        f"Otherwise apply these rules: "
        f"rule_route: ratio_to_baseline > {mult}. "
        f"rule_zscore_high: abs(z_score) > 8. "
        f"rule_zscore_med: abs(z_score) > 5 and not rule_zscore_high. "
        f"Risk level: CRITICAL if rule_route AND rule_zscore_high, "
        f"HIGH if rule_route OR rule_zscore_high, "
        f"MEDIUM if rule_zscore_med, else LOW. "
        f"Use np.select for conditions. "
        f"Print risk_level value_counts. "
        f"Save to '{OUTPUT_DIR}/risk_profiled.csv' without index."
    )


def _build_report_prompt():
    return (
        f"Load '{OUTPUT_DIR}/risk_profiled.csv' with pandas. "
        f"Take top 5 rows by z_score. "
        f"For each row, call LM Studio API (OpenAI SDK, base_url='{LM_STUDIO_BASE_URL}', "
        f"api_key='{LM_STUDIO_API_KEY}', model='{LM_STUDIO_MODEL}') "
        f"asking to explain the transit anomaly in 2 sentences. "
        f"Build a text report with header 'TRANSIT ANOMALY REPORT' and date. "
        f"Save report to '{OUTPUT_DIR}/anomaly_report.txt'. "
        f"Save JSON version to '{OUTPUT_DIR}/anomaly_report.json'. "
        f"Print the report."
    )


# ── Build task list ──────────────────────────────────────────────────────────
TASKS = [
    ("data_loading_allarmi",   _build_data_prompt()),
    ("data_loading_tipologia", _build_data_prompt_2()),
    ("merge",                  _build_merge_prompt()),
    ("baseline_building",      _build_baseline_prompt()),
    ("outlier_detection",      _build_outlier_prompt()),
    ("risk_profiling",         _build_risk_prompt()),
    ("report_generation",      _build_report_prompt()),
]