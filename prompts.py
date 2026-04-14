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
    FINDINGS_JSON,
)

# ─────────────────────────────────────────
# Findings persistence helper
# ─────────────────────────────────────────

def _findings_guidance(task_key: str, extra_notes: str = "") -> str:
    base = (
        f"Maintain a shared findings JSON at '{FINDINGS_JSON}'. "
        f"At the start, attempt to load it; if missing, empty, invalid, or corrupted, initialize an empty dict instead of failing. "
        f"Store new information under the key '{task_key}' while preserving existing keys for other tasks. "
        f"Use concise, machine-readable fields with only native Python JSON-serializable types such as dict, list, str, int, float, bool, or null. "
        f"Convert pandas and numpy values to native Python types before saving. "
        f"Convert tuples to lists before saving. "
        f"After completing the task, update the entry for '{task_key}' and write the full JSON back by overwriting the file. "
    )
    if extra_notes:
        base += extra_notes
    return base

# ─────────────────────────────────────────
# STRUCTURAL CLEANING (reusable template)
# ─────────────────────────────────────────

def _build_structural_cleaning_prompt(input_path, output_path):
    return (
        f"You are a data cleaning agent for tabular datasets used in downstream merge and anomaly detection tasks. "

        f"Load the dataset at '{input_path}' and produce a cleaned version saved to '{output_path}'. "

        f"Work autonomously and infer the required Python libraries. "

        f"First inspect the raw schema and print the original shape and original column names. "
        f"Then standardize column names to lowercase snake_case. "

        f"Immediately after renaming, resolve duplicate column names deterministically by occurrence order. "
        f"When duplicate column labels are present, rebuild the full ordered list of column names so that each occurrence gets a unique stable name. "
        f"Do not use ambiguous renaming by duplicated label alone. "
        f"Do not continue until column names are truly unique and explicitly verified. "

        f"After the schema is safe, clean values column by column. "
        f"The cleaned dataset must have all column names in lowercase snake_case and all textual values in lowercase. "
        f"This is a mandatory output requirement for the cleaned file. "
        f"For every column, apply text normalization to all values that are textual, regardless of how the column type is internally represented. "
        f"Do not rely on a specific column type such as 'object' to detect text. "
        f"Instead, treat any value that behaves like text as textual content and normalize it. "
        f"Convert all textual values to lowercase, remove outer whitespace, and normalize repeated internal whitespace. "
        f"Preserve null values as null. "
        f"If a column contains both text and non-text values, normalize the textual cells to lowercase and leave the non-text cells unchanged. "
        f"Do not skip lowercase normalization for textual values because of mixed content in the same column. "
        f"Do not change the semantic type of non-text values only for convenience. "
        f"The lowercase normalization of textual values must be applied consistently across all datasets, even if different internal data types are used. "
        f"Use one shared and globally defined set of missing-value patterns before any column-level cleaning logic starts. "
        f"Do not define cleaning rules or missing-value rules only inside conditional branches if they are reused elsewhere. "
        f"Keep the cleaning flow structurally simple and deterministic across datasets. "
        f"Use the same cleaning strategy for every dataset of this task category, rather than inventing dataset-specific logic unless strictly necessary for loadability. "
        f"Avoid branch-specific variables that may be unavailable in other branches. "
        f"The generated code must run top-to-bottom without relying on variables defined only inside a conditional block. "
        f"Detect values that semantically represent missing, undefined, or non-informative entries by analyzing the column content, and convert them into proper missing values where appropriate. "
        f"Then remove duplicate rows. "

        f"Handle missing values conservatively and preserve missingness when imputation could distort downstream anomaly detection. "

        f"Before saving, validate that the dataframe is non-empty and has unique column names. "
        f"Do not save any output if those checks fail. "

        f"Print only a short summary with: original shape, final shape, final column names, duplicate-name collisions handled, and duplicate rows removed. "
    )


# ─────────────────────────────────────────
# DATASET-SPECIFIC CLEANING AGENTS
# ─────────────────────────────────────────

def _build_data_prompt():
    return (
        _build_structural_cleaning_prompt(
            ALLARMI_CSV,
            f"{OUTPUT_DIR}/allarmi_clean.csv"
        )
        + _findings_guidance(
            "data_loading_allarmi",
            "Capture shape_before, shape_after, columns_final, duplicate_name_collisions, duplicate_rows_removed. "
        )
    )


def _build_data_prompt_2():
    return (
        _build_structural_cleaning_prompt(
            TIPOLOGIA_CSV,
            f"{OUTPUT_DIR}/tipologia_clean.csv"
        )
        + _findings_guidance(
            "data_loading_tipologia",
            "Capture shape_before, shape_after, columns_final, duplicate_name_collisions, duplicate_rows_removed. "
        )
    )


# ─────────────────────────────────────────
# SEMANTIC NORMALIZATION AGENT
# ─────────────────────────────────────────

def _build_semantic_normalization_prompt(input_path, output_path, findings_key):
    return (
        f"You are a semantic normalization agent for tabular datasets already cleaned at schema level. "
        f"Load the dataset at '{input_path}' and produce a semantically refined version saved to '{output_path}'. "

        f"Work autonomously and infer the required Python libraries. "

        f"The dataset already has a safe schema, so do not rename, reorder, merge, split, or drop columns unless strictly necessary. "
        f"Focus only on intra-column semantic consistency. "

        f"For each column, inspect non-null values and infer the dominant representation. "
        f"Ignore missing or non-informative values when inferring patterns. "

        f"Normalize only high-confidence minority variants. "
        f"Do not collapse distinct categories. "
        f"Preserve uncertain or difform values and report them. "

        f"Before saving, validate that the dataframe is non-empty and still loadable. "
        f"The dataset must be loaded, processed, and saved within the same execution flow. "
        f"All steps must run automatically without requiring any explicit invocation or entry point. "
        f"If validation passes, write the output file immediately using to_csv. "
        f"Do not check whether the output file already exists before writing it. "
        f"The output file must always be created during execution if the dataframe is valid. "

        f"Save the semantically refined dataset to '{output_path}' without index. "
        + _findings_guidance(
            findings_key,
            "Include shape_before, shape_after, normalization_mappings, preserved_difform_values. "
        )
    )

def _build_merge_prompt():
    return (
        f"You are a data integration agent responsible for combining two cleaned tabular datasets into a single unified dataset for downstream anomaly detection. "

        f"Load the dataset at '{OUTPUT_DIR}/allarmi_clean.csv' and the dataset at '{OUTPUT_DIR}/tipologia_clean.csv' using pandas. "

        f"Work autonomously and infer the required Python libraries. "

        f"First, inspect both dataframes independently: print their shapes and column names. "
        f"Then identify the common columns between the two dataframes and print them explicitly before proceeding. "

        f"Merge the two dataframes on the common columns using an outer join to preserve all records from both sources. "
        f"After merging, ensure that no duplicate columns remain in the result. "
        f"Do not drop or rename any column unless it is a confirmed structural duplicate introduced by the merge. "

        f"Do not perform any imputation, encoding, or transformation on the merged data. "
        f"Preserve the original values and types from both sources as-is. "

        f"Before saving, validate that the merged dataframe is non-empty and has unique column names. "
        f"Do not save any output if those checks fail. "

        f"Save the merged dataset to '{OUTPUT_DIR}/merged_data.csv' without index. "
        f"Ensure that the dataset is loaded, processed, and saved within the same execution flow. "
        f"Avoid defining execution entry points or structures that require explicit invocation. "
        f"Assume that the code will be executed exactly as written, so all steps must run immediately. "

        f"Print only a short summary with: shape of allarmi_clean, shape of tipologia_clean, common columns used for merge, final merged shape, and duplicate columns removed. "
        + _findings_guidance(
            "merge",
            "Record shapes_allarmi_tipologia, common_columns, merged_shape, duplicate_columns_removed. "
        )
    )

# ── Task 4: Group by route ──────────────────────────────────────────────────
def _build_baseline_prompt():
    return (
        f"You are a data aggregation agent responsible for building a route-level summary dataset to be used as baseline input for downstream anomaly detection. "

        f"Load the dataset at '{OUTPUT_DIR}/merged_data.csv'. "

        f"Work autonomously and infer the required Python libraries. "

        f"First, inspect the loaded dataframe: print its shape and column names before proceeding. "

        f"Create a 'route' column by combining the values of 'areoporto_partenza' and 'areoporto_arrivo'. "
        f"Do not drop the original airport columns after creating the route column. "

        f"Build the aggregation strategy for every column except 'route'. "
        f"Do not hardcode column names when defining the aggregation strategy. "
        f"Determine whether a column is numeric using pandas-native type inspection, not numpy subtype checks. "
        f"For numeric columns, aggregate by summing their values. "
        f"For non-numeric columns, aggregate by taking the first observed value. "
        f"Do not apply any transformation or normalization to the aggregated values. "

        f"Group the dataframe by 'route' and aggregate all other columns according to the strategy above. "
        f"After aggregation, ensure that 'route' is present as a standard column in the final dataframe and not only as an index. "

        f"Before saving, validate that the resulting dataframe is non-empty and has unique column names. "
        f"Do not save any output if those checks fail. "

        f"Save the aggregated dataset to '{OUTPUT_DIR}/routes_summary.csv' without index. "
        f"Ensure that the dataset is loaded, processed, and saved within the same execution flow. "
        f"Avoid defining execution entry points or structures that require explicit invocation. "
        f"Assume that the code will be executed exactly as written, so all steps must run immediately. "

        f"Print only a short summary with: original merged shape, number of unique routes found, final routes_summary shape. "
        + _findings_guidance(
            "baseline_grouping",
            "Store merged_shape, unique_routes, aggregation_strategy_summary (list of numeric/non_numeric columns), routes_summary_shape. "
        )
    )

# ── Task 5: Baseline statistics ──────────────────────────────────────────────
def _build_baseline_stats_prompt():
    return (
        f"Load '{OUTPUT_DIR}/routes_summary.csv' with pandas. "
        f"Import numpy as np. "
        f"Ensure 'allarmati' is numeric using pd.to_numeric(errors='coerce').fillna(0). "
        f"Compute global mean and std of 'allarmati' across all routes. "
        f"Add column 'rolling_mean_alarms' = global mean. "
        f"Add column 'rolling_std_alarms' = global std. If std is 0, set it to 1. "
        f"Add column 'z_score': (allarmati - rolling_mean_alarms) / rolling_std_alarms. "
        f"Add column 'ratio_to_baseline': allarmati / rolling_mean_alarms. Replace inf and -inf with 0. "
        f"Print global mean and std. "
        f"Print shape. "
        f"Print top 10 rows by z_score descending showing route, allarmati, rolling_mean_alarms, z_score. "
        f"Save the full dataframe to '{OUTPUT_DIR}/baseline_data.csv' without index. "
        + _findings_guidance(
            "baseline_stats",
            "Store global_mean_allarmati, global_std_allarmati, baseline_shape, top_routes_by_z (list of dicts with route, allarmati, z_score). "
        )
    )

# ── Task 6: Outlier Detection ───────────────────────────────────────────────
def _build_outlier_prompt():
    algo = DEFAULT_OUTLIER_ALGORITHM
    contam = ISOLATION_FOREST_CONTAMINATION
    neighbors = LOF_N_NEIGHBORS
    zscore_t = ZSCORE_THRESHOLD

    return (
        f"You are an outlier detection agent operating on a route-level baseline dataset. "

        f"Load the dataset at '{OUTPUT_DIR}/baseline_data.csv'. "

        f"If a shared findings JSON exists at '{FINDINGS_JSON}', load it and reuse relevant information from previous steps "
        f"(especially baseline statistics and column validation). Continue even if it is missing. "

        f"Work autonomously and infer the required Python libraries. "

        f"First inspect the dataset: print shape and column names. "

        f"Ensure that the columns representing engineered features are valid, numeric, and usable for modeling. "
        f"The expected features are 'allarmati', 'z_score', and 'ratio_to_baseline'. "
        f"Coerce invalid values to numeric form and handle non-finite values safely so that the dataset is model-ready. "

        f"Construct a feature matrix using exactly these three features, preserving row alignment with the original dataset. "

        + (
            f"Apply an Isolation Forest model using a contamination level of {contam} to detect anomalies. "
            if algo == "IsolationForest" else
            f"Apply a Local Outlier Factor model using {neighbors} neighbors and contamination {contam}. "
            if algo == "LOF" else
            f"Detect anomalies using a z-score threshold of {zscore_t} on the absolute value of z_score. "
        )

        + f"Store the result in a new boolean column named 'anomaly'. "
        f"Do not drop any columns or rows. Preserve the full dataset. "

        f"Print a short summary with: total number of rows and number of detected anomalies. "

        f"Print the top 10 anomalous rows sorted by highest anomaly severity using z_score, "
        f"displaying only 'route', 'allarmati', and 'z_score'. "

        f"Before saving, validate that the dataframe is non-empty and consistent. "
        f"If validation passes, save the dataset to '{OUTPUT_DIR}/outlier_results.csv' without index. "

        f"The dataset must be loaded, processed, and saved within the same execution flow. "
        f"All steps must run automatically without requiring explicit invocation. "

        + _findings_guidance(
            "outlier_detection",
            "Store total_rows as int, anomaly_rows as int, algorithm_used as string, "
            "feature_columns as list of strings, and top_anomalies as a list of plain Python dicts with route, allarmati, z_score. "
        )
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
        + _findings_guidance(
            "risk_profiling",
            "Store anomaly_rows, risk_level_counts, rules_used (text). "
        )
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
        + _findings_guidance(
            "report_generation",
            "Store top_anomalies_in_report (list), report_txt_path, report_json_path. "
        )
    )


# ── Build task list ──────────────────────────────────────────────────────────
TASKS = [
    ("data_loading_allarmi",   _build_data_prompt()),
    ("data_loading_tipologia", _build_data_prompt_2()),

    ("semantic_allarmi", _build_semantic_normalization_prompt(
        f"{OUTPUT_DIR}/allarmi_clean.csv",
        f"{OUTPUT_DIR}/allarmi_semantic.csv",
        "semantic_allarmi",
    )),
    ("semantic_tipologia", _build_semantic_normalization_prompt(
        f"{OUTPUT_DIR}/tipologia_clean.csv",
        f"{OUTPUT_DIR}/tipologia_semantic.csv",
        "semantic_tipologia",
    )),

    ("merge",                  _build_merge_prompt()),
    ("baseline_grouping",      _build_baseline_prompt()),
    ("baseline_stats",         _build_baseline_stats_prompt()),
    ("outlier_detection",      _build_outlier_prompt()),
    ("risk_profiling",         _build_risk_prompt()),
    ("report_generation",      _build_report_prompt()),
]