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
    return _build_structural_cleaning_prompt(
        ALLARMI_CSV,
        f"{OUTPUT_DIR}/allarmi_clean.csv"
    )


def _build_data_prompt_2():
    return _build_structural_cleaning_prompt(
        TIPOLOGIA_CSV,
        f"{OUTPUT_DIR}/tipologia_clean.csv"
    )


# ─────────────────────────────────────────
# SEMANTIC NORMALIZATION AGENT
# ─────────────────────────────────────────

def _build_semantic_normalization_prompt(input_path, output_path):
    return (
        f"You are a semantic normalization agent for tabular datasets already cleaned at schema level. "

        f"Load the dataset at '{input_path}' and produce a semantically refined version saved to '{output_path}'. "

        f"Work autonomously and infer the required Python libraries. "

        f"The dataset already has a safe schema, so do not rename, reorder, merge, split, or drop columns unless this is strictly necessary to preserve loadability. "
        f"Focus only on intra-column semantic consistency. "

        f"For each column, inspect the observed non-null values and infer the dominant representation standard from the data itself. "
        f"Ignore values that are semantically missing, undefined, or non-informative when inferring the dominant representation and when performing normalization. "
        f"Determine that standard using the prevailing format, notation, structure, length patterns, punctuation, spacing, casing, and semantic consistency shown by the majority of valid values in the column. "
        f"Identify minority variants that are likely to represent the same underlying concept but appear in a different representation. "
        f"Normalize those minority variants to the dominant standard only when the mapping is clear, high-confidence, and strongly supported by the column pattern. "

        f"Do not rely only on the single most frequent value. "
        f"Do not collapse distinct categorical values merely because they look superficially similar. "
        f"If a value is unusual but not safely mappable to the dominant standard, preserve it and report it as difform rather than correcting it aggressively. "

        f"Do not alter numeric meaning, entity identity, or business semantics without strong evidence from the column distribution. "

        f"For each column that is normalized, report the original minority representations that were mapped to the dominant standard, and report which difform values were preserved because they were not safely correctable. "

        f"Before saving, validate that the dataframe remains non-empty and loadable. "
        f"Do not save any output if validation fails. "
        f"Save the semantically refined dataset to '{output_path}' without index. "
        f"Ensure that the dataset is loaded, processed, and saved within the same execution flow. "
        f"The output file must always be written during execution. "
        f"Avoid defining execution entry points or structures that require explicit invocation. "
        f"Assume that the code will be executed exactly as written, so all steps must run immediately. "    
)

# ── Task 3: Merge ────────────────────────────────────────────────────────────
## def _build_merge_prompt():
    return (
        f"Load '{OUTPUT_DIR}/allarmi_clean.csv' and "
        f"'{OUTPUT_DIR}/tipologia_clean.csv' with pandas. "
        f"Find the common columns between the two dataframes and print them. "
        f"Merge on the common columns using outer join. "
        f"Remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
        f"Print shape of the merged dataframe. "
        f"Save to '{OUTPUT_DIR}/merged_data.csv' without index."
    )

## def _build_merge_prompt():
    return (
        f"You are a data integration agent responsible for combining two cleaned tabular datasets into a single unified dataset for downstream anomaly detection. "

        f"Load the dataset at '{OUTPUT_DIR}/allarmi_clean.csv' and the dataset at '{OUTPUT_DIR}/tipologia_clean.csv' using pandas. "

        f"Work autonomously and infer the required Python libraries. "

        f"First, inspect both dataframes independently: print their shapes and column names. "
        f"Then identify the common columns between the two dataframes and print them explicitly before proceeding. "

        f"Merge the two dataframes on the common columns using an outer join to preserve all records from both sources. "
        f"After merging, remove duplicate columns using: df = df.loc[:, ~df.columns.duplicated()]. "
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

    ("semantic_allarmi", _build_semantic_normalization_prompt(
        f"{OUTPUT_DIR}/allarmi_clean.csv",
        f"{OUTPUT_DIR}/allarmi_semantic.csv",
    )),
    ("semantic_tipologia", _build_semantic_normalization_prompt(
        f"{OUTPUT_DIR}/tipologia_clean.csv",
        f"{OUTPUT_DIR}/tipologia_semantic.csv",
    )),

    ("merge",                  _build_merge_prompt()),
    ("baseline_grouping",      _build_baseline_prompt()),
    ("baseline_stats",         _build_baseline_stats_prompt()),
    ("outlier_detection",      _build_outlier_prompt()),
    ("risk_profiling",         _build_risk_prompt()),
    ("report_generation",      _build_report_prompt()),
]
