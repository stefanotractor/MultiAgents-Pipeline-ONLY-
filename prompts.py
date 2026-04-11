"""
Code templates for each pipeline task — v2 Light.

Uses string concatenation instead of f-strings to avoid escaping issues
with regex patterns and dict literals inside code templates.
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
)


def _build_data_code():
    """
    Simplified data agent: hardcoded Python code for a deterministic merge.
    
    Strategy:
    - Load both CSVs
    - Normalize column names to lowercase snake_case
    - Identify shared columns automatically via set intersection
    - Clean merge keys (strip, uppercase for IATA codes)
    - Concatenate (row-bind) instead of join, since both datasets
      share the same flight-level grain but carry different measures
    - If true shared keys exist, do a left join; otherwise just concat
    - Save to merged_data.csv
    """
    allarmi = repr(ALLARMI_CSV)
    tipologia = repr(TIPOLOGIA_CSV)

    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# ── 1. Load ──────────────────────────────────────────────\n"
        "al = pd.read_csv(" + allarmi + ", dtype=str)\n"
        "tp = pd.read_csv(" + tipologia + ", dtype=str)\n"
        "print('ALLARMI raw:', al.shape)\n"
        "print('TIPOLOGIA raw:', tp.shape)\n"
        "\n"
        "# ── 2. Normalize column names ────────────────────────────\n"
        "def norm_cols(df):\n"
        "    df.columns = (df.columns.str.strip()\n"
        "                  .str.lower()\n"
        "                  .str.replace(' ', '_', regex=False)\n"
        "                  .str.replace('%', '_', regex=False)\n"
        "                  .str.replace('__', '_', regex=False))\n"
        "    df = df.loc[:, ~df.columns.duplicated()]\n"
        "    return df\n"
        "\n"
        "al = norm_cols(al)\n"
        "tp = norm_cols(tp)\n"
        "\n"
        "print('ALLARMI cols:', list(al.columns))\n"
        "print('TIPOLOGIA cols:', list(tp.columns))\n"
        "\n"
        "# ── 3. Find shared columns ───────────────────────────────\n"
        "shared = sorted(set(al.columns) & set(tp.columns))\n"
        "print('Shared columns:', shared)\n"
        "\n"
        "# ── 4. Basic cleaning on both DataFrames ─────────────────\n"
        "def clean_df(df):\n"
        "    for col in df.columns:\n"
        "        if df[col].dtype == object:\n"
        "            df[col] = df[col].str.strip()\n"
        "            # Replace common missing-value markers with NaN\n"
        "            df[col] = df[col].replace(\n"
        "                ['N.D.', 'n.d.', 'ND', 'nd', '?', '-', '//', 'NULL', 'null', 'None', ''], \n"
        "                np.nan\n"
        "            )\n"
        "    # Uppercase IATA airport codes if column exists\n"
        "    for col in ['areoporto_partenza', 'areoporto_arrivo']:\n"
        "        if col in df.columns:\n"
        "            df[col] = df[col].str.upper()\n"
        "    return df\n"
        "\n"
        "al = clean_df(al)\n"
        "tp = clean_df(tp)\n"
        "\n"
        "# ── 5. Merge strategy ────────────────────────────────────\n"
        "# Both datasets describe flights at different granularity.\n"
        "# ALLARMI = one row per flight (aggregate alarm counts).\n"
        "# TIPOLOGIA = one row per traveler on that flight.\n"
        "# We merge on shared flight-level keys so each traveler row\n"
        "# gets the alarm-level info from ALLARMI.\n"
        "#\n"
        "# Best merge keys: areoporto_partenza, areoporto_arrivo,\n"
        "# plus a date/time key if available.\n"
        "\n"
        "merge_keys = [c for c in ['areoporto_partenza', 'areoporto_arrivo',\n"
        "                          'anno_partenza', 'mese_partenza',\n"
        "                          'codice_paese_part', 'codice_paese_arr']\n"
        "              if c in shared]\n"
        "\n"
        "# Fallback: if no good keys, just use whatever is shared\n"
        "if len(merge_keys) < 2:\n"
        "    merge_keys = shared\n"
        "\n"
        "print('Merge keys:', merge_keys)\n"
        "\n"
        "if len(merge_keys) > 0:\n"
        "    df = tp.merge(al, on=merge_keys, how='left', suffixes=('', '_allarmi'))\n"
        "else:\n"
        "    # Last resort: just concatenate row-wise\n"
        "    df = pd.concat([al, tp], ignore_index=True)\n"
        "\n"
        "print('Merged shape:', df.shape)\n"
        "\n"
        "# ── 6. Convert numeric columns ──────────────────────────\n"
        "numeric_candidates = ['allarmati', 'tot', 'entrati', 'investigati',\n"
        "                      'anno_partenza', 'mese_partenza', 'gate']\n"
        "for col in numeric_candidates:\n"
        "    if col in df.columns:\n"
        "        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)\n"
        "\n"
        "# ── 7. Save ─────────────────────────────────────────────\n"
        "df.to_csv('merged_data.csv', index=False)\n"
        "print('Columns:', list(df.columns))\n"
        "print('Shape:', df.shape)\n"
        "print('Null counts:')\n"
        "print(df.isnull().sum().to_string())\n"
        "print('First 3 rows:')\n"
        "print(df.head(3).to_string())\n"
        "print('DONE: merged_data.csv saved')\n"
    )


def _build_baseline_code():
    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "df = pd.read_csv('merged_data.csv')\n"
        "print('Loaded:', df.shape)\n"
        "\n"
        "# Find alarm column\n"
        "alarm_col = None\n"
        "for c in ['allarmati','allarmati_trav','tot','tot_alarm']:\n"
        "    if c in df.columns:\n"
        "        alarm_col = c\n"
        "        break\n"
        "if alarm_col is None:\n"
        "    for c in df.columns:\n"
        "        if pd.api.types.is_numeric_dtype(df[c]):\n"
        "            alarm_col = c\n"
        "            break\n"
        "print('Alarm column:', alarm_col)\n"
        "df[alarm_col] = pd.to_numeric(df[alarm_col], errors='coerce').fillna(0)\n"
        "\n"
        "# Route key\n"
        "if 'areoporto_partenza' in df.columns and 'areoporto_arrivo' in df.columns:\n"
        "    df['route'] = df['areoporto_partenza'].astype(str) + '->' + df['areoporto_arrivo'].astype(str)\n"
        "else:\n"
        "    df['route'] = 'UNKNOWN'\n"
        "\n"
        "# Baselines per route\n"
        "stats = df.groupby('route')[alarm_col].agg(['mean','std','count']).reset_index()\n"
        "stats.columns = ['route','rolling_mean_alarms','rolling_std_alarms','count']\n"
        "stats['rolling_std_alarms'] = stats['rolling_std_alarms'].fillna(1)\n"
        "df = df.merge(stats[['route','rolling_mean_alarms','rolling_std_alarms']], on='route', how='left')\n"
        "df.rename(columns={alarm_col: 'actual_alarms'}, inplace=True)\n"
        "\n"
        "df.to_csv('baseline_data.csv', index=False)\n"
        "print('Top 5 routes:', stats.nlargest(5,'rolling_mean_alarms').to_string())\n"
        "print('DONE: baseline_data.csv saved')\n"
    )


def _build_outlier_code():
    algo = DEFAULT_OUTLIER_ALGORITHM
    contam = ISOLATION_FOREST_CONTAMINATION
    neighbors = LOF_N_NEIGHBORS
    zscore_t = ZSCORE_THRESHOLD

    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from sklearn.ensemble import IsolationForest\n"
        "from sklearn.neighbors import LocalOutlierFactor\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "df = pd.read_csv('baseline_data.csv')\n"
        "print('Loaded:', df.shape)\n"
        "\n"
        "df['actual_alarms'] = pd.to_numeric(df['actual_alarms'], errors='coerce').fillna(0)\n"
        "df['rolling_mean_alarms'] = pd.to_numeric(df['rolling_mean_alarms'], errors='coerce').fillna(0)\n"
        "df['rolling_std_alarms'] = pd.to_numeric(df['rolling_std_alarms'], errors='coerce').fillna(1)\n"
        "df['deviation'] = df['actual_alarms'] - df['rolling_mean_alarms']\n"
        "df['z_score'] = df['deviation'] / df['rolling_std_alarms'].replace(0, 1)\n"
        "df['ratio_to_baseline'] = df['actual_alarms'] / df['rolling_mean_alarms'].replace(0, 1)\n"
        "\n"
        "feats = df[['actual_alarms','deviation','z_score','ratio_to_baseline']].fillna(0)\n"
        "feats = feats.replace([np.inf, -np.inf], 0)\n"
        "\n"
        "algo = '" + algo + "'\n"
        "print('Algorithm:', algo)\n"
        "if algo == 'IsolationForest':\n"
        "    m = IsolationForest(contamination=" + str(contam) + ", random_state=42)\n"
        "    df['anomaly'] = m.fit_predict(feats) == -1\n"
        "elif algo == 'LOF':\n"
        "    m = LocalOutlierFactor(n_neighbors=" + str(neighbors) + ")\n"
        "    df['anomaly'] = m.fit_predict(feats) == -1\n"
        "else:\n"
        "    df['anomaly'] = df['z_score'].abs() > " + str(zscore_t) + "\n"
        "\n"
        "df.to_csv('outlier_results.csv', index=False)\n"
        "n = df['anomaly'].sum()\n"
        "print(f'Anomalies: {n}/{len(df)} ({n/len(df)*100:.1f}%)')\n"
        "print('Top 10:', df[df['anomaly']].nlargest(10,'z_score')[['route','actual_alarms','rolling_mean_alarms','z_score']].to_string())\n"
        "print('DONE: outlier_results.csv saved')\n"
    )


def _build_risk_code():
    mult = ALERT_RATE_MULTIPLIER
    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "df = pd.read_csv('outlier_results.csv')\n"
        "anom = df[df['anomaly'] == True].copy()\n"
        "print(f'Anomalies to profile: {len(anom)}')\n"
        "\n"
        "if len(anom) == 0:\n"
        "    pd.DataFrame().to_csv('risk_profiled.csv', index=False)\n"
        "    print('No anomalies. Empty file saved.')\n"
        "else:\n"
        "    anom['rule_route'] = anom['ratio_to_baseline'] > " + str(mult) + "\n"
        "    anom['rule_zscore_high'] = anom['z_score'].abs() > 8\n"
        "    anom['rule_zscore_med'] = (anom['z_score'].abs() > 5) & (~anom['rule_zscore_high'])\n"
        "    conditions = [\n"
        "        anom['rule_route'] & anom['rule_zscore_high'],\n"
        "        anom['rule_route'] | anom['rule_zscore_high'],\n"
        "        anom['rule_zscore_med'],\n"
        "    ]\n"
        "    choices = ['CRITICAL','HIGH','MEDIUM']\n"
        "    anom['risk_level'] = np.select(conditions, choices, default='LOW')\n"
        "    anom.to_csv('risk_profiled.csv', index=False)\n"
        "    print('Risk distribution:')\n"
        "    print(anom['risk_level'].value_counts().to_string())\n"
        "    print('DONE: risk_profiled.csv saved')\n"
    )


def _build_report_code():
    base_url = LM_STUDIO_BASE_URL
    api_key = LM_STUDIO_API_KEY
    model = LM_STUDIO_MODEL

    return (
        "import pandas as pd, json\n"
        "from datetime import datetime\n"
        "from openai import OpenAI\n"
        "df = pd.read_csv('risk_profiled.csv')\n"
        "client = OpenAI(base_url='" + base_url + "', api_key='" + api_key + "')\n"
        "top = df.nlargest(5, 'z_score') if 'z_score' in df.columns else df.head(5)\n"
        "report = []\n"
        "for _, r in top.iterrows():\n"
        "    ctx = str(dict(route=r.get('route','?'), alarms=r.get('actual_alarms',0), baseline=round(r.get('rolling_mean_alarms',0),1), zscore=round(r.get('z_score',0),2), risk=r.get('risk_level','?')))\n"
        "    try:\n"
        "        resp = client.chat.completions.create(model='" + model + "', messages=[{'role':'system','content':'Explain this transit anomaly in 2 sentences.'},{'role':'user','content':ctx}], max_tokens=100, temperature=0.3)\n"
        "        expl = resp.choices[0].message.content.strip()\n"
        "    except Exception as e:\n"
        "        expl = str(e)\n"
        "    report.append({'ctx': ctx, 'explanation': expl})\n"
        "txt = 'TRANSIT ANOMALY REPORT — ' + datetime.now().strftime('%Y-%m-%d') + '\\n'\n"
        "txt += 'Total anomalies: ' + str(len(df)) + '\\n\\n'\n"
        "for i, item in enumerate(report):\n"
        "    txt += '--- #' + str(i+1) + ' ---\\n' + item['ctx'] + '\\n' + item['explanation'] + '\\n\\n'\n"
        "with open('anomaly_report.txt','w') as f:\n"
        "    f.write(txt)\n"
        "with open('anomaly_report.json','w') as f:\n"
        "    json.dump(report, f, indent=2)\n"
        "print(txt)\n"
        "print('DONE: anomaly_report.txt saved')\n"
    )


# ── Build prompts ────────────────────────────────────────────────────────────
def _make_prompt(code: str) -> str:
    return (
        "Execute this code using the python_repl tool. "
        "Do NOT modify it. Just call python_repl with the code below.\n\n"
        + code
    )


DATA_PROMPT = _make_prompt(_build_data_code())
BASELINE_PROMPT = _make_prompt(_build_baseline_code())
OUTLIER_PROMPT = _make_prompt(_build_outlier_code())
RISK_PROMPT = _make_prompt(_build_risk_code())
REPORT_PROMPT = _make_prompt(_build_report_code())

TASKS = [
    ("data_loading",       DATA_PROMPT),
    ("baseline_building",  BASELINE_PROMPT),
    ("outlier_detection",  OUTLIER_PROMPT),
    ("risk_profiling",     RISK_PROMPT),
    ("report_generation",  REPORT_PROMPT),
]
