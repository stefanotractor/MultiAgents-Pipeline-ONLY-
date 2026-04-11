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
    allarmi = repr(ALLARMI_CSV)
    tipologia = repr(TIPOLOGIA_CSV)

    return (
            "You have two CSV files to load, clean, and merge:\n"
            "- " + allarmi + "\n"
            "- " + tipologia + "\n\n"
            "Do the following in a SINGLE python_repl call:\n\n"
            "1. Load both CSVs with pandas\n"
            "2. Standardize ALL column names to lowercase snake_case\n"
            "3. Remove duplicate columns (same data, different name)\n"
            "4. For EVERY column, detect and fix:\n"
            "   - Missing values: NaN, 'N.D.', '?', '-', '//', empty strings, spaces\n"
            "   - Numeric columns stored as strings: strip non-numeric chars, convert to int/float, fill NaN with 0\n"
            "   - Date columns: parse to datetime, handle mixed formats\n"
            "   - Categorical strings: strip whitespace, lowercase everything for consistency\n"
            "   - Typos and inconsistencies: e.g. 2-letter country codes to 3-letter (IT->ITA, GB->GBR)\n"
            "5. Merge the two DataFrames on shared key columns using outer join\n"
            "6. Save result to 'merged_data.csv'\n"
            "7. Print shape, dtypes, null counts, and first 3 rows\n"
            "\n"
            "IMPORTANT: The output merged_data.csv MUST contain these columns "
            "(with exactly these names in lowercase):\n"
            "areoporto_arrivo, areoporto_partenza, data_partenza, anno_partenza, "
            "mese_partenza, allarmati, nazionalita, tipo_documento, genere, "
            "fascia_eta, paese_part, codice_paese_arr, codice_paese_part, "
            "compagnia_aerea, numero_volo, esito_controllo, tot, entrati, investigati\n"
            "Do NOT rename these columns to anything else.\n"
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
