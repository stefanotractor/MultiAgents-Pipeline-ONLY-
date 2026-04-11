# Transit Anomaly Detection — Multi-Agent System (v2 Light)

## Cosa cambia rispetto a v1

La v1 chiedeva al modello 7B di **scrivere codice da zero** a partire da prompt
descrittivi. Il modello non era abbastanza capace → loop infiniti, niente file creati.

La v2 usa un approccio **ibrido**: i prompt contengono il codice Python pronto,
e il compito dell'agente code_executor è semplicemente **chiamare il tool python_repl**
con quel codice. L'architettura multi-agente (supervisor → code_executor → validator)
resta identica — cambia solo quanto lavoro deve fare l'LLM.

## Architettura

```
         ┌────────────┐
         │ SUPERVISOR  │  ← Routing logic (no LLM)
         └─────┬──────┘
               │
     ┌─────────┴─────────┐
     ▼                    ▼
┌──────────┐       ┌───────────┐
│ CODE     │       │ VALIDATOR │
│ EXECUTOR │       │           │
│ (REPL)   │       │ (check)   │
└──────────┘       └───────────┘
```

## Usage

```bash
pip install -r requirements.txt
python main.py --verbose
python main.py --algorithm LOF --verbose
```

## Pipeline (5 tasks)

1. Data Loading & Cleaning → `merged_data.csv`
2. Baseline Building → `baseline_data.csv`
3. Outlier Detection → `outlier_results.csv`
4. Risk Profiling → `risk_profiled.csv`
5. Report Generation → `anomaly_report.txt` + `.json`
