#!/usr/bin/env python3
"""
Transit Anomaly Detection — Multi-Agent System (v2 Light)
Usage: python main.py [--verbose] [--algorithm IsolationForest|LOF|zscore]
"""
import argparse, os, sys
from datetime import datetime
from langchain_core.messages import HumanMessage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["IsolationForest","LOF","zscore"], default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--query", type=str, default=None,
        help="Pandas query filter, e.g. \"areoporto_partenza == 'FCO'\"")
    args = parser.parse_args()

    if args.algorithm:
        import config
        config.DEFAULT_OUTLIER_ALGORITHM = args.algorithm
        import importlib, prompts
        importlib.reload(prompts)

    from config import OUTPUT_DIR, LM_STUDIO_MODEL, DEFAULT_OUTLIER_ALGORITHM
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    if args.query:
        import pandas as pd, shutil
        from openai import OpenAI
        from config import ALLARMI_CSV, TIPOLOGIA_CSV, LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL

        # Load one CSV to get column names and sample values
        sample_df = pd.read_csv(ALLARMI_CSV, nrows=5)
        sample_df.columns = sample_df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('%', '_')
        cols = sample_df.columns.unique()
        col_info = {c: sample_df[c].head(3).tolist() if sample_df[c].ndim == 1 else sample_df[c].iloc[:, 0].head(3).tolist() for c in cols}
        # Ask LLM to translate natural language to pandas query
        client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
        system_msg = (
            "You translate user queries into pandas DataFrame .query() expressions. "
            "Reply with ONLY the query string, nothing else. No quotes around it. "
            "All column names are lowercase with underscores. "
            "Airport IATA codes: Fiumicino=FCO, Ciampino=CIA, Malpensa=MXP, Linate=LIN, "
            "Bergamo=BGY, Venezia=VCE, Bologna=BLQ, Napoli=NAP, Catania=CTA, Pisa=PSA, "
            "Bari=BRI, Palermo=PMO, Torino=TRN, Verona=VRN. "
            "Available columns and sample values: " + str(col_info)
        )
        resp = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": args.query}
            ],
            temperature=0.0, max_tokens=100,
        )
        pandas_query = resp.choices[0].message.content.strip()
        print(f"  User query: {args.query}")
        print(f"  LLM interpreted as: {pandas_query}")

        # Apply filter to copies of CSVs
        for csv_path in [ALLARMI_CSV, TIPOLOGIA_CSV]:
            dst = os.path.join(OUTPUT_DIR, os.path.basename(csv_path))
            shutil.copy2(csv_path, dst)
            df = pd.read_csv(dst)
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('%', '_')
            df = df.loc[:, ~df.columns.duplicated()]
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.strip()
            try:
                filtered = df.query(pandas_query)
                filtered.to_csv(dst, index=False)
                print(f"  Filtered {os.path.basename(csv_path)}: {len(filtered)} rows")
            except Exception as e:
                print(f"  Filter failed: {e}")
                print(f"  Keeping original data")
                df.to_csv(dst, index=False)

        import config
        config.ALLARMI_CSV = os.path.join(OUTPUT_DIR, "ALLARMI.csv")
        config.TIPOLOGIA_CSV = os.path.join(OUTPUT_DIR, "TIPOLOGIA_VIAGGIATORE.csv")
        import importlib, prompts
        importlib.reload(prompts)

    print("=" * 60)
    print("  TRANSIT ANOMALY DETECTION — MULTI-AGENT (v2)")
    print("=" * 60)
    print(f"  Time:       {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Model:      {LM_STUDIO_MODEL}")
    print(f"  Algorithm:  {DEFAULT_OUTLIER_ALGORITHM}")
    print(f"  Output:     {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    from agents import build_graph
    graph = build_graph()

    state = {
        "messages": [HumanMessage(content="Start pipeline.")],
        "current_task_index": 0,
        "task_status": "pending",
        "next": "supervisor",
        "retry_count": 0,
    }

    try:
        for event in graph.stream(state, {"recursion_limit": 50}):
            for node, data in event.items():
                if node == "__end__":
                    continue
                msgs = data.get("messages", [])
                if msgs:
                    content = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
                    if args.verbose:
                        print(f"\n{'='*50}")
                        print(f"  [{node.upper()}]")
                        print(f"{'='*50}")
                        # Truncate for readability
                        if len(content) > 1000:
                            content = content[:1000] + "\n... [truncated]"
                        print(content)
                    else:
                        line = content.split("\n")[0][:80]
                        print(f"  [{node}] {line}")
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    for f in ["merged_data.csv","baseline_data.csv","outlier_results.csv",
              "risk_profiled.csv","anomaly_report.txt","anomaly_report.json"]:
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(p):
            print(f"  ✓ {f} ({os.path.getsize(p):,} bytes)")
        else:
            print(f"  ✗ {f}")

    report = os.path.join(OUTPUT_DIR, "anomaly_report.txt")
    if os.path.exists(report):
        print("\n" + "=" * 60)
        print("  REPORT")
        print("=" * 60)
        with open(report) as f:
            print(f.read())


if __name__ == "__main__":
    main()
