"""
Configuration for Transit Anomaly Detection — Multi-Agent System (v2 Light).
"""
import os

# ── LM Studio ────────────────────────────────────────────────────────────────
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "google/gemma-3-4b"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ALLARMI_CSV = os.path.join(DATA_DIR, "ALLARMI.csv")
TIPOLOGIA_CSV = os.path.join(DATA_DIR, "TIPOLOGIA_VIAGGIATORE.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ── Outlier Detection ────────────────────────────────────────────────────────
DEFAULT_OUTLIER_ALGORITHM = "IsolationForest"
ISOLATION_FOREST_CONTAMINATION = 0.05
LOF_N_NEIGHBORS = 20
ZSCORE_THRESHOLD = 3.0

# ── Risk Thresholds ──────────────────────────────────────────────────────────
ALERT_RATE_MULTIPLIER = 3.0
