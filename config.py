"""
Configuration for Transit Anomaly Detection — Multi-Agent System
"""
import os
from dotenv import load_dotenv

load_dotenv()

LM_STUDIO_BASE_URL = "https://api.groq.com/openai/v1"
LM_STUDIO_API_KEY  = os.getenv("GROQ_API_KEY", "")
LM_STUDIO_MODEL    = "llama-3.3-70b-versatile"

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
