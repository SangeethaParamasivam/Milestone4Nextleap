"""§8.5 gate driver. Run with: .\.venv\Scripts\python.exe scratch/try_retrieval.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import Decision
from app.retriever import retrieve
from config import load_config

cfg = load_config()
for q in [
    "What is the expense ratio of HDFC Large Cap Fund - Direct Growth?",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "What is the expense ratio of Parag Parflex Fund?",
    "Who won the 2024 cricket world cup?",
]:
    r = retrieve(q, Decision("FACTUAL", None, "manual"), cfg)
    print(q, "->", r.status, [(h.chunk.scheme_name, h.chunk.topic, round(h.score, 4)) for h in r.hits])
