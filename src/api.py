import sqlite3
import pandas as pd
from fastapi import FastAPI

from src.bandit import GaussianThompsonSampler


DB_PATH = "data/prices.db"

app = FastAPI(title="Adaptive Asset Allocation API")


def load_assets():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices LIMIT 1", conn)
    conn.close()

    return [c for c in df.columns if c != "Date"]


assets = load_assets()
bandit = GaussianThompsonSampler(assets)


@app.get("/allocation")
def get_allocation():
    samples = bandit.sample()

    total = sum(max(v, 0) for v in samples.values())
    if total == 0:
        weights = {a: 1 / len(assets) for a in assets}
    else:
        weights = {a: max(v, 0) / total for a, v in samples.items()}

    return {
        "weights": weights
    }