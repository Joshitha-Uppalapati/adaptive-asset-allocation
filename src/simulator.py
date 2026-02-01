import sqlite3
import pandas as pd

from src.bandit import GaussianThompsonSampler
from src.metrics import cumulative_return, max_drawdown, sharpe_ratio


DB_PATH = "data/prices.db"


def load_prices():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df


def compute_returns(price_df):
    return price_df.pct_change().dropna()


def run_equal_weight_baseline(returns, initial_capital):
    capital = initial_capital
    history = []

    for date, row in returns.iterrows():
        daily_return = row.mean()
        capital *= (1 + daily_return)

        history.append(
            {
                "date": date,
                "capital": capital,
            }
        )

    return pd.DataFrame(history)


def run_simulation(initial_capital=10000):
    prices = load_prices()
    returns = compute_returns(prices)

    assets = list(returns.columns)
    bandit = GaussianThompsonSampler(assets)

    capital = initial_capital
    history = []

    for date, row in returns.iterrows():
        chosen_asset = bandit.select_asset()
        daily_return = row[chosen_asset]

        capital *= (1 + daily_return)
        bandit.update(chosen_asset, daily_return)

        history.append(
            {
                "date": date,
                "asset": chosen_asset,
                "daily_return": daily_return,
                "capital": capital,
            }
        )

    results = pd.DataFrame(history)
    daily_returns = results["capital"].pct_change().dropna()

    bandit_summary = {
        "cumulative_return": cumulative_return(results["capital"]),
        "max_drawdown": max_drawdown(results["capital"]),
        "sharpe_ratio": sharpe_ratio(daily_returns),
    }

    baseline = run_equal_weight_baseline(returns, initial_capital)

    return results, bandit_summary, baseline


if __name__ == "__main__" and __package__ is None:
    raise RuntimeError(
        "Run this module with: python -m src.simulator"
    )
