import sqlite3
import pandas as pd

from bandit import GaussianThompsonSampler

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

    return pd.DataFrame(history)


if __name__ == "__main__":
    results = run_simulation()
    print(results.tail())
