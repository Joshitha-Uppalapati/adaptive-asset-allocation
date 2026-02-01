import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from src.bandit import GaussianThompsonSampler
from src.metrics import cumulative_return, max_drawdown, sharpe_ratio


DB_PATH = "data/prices.db"


def load_prices() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().dropna()


def run_equal_weight_baseline(
    returns: pd.DataFrame, initial_capital: float
) -> pd.DataFrame:
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


def run_simulation(initial_capital: float = 10000.0):
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

    bandit_metrics = {
        "cumulative_return": float(cumulative_return(results["capital"])),
        "max_drawdown": float(max_drawdown(results["capital"])),
        "sharpe_ratio": float(sharpe_ratio(daily_returns)),
    }

    baseline = run_equal_weight_baseline(returns, initial_capital)

    return results, bandit_metrics, baseline

def save_comparison_plot(results, baseline):
    plt.figure(figsize=(10, 6))

    plt.plot(
        results["date"],
        results["capital"],
        label="Gaussian Thompson Sampling",
        linewidth=2,
    )

    plt.plot(
        baseline["date"],
        baseline["capital"],
        label="Equal-Weight Baseline",
        linestyle="--",
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Adaptive Asset Allocation vs Baseline")
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/portfolio_value.png")
    plt.close()

def main():
    results, bandit_metrics, baseline = run_simulation()

    print("\nBandit strategy metrics:")
    for k, v in bandit_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(
        f"\nBaseline final capital: {baseline['capital'].iloc[-1]:.2f}"
    )
    print(
        f"Bandit final capital:   {results['capital'].iloc[-1]:.2f}"
    )


if __name__ == "__main__":
    results, bandit_summary, baseline = run_simulation()

    print("\nBandit strategy metrics:")
    for k, v in bandit_summary.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nBaseline final capital: {baseline['capital'].iloc[-1]:.2f}")
    print(f"Bandit final capital:   {results['capital'].iloc[-1]:.2f}")

    save_comparison_plot(results, baseline)