import sqlite3
import yaml
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd


DB_PATH = "data/prices.db"


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_date_range(lookback_days: int):
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_prices(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False
    )

    # yfinance returns multi-index columns when multiple tickers are used
    close_prices = data["Close"]

    # Forward fill missing values (holidays, partial trading days)
    close_prices = close_prices.ffill()

    return close_prices


def save_to_db(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)

    df.reset_index(inplace=True)
    df.to_sql(
        name="prices",
        con=conn,
        if_exists="replace",
        index=False
    )

    conn.close()


def main():
    config = load_config()
    assets = config["assets"]
    lookback = config["lookback_period"]

    start_date, end_date = get_date_range(lookback)

    prices = fetch_prices(
        tickers=assets,
        start_date=start_date,
        end_date=end_date
    )

    save_to_db(prices)
    print(f"Saved price data for {len(assets)} assets")


if __name__ == "__main__":
    main()
