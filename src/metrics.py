import numpy as np
import pandas as pd


def cumulative_return(capital_series: pd.Series):
    return float(capital_series.iloc[-1] / capital_series.iloc[0] - 1)


def max_drawdown(capital_series: pd.Series):
    running_max = capital_series.cummax()
    drawdown = (capital_series - running_max) / running_max
    return float(drawdown.min())


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate=0.0):
    excess_returns = daily_returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())