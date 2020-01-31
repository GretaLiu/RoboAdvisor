from pandas import DataFrame, Series
import numpy as np
import pandas as pd


def drawdown(portfolio_values: pd.Series):
    max_from_left = np.maximum.accumulate(portfolio_values.values)
    min_from_right = np.minimum.accumulate(portfolio_values.values[::-1])[::-1]
    diff = max_from_left - min_from_right
    i_max_drawdown = np.argmax(diff).item()
    start, end = i_max_drawdown, i_max_drawdown
    while start > 1 and max_from_left[start] == max_from_left[start - 1]:
        start -= 1
    while (
        end < len(min_from_right) - 1 and min_from_right[end] == min_from_right[end + 1]
    ):
        end += 1
    return diff[i_max_drawdown], end - start


def beta_to_mkt(portfolio_returns: pd.Series, benchmark_returns: pd.Series):
    # code written here to measure the beta to market of the portfolio
    # assume that portfolio returns is a DataFrame with "Date" and porfolio returns
    portfolio_returns.reset_index(drop=True)  # drop "Date" column
    benchmark_returns.reset_index(drop=True)  # drop "Date" column
    assert len(portfolio_returns) == len(
        benchmark_returns
    ), "Dimension mismatch"  # length needs to match
    portfolio_var = np.var(portfolio_returns)
    portfolio_benchmark_covar = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    beta = portfolio_benchmark_covar / portfolio_var
    return beta


def treynor_top(portfolio_returns: pd.Series, factors: pd.DataFrame):
    rf = factors.loc[:, "rf"] / 100
    diff = portfolio_returns.mean() - rf.mean()
    return diff


def sharpe_ratio(portfolio_returns: pd.Series):
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()
    return sharpe_ratio


def jensens_alpha(
    historical_prices: DataFrame,
    weights: Series,
    factors: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
):
    # input is the prices of all the stocks invested within the period
    # weight is the portfolio weights at the beginning of the period
    factors = factors[1:]
    factors = factors.apply(lambda x: x / 100)
    riskfree = factors.loc[:, "rf"]
    riskfree = riskfree.reset_index(drop=True)
    portfolio_returns.drop("Date", axis=1)  # drop "Date" column
    benchmark_returns.drop("Date", axis=1)  # drop "Date" column
    assert len(portfolio_returns) == len(
        riskfree
    ), "Dimension mismatch"  # length needs to match
    portfolio_rerturns = portfolio_returns.iloc[
        :, 0
    ].values  # cast the portfolio returns into a 1d array
    benchmark_returns = benchmark_returns.iloc[
        :, 0
    ].values  # cast the benchmark returns into a 1d array
    rf = rf.iloc[:, 0].values  # cast the riskfree returns into a 1d array
    portfolio_var = np.var(portfolio_rerturns)
    portfolio_benchmark_covar = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    beta = portfolio_benchmark_covar / portfolio_var
    returns = historical_prices / historical_prices.iloc[0]

    for i in range(len(weights)):
        returns[returns.columns[i]] = returns[returns.columns[i]].multiply(weights[i])
    returns = returns * 100000
    # add a total portfolio column
    returns["Total"] = returns.sum(axis=1)

    # Daily Return
    returns["Daily Return"] = returns["Total"].pct_change(1)

    # jensens_alpha
    ja = portfolio_val["Daily Return"].mean() - (
        riskfree.mean() + beta * (benchmark_returns - riskfree.mean())
    )

    # Annual jensens_alpha
    Aja = (252 ** 0.5) * ja

    return sharpe_ratio, ASR
