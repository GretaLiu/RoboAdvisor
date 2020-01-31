import pandas as pd
import numpy as np
import cvxpy as cvx
from numpy.linalg import inv
from scipy.stats.mstats import gmean
from scipy.linalg import cholesky, block_diag
from statsmodels.stats.moment_helpers import cov2corr
from scipy.optimize import linprog


def generate(
    mu: pd.Series, Q: pd.DataFrame, nPaths: int = 100, repeat: int = 250, T: int = 6,
):
    total = np.zeros(len(mu))
    num_asset = len(mu)
    rho = cov2corr(Q)
    L = cholesky(rho, lower=True)
    dt = T
    confidence_level = 0.95
    variances = np.diag(Q)
    f = np.zeros(num_asset + nPaths + 1)
    f[:nPaths] = 1 / ((1 - confidence_level) * nPaths)
    f[nPaths : num_asset + nPaths] = 0
    f[-1] = 1
    A = np.array(
        [[0.0 for k in range(nPaths + num_asset + 1)] for j in range(2 * nPaths)]
    )

    A[:nPaths, :nPaths] = -1 * np.eye(nPaths)
    A[nPaths : (2 * nPaths), :nPaths] = -1 * np.eye(nPaths)
    A[nPaths : (2 * nPaths), -1] = -1

    Aeq = np.array([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(1)])
    Aeq[0, nPaths : (nPaths + num_asset)] = 1

    beq = np.array([1])

    b = np.array([0.0 for k in range(3 * nPaths + num_asset + 1)])
    b[(2 * nPaths) : (3 * nPaths)] = 1000000000000
    b[(3 * nPaths) : num_asset + 3 * nPaths] = 0
    b[-1] = 1000000000000

    temp = -1 * np.eye(nPaths + num_asset + 1)

    exp_term_1 = ((mu.to_numpy() - 0.5 * variances) * dt).reshape(-1, 1)
    exp_term_2 = np.sqrt(variances * dt).reshape(-1, 1)

    for i in range(repeat):
        S = np.zeros((num_asset, 2, nPaths))
        S[:, 0, :] = 100

        xi = np.dot(L, np.random.randn(num_asset, nPaths))
        S[:, 1, :] = S[:, 0, :] * np.exp(exp_term_1 + exp_term_2 * xi)

        # returns_sample n_asset * nPeriod * nPaths
        returns_sample = S[:, -1, :] / S[:, 0, :] - 1

        for i in range((nPaths), (2 * nPaths)):
            A[i, nPaths : (nPaths + num_asset)] = -returns_sample[:, i - nPaths]

        A_ub = np.concatenate((A, temp), axis=0)

        res = linprog(
            c=f, A_ub=A_ub, b_ub=b, A_eq=Aeq, b_eq=beq, method="interior-point"
        )
        total = np.add(total, res.x[nPaths : nPaths + num_asset])

    return pd.Series(total / repeat, index=mu.index)


def test():
    # make sure generate is working with a tiny example
    prices_org = pd.read_csv("adjClose.csv")
    factor_org = pd.read_csv("factors.csv")

    # make sure fama-french factors and prices have the same length of datas
    prices_org["Date"] = prices_org["date"].apply(lambda x: x[:7])
    prices = pd.merge(prices_org, factor_org["Date"], on="Date", how="inner")

    # save dates
    dates = pd.to_datetime(prices["date"])
    dates = dates[1:]

    # remove dates in data
    prices = prices.drop(["date", "Date"], axis=1)
    factors = factor_org.drop("Date", axis=1)

    test_yr = 12
    prices = prices[:-test_yr]
    factors = factors[:-test_yr]
    test_prices = prices[-test_yr:]

    # organize data into riskfree, factors, and returns
    factors = factors[1:]
    factors = factors.apply(lambda x: x / 100)
    riskfree = factors.loc[:, "RF"]
    riskfree = riskfree.reset_index(drop=True)
    factors = factors.drop("RF", axis=1)
    returns = prices[:-1] / prices[1:].values - 1
    returns = returns.apply(lambda x: x - riskfree)
    pass


if __name__ == "__main__":
    test()
