import pandas as pd
import numpy as np
import cvxpy as cvx


def generate(
    mu: pd.DataFrame,
    Q: pd.DataFrame,
    risk_aversion: float,
    targetRet: float,
    shortable: bool,
) -> pd.Series:
    num_asset = len(mu)

    P = risk_aversion * Q
    q = -1 * mu
    q = np.zeros(len(mu))

    # upper and lower bound
    b = np.hstack((np.zeros(num_asset), targetRet))
    A = np.vstack((np.eye(num_asset), mu))

    # sum of weights = 1
    Aeq = np.ones(num_asset)
    beq = 1

    x = cvx.Variable(num_asset)
    prob = cvx.Problem(
        cvx.Minimize((1 / 2) * cvx.quad_form(x, P) + q.T @ x),
        [A @ x >= b, Aeq @ x == beq],
    )
    prob.solve()

    new_weights = pd.Series(x.value, index=mu.index)
    return new_weights


def test():
    # make sure generate is working with a tiny example
    pass


if __name__ == "__main__":
    test()
