import pandas as pd


def estimate(prices: pd.DataFrame, factors: pd.DataFrame):
    # Modify code here
    mu = None
    Q = None
    return mu, Q


def test():
    """run this function to test"""
    prices = pd.read_csv("../data/adjClose.csv", index_col=["date"])
    factors = pd.read_csv("../data/factors.csv", index_col=["Date"])
    mu, Q = estimate(prices, factors)
    # check one example
    assert mu == []
    assert Q == [[]]


if __name__ == "__main__":
    test()
