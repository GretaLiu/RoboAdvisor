import pandas as pd


def generate(
    mu: pd.DataFrame,
    Q: pd.DataFrame,
    risk_aversion: float,
    targetRet: float,
    shortable: bool,
    previous_portfolio_weights: pd.Series,
)->pd.Series:
    new_weights = pd.Series()
    return new_weights


def test():
    # make sure generate is working with a tiny example
    pass


if __name__ == "__main__":
    test()
