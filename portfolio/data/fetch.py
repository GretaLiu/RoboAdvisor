from tiingo import TiingoClient
from typing import List
from pandas import DataFrame, concat
from pandas_datareader import famafrench
import os
import datetime

start_date = datetime.datetime.now() - datetime.timedelta(days=13 * 365)

dir_path = os.path.dirname(os.path.realpath(__file__))

client = TiingoClient(
    {"session": True, "api_key": "9ca5dfd053eb30abcb2610069a54cacd2afb4763"}
)


def get_historical_price(
    tickers: List[str], startDate: str = start_date, frequency: str = "monthly"
):
    assert tickers  # make sure there are at least one ticker
    print("Getting prices for", " ".join(tickers))
    df = client.get_dataframe(
        tickers, metric_name="adjClose", startDate=startDate, frequency=frequency
    )
    return df


def get_factors(startDate: str = start_date):
    dataset_names = famafrench.get_available_datasets()
    print("Getting Fama French 3 factors")
    factors = famafrench.FamaFrenchReader("F-F_Research_Data_Factors", start=startDate)
    ff3factors: DataFrame = factors.read()[0]
    print("Getting Fama French momentum factor")
    momentum = famafrench.FamaFrenchReader("F-F_Momentum_Factor", start=startDate)
    momfactors: DataFrame = momentum.read()[0]
    together = concat([ff3factors, momfactors], axis=1)
    return together


if __name__ == "__main__":
    get_factors().to_csv(os.path.join(dir_path, "factors.csv"))
    get_historical_price(["AAPL", "GOOGL", "MSFT", "SPY", "EEM", "XLF", "XLE"]).to_csv(
        os.path.join(dir_path, "adjClose.csv")
    )
