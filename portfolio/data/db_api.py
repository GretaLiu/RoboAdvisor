import sqlalchemy as db
import pandas as pd
import numpy as np

engine = db.create_engine(
    "postgresql+psycopg2://nzehvkdfogjlxh:4dd1f72a746963e155f5331796005be87d6750164d1698820cc561987beac08d@ec2-54-225-115-177.compute-1.amazonaws.com:5432/de82kq7ni8ngs9"
)
# always share one connection? for now
conn = engine.connect()


def get_historical_price():
    return pd.read_sql_table("stock_price", conn, index_col='date')


def get_factors():
    return pd.read_sql_table("FF_factors", conn, index_col='date')


def save_portfolio(portfolio):
    p_series = pd.Series(portfolio)
    p_df = pd.DataFrame([p_series]).set_index("id")
    p_df.to_sql(
        "portfolio",
        conn,
        if_exists="append",
        dtype={"shares": db.types.JSON, "investment_profile": db.types.JSON},
    )


def load_portfolios(email: str, curr_price: pd.Series) -> pd.DataFrame:
    portfolios = pd.read_sql_query(
        "SELECT * FROM portfolio where user_id=%(email)s",
        conn,
        params={"email": email},
        index_col="id",
    )
    portfolios["name"] = portfolios["investment_profile"].map(
        lambda p: p.get("name", p["purpose"])
    )
    portfolios["value"] = portfolios["shares"].map(
        lambda shares: (pd.Series(shares) * curr_price).sum()
    )
    portfolios["maturity"] = portfolios["investment_profile"].map(
        lambda p: p.get("years", np.nan)
    )
    # portfolios["returns"] = portfolios["value"] / portfolios["initial_amount"]
    return portfolios
