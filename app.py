from flask import Flask, escape, request
from flask_cors import CORS
from investment_profile import build_profile, AVERSION
import invest
from datetime import datetime
from dataclasses import asdict
from portfolio.data import fetch, db_api
from portfolio.parameter import carhart_jing
from portfolio.analytics import portfolio_analytics
from portfolio.model import mvo_jing, cvar_cong
from portfolio.analytics.measure import portfolio_performance, scenarios
import os

tickers = ["AAPL", "GOOGL", "MSFT", "SPY", "EEM", "XLF", "XLE"]

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    name = request.args.get("name", "World")
    return f"Hello, {escape(name)}!"


@app.route("/list/<string:email>", methods=["GET"])
def list_portfolios(email: str):
    prices = db_api.get_historical_price().fillna(method="bfill")
    portfolios = db_api.load_portfolios(email, prices.iloc[-1, :])
    return portfolios.to_json()


def cap_weights(weights):
    significant_weights = weights.sort_values()[-10:]
    scaling_factor = 1 / significant_weights.sum()
    return significant_weights * scaling_factor


@app.route("/generate", methods=["POST"])
def generate_portfolio():
    profile = request.get_json()
    investment_profile = build_profile(profile)
    factors = db_api.get_factors()
    prices = db_api.get_historical_price().fillna(method="bfill")
    mu, Q = carhart_jing.estimate(prices.iloc[:12, :], factors.iloc[:12, :])
    weights = cvar_cong.generate(mu, Q)
    weights = cap_weights(weights)
    # historical performance
    portfolio_values = portfolio_performance(weights, prices)
    # initial investment
    price_years = (prices.index[-1] - prices.index[0]).days / 365
    # 3 cases
    worse_case, expected_case, better_case = scenarios(
        portfolio_values, investment_profile.get_measurement_years()
    )
    initial_amount = investment_profile.calc_initial(expected_case)
    scaled_portfolio_values = portfolio_values * initial_amount
    portfolio_returns = (
        scaled_portfolio_values[1:] / scaled_portfolio_values[:-1].values - 1
    )
    spy = prices["SPY"] * initial_amount / prices["SPY"][0]
    spy_returns = spy[1:] / spy[:-1].values - 1
    max_drawdown, drawdown_days = portfolio_analytics.drawdown(scaled_portfolio_values)
    beta = portfolio_analytics.beta_to_mkt(portfolio_returns, spy_returns)
    treynor = (
        portfolio_analytics.treynor_top(portfolio_returns * 12, factors * 12)
    ) / beta
    sharpe = portfolio_analytics.sharpe_ratio((portfolio_returns * 12))

    return {
        "portfolio_values": (portfolio_values * initial_amount).to_list(),
        # pandas timestamp is accurate to ns, but javascript reads only up to ms
        "date": [d.value // 1000000 for d in portfolio_values.index],
        "weights": weights.to_dict(),
        "cases": [worse_case, expected_case, better_case],
        "benchmark": spy.to_list(),
        "investment_profile": asdict(investment_profile),
        "max_drawdown": max_drawdown,
        "drawdown_days": drawdown_days,
        "beta": beta,
        "treynor": treynor,
        "sharpe": sharpe,
    }


@app.route("/confirm", methods=["POST"])
def confirm_portfolio():
    body = request.get_json()
    prices = db_api.get_historical_price().fillna(method="bfill")
    portfolio = invest.Portfolio.build(
        email=body["email"],
        weights=body["weights"],
        prices=prices,
        initial_amount=body["initial_amount"],
        investment_profile=body["investment_profile"],
    )
    db_api.save_portfolio(asdict(portfolio))
    return asdict(portfolio)


if __name__ == "__main__":
    app.run("0.0.0.0", os.environ["PORT"] or 5000)

