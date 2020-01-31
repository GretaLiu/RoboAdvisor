from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from datetime import datetime
from investment_profile import (
    WealthInvestment,
    RetirementInvestment,
    PurchaseInvestment,
)
import pandas as pd
from uuid import uuid4


@dataclass
class Portfolio:
    id: str
    user_id: str
    start_date: datetime
    shares: Dict[str, float]
    initial_amount: float
    investment_profile: Union[
        WealthInvestment, RetirementInvestment, PurchaseInvestment
    ]

    @staticmethod
    def build(
        email: str,
        weights,
        prices: pd.DataFrame,
        initial_amount: float,
        investment_profile,
    ):
        weights_s = pd.Series(weights)
        shares = weights_s * initial_amount / prices[weights_s.index].iloc[-1, :]
        return Portfolio(
            id=uuid4(),
            user_id=email,
            start_date=datetime.now(),
            shares=shares.to_dict(),
            investment_profile=investment_profile,
            initial_amount=initial_amount,
        )


@dataclass
class PortfolioShareChange:
    portfolio_id: str
    timestamp: datetime
    shares_delta: Dict[str, float]
    reason: str


def weights_to_shares(weights, initial_amount, prices: pd.DataFrame):
    weights_s = pd.Series(weights)
    shares = weights_s * initial_amount / prices[weights.index].iloc[-1, :]
    return shares
