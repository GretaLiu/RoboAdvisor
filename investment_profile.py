from dataclasses import dataclass
from datetime import timedelta, datetime


class PURPOSES:
    large_purchase = "A future large purchase"
    retirement = "Retirement"
    building_wealth = "Building wealth"


class RISKTOLERANCE:
    conservative = "Conservative"
    moderate = "Moderate"
    risky = "Risky"


AVERSION = {
    RISKTOLERANCE.conservative: 1,
    RISKTOLERANCE.moderate: 0.5,
    RISKTOLERANCE.risky: 0,
}


@dataclass
class PurchaseInvestment:
    purpose: str
    name: str
    years: int
    amount: float
    risk_tolerance: str

    def calc_initial(self, exp_total_return):
        return self.amount / exp_total_return

    def get_measurement_years(self):
        return 5


@dataclass
class RetirementInvestment:
    purpose: str
    years: int
    amount: float
    risk_tolerance: str

    def calc_initial(self, exp_total_return):
        return self.amount / exp_total_return

    def get_measurement_years(self):
        return 5


@dataclass
class WealthInvestment:
    purpose: str
    amount: float
    risk_tolerance: str

    def calc_initial(self, exp_total_return):
        return self.amount / exp_total_return

    # the years used to measure performance, not real
    def get_measurement_years(self):
        return 5


def build_profile(profile_dict):
    if profile_dict["purpose"] == PURPOSES.large_purchase:
        return PurchaseInvestment(
            purpose=profile_dict["purpose"],
            name=profile_dict["name"],
            years=profile_dict["years"],
            amount=profile_dict["amount"],
            risk_tolerance=profile_dict["risk_tolerance"],
        )
    elif profile_dict["purpose"] == PURPOSES.retirement:
        return RetirementInvestment(
            purpose=profile_dict["purpose"],
            years=profile_dict["years"],
            amount=profile_dict["amount"],
            risk_tolerance=profile_dict["risk_tolerance"],
        )
    elif profile_dict["purpose"] == PURPOSES.building_wealth:
        return WealthInvestment(
            purpose=profile_dict["purpose"],
            amount=profile_dict["amount"],
            risk_tolerance=profile_dict["risk_tolerance"],
        )
