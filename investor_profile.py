"""
investor_profile.py
--------------------
Captures and validates investor profile inputs.
Maps risk tolerance to portfolio parameters that drive downstream decisions:
  - Position sizing
  - Stop-loss width
  - Allocation per stock
  - Preferred strategy type
"""

from dataclasses import dataclass, field
from typing import Literal


RISK_PROFILES = {
    'low': {
        'label': 'Conservative',
        'max_stock_allocation_pct': 5,    # Max 5% of capital in any single stock
        'sl_multiplier': 1.5,             # Tight stop loss
        'risk_per_trade_pct': 0.5,        # Risk 0.5% of capital per trade
        'preferred_strategies': ['dividend', 'blue_chip', 'index'],
        'min_rr_ratio': 3.0,              # Minimum Risk:Reward ratio required
        'max_pe': 25,                     # Only buy stocks with P/E < 25
        'description': 'Capital preservation. Slow and steady growth. '
                       'Favour large-cap, dividend-paying stocks.'
    },
    'medium': {
        'label': 'Balanced',
        'max_stock_allocation_pct': 10,
        'sl_multiplier': 2.0,
        'risk_per_trade_pct': 1.0,
        'preferred_strategies': ['growth', 'swing_trade', 'momentum'],
        'min_rr_ratio': 2.0,
        'max_pe': 40,
        'description': 'Balanced growth and risk. Mix of growth stocks '
                       'and stable assets. Suitable for 3-5 year horizons.'
    },
    'high': {
        'label': 'Aggressive',
        'max_stock_allocation_pct': 20,
        'sl_multiplier': 3.0,
        'risk_per_trade_pct': 2.0,
        'preferred_strategies': ['momentum', 'breakout', 'smallcap', 'options'],
        'min_rr_ratio': 1.5,
        'max_pe': 60,
        'description': 'Maximum growth. Accepts large drawdowns. '
                       'Small/mid-cap and high-momentum stocks. Long horizon required.'
    }
}

INVESTMENT_HORIZON_STRATEGY = {
    (0, 1):   'short_term_trading',    # <1 year
    (1, 3):   'medium_term_growth',   # 1–3 years
    (3, 7):   'long_term_growth',     # 3–7 years
    (7, 999): 'wealth_compounding',   # 7+ years
}


@dataclass
class InvestorProfile:
    """
    Immutable investor profile data class.

    Attributes:
        name: Investor's name
        investment_amount: Total capital to invest (in INR or USD)
        risk_tolerance: 'low' | 'medium' | 'high'
        investment_horizon_years: How long to stay invested
        target_return_pct: Annual return target (optional)
        currency: Currency symbol for display
    """
    name: str
    investment_amount: float
    risk_tolerance: Literal['low', 'medium', 'high']
    investment_horizon_years: float
    target_return_pct: float = 12.0
    currency: str = '₹'

    def __post_init__(self):
        if self.risk_tolerance not in RISK_PROFILES:
            raise ValueError(f"risk_tolerance must be one of {list(RISK_PROFILES.keys())}")
        if self.investment_amount <= 0:
            raise ValueError("investment_amount must be positive.")
        if self.investment_horizon_years <= 0:
            raise ValueError("investment_horizon_years must be positive.")

    @property
    def profile_params(self) -> dict:
        """Return the risk-profile parameter set."""
        return RISK_PROFILES[self.risk_tolerance]

    @property
    def strategy(self) -> str:
        """Infer the appropriate strategy based on investment horizon."""
        for (low, high), strat in INVESTMENT_HORIZON_STRATEGY.items():
            if low <= self.investment_horizon_years < high:
                return strat
        return 'wealth_compounding'

    @property
    def max_single_stock_investment(self) -> float:
        """Maximum amount to allocate to a single stock."""
        pct = self.profile_params['max_stock_allocation_pct']
        return self.investment_amount * (pct / 100)

    @property
    def risk_per_trade(self) -> float:
        """Maximum capital to risk in a single trade."""
        pct = self.profile_params['risk_per_trade_pct']
        return self.investment_amount * (pct / 100)

    def summary(self) -> str:
        """Human-readable profile summary."""
        p = self.profile_params
        lines = [
            "╔══════════════════════════════════════╗",
            "  INVESTOR PROFILE",
            "══════════════════════════════════════",
            f"  Name            : {self.name}",
            f"  Capital         : {self.currency}{self.investment_amount:,.0f}",
            f"  Risk Tolerance  : {p['label']} ({self.risk_tolerance.upper()})",
            f"  Horizon         : {self.investment_horizon_years} years",
            f"  Strategy        : {self.strategy.replace('_', ' ').title()}",
            f"  Max per Stock   : {self.currency}{self.max_single_stock_investment:,.0f} ({p['max_stock_allocation_pct']}%)",
            f"  Risk per Trade  : {self.currency}{self.risk_per_trade:,.0f} ({p['risk_per_trade_pct']}%)",
            f"  Target Return   : {self.target_return_pct}% p.a.",
            f"  SL Multiplier   : {p['sl_multiplier']}× ATR",
            f"  Min R:R Ratio   : {p['min_rr_ratio']}:1",
            "",
            f"  {p['description']}",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


def collect_investor_profile_interactive() -> InvestorProfile:
    """
    Interactively collect investor profile from the command line.

    Returns:
        InvestorProfile instance
    """
    print("\n" + "═" * 40)
    print("  AI STOCK ADVISOR — INVESTOR SETUP")
    print("═" * 40)

    name = input("  Your name: ").strip() or "Investor"

    while True:
        try:
            amount = float(input("  Investment amount (₹): ").replace(',', ''))
            if amount > 0:
                break
            print("  Please enter a positive amount.")
        except ValueError:
            print("  Invalid input. Enter a number.")

    while True:
        risk = input("  Risk tolerance [low / medium / high]: ").strip().lower()
        if risk in RISK_PROFILES:
            break
        print("  Choose from: low, medium, high")

    while True:
        try:
            horizon = float(input("  Investment horizon (years): "))
            if horizon > 0:
                break
        except ValueError:
            pass
        print("  Please enter a positive number.")

    while True:
        try:
            target = input("  Target annual return % [default 12]: ").strip()
            target = float(target) if target else 12.0
            break
        except ValueError:
            print("  Enter a number (e.g., 15).")

    profile = InvestorProfile(
        name=name,
        investment_amount=amount,
        risk_tolerance=risk,
        investment_horizon_years=horizon,
        target_return_pct=target
    )

    print(profile.summary())
    return profile


def create_default_profile(
    name: str = "Default Investor",
    amount: float = 100000,
    risk: str = "medium",
    horizon: float = 3.0
) -> InvestorProfile:
    """Create a profile with sensible defaults for testing/demo purposes."""
    return InvestorProfile(
        name=name,
        investment_amount=amount,
        risk_tolerance=risk,
        investment_horizon_years=horizon
    )
